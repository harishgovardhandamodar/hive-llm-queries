"""
Cluster Summary — generate narrative summaries and key takeaways from clusters.

Uses Ollama to produce:
  1. A 2-3 sentence narrative summary of what the cluster is about
  2. 3-5 key takeaways (bullet-style, one sentence each)
  3. 3 representative questions from the cluster
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from cluster_store import get_cluster, STORE_DIR

SUMMARY_URL = os.environ.get("LABEL_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

SUMMARY_PROMPT = """Analyze these Q&A pairs and return ONLY valid JSON.

Q&A:
{pairs}

Return ONLY this JSON (no other text, no markdown):
{"summary":"<2-3 sentence summary>","takeaways":["<key takeaway 1>","<key takeaway 2>","<key takeaway 3>"],"questions":["<representative question 1>","<representative question 2>","<representative question 3>"]}"""


def compute_distributions(items: list[dict]) -> dict:
    intents = {}
    topics = {}
    for item in items:
        intent = item.get("intent") or "other"
        intents[intent] = intents.get(intent, 0) + 1
        for t in item.get("topics") or []:
            if t:
                topics[t] = topics.get(t, 0) + 1
    sorted_intents = sorted(intents.items(), key=lambda kv: -kv[1])
    sorted_topics = sorted(topics.items(), key=lambda kv: -kv[1])
    total = len(items)
    return {
        "intents": [{"name": k, "count": v, "pct": round(v / total * 100)} for k, v in sorted_intents],
        "topics": [{"name": k, "count": v, "pct": round(v / total * 100)} for k, v in sorted_topics[:10]],
    }


def compute_concept_distribution(cluster_id: str) -> list[dict]:
    try:
        from cluster_kg import get_cluster_knowledge_graph
        kg = get_cluster_knowledge_graph(cluster_id)
        if not kg or not kg.get("nodes"):
            return []
        concepts = {}
        for n in kg["nodes"]:
            ct = n.get("concept_type") or n.get("type") or "concept"
            concepts[ct] = concepts.get(ct, 0) + (n.get("frequency") or 1)
        total = sum(concepts.values())
        return [{"name": k, "count": v, "pct": round(v / total * 100)} for k, v in sorted(concepts.items(), key=lambda kv: -kv[1])]
    except Exception:
        return []


def build_cluster_summary(cluster: dict, model: str = DEFAULT_MODEL) -> dict:
    items = cluster.get("items", [])
    cluster_id = cluster.get("id", "")
    if not items:
        return {
            "summary": "No Q&A pairs in this cluster.",
            "takeaways": [],
            "representative_questions": [],
            "distribution": {"intents": [], "topics": [], "concepts": []},
        }

    sample = items[:6]
    pairs_lines = []
    for i, item in enumerate(sample):
        q = (item.get("query") or "").strip()[:200]
        a = (item.get("answer") or "").strip()[:200]
        pairs_lines.append(f"Q{i+1}: {q}\nA{i+1}: {a}")

    pairs_text = "\n\n".join(pairs_lines)

    prompt = SUMMARY_PROMPT.replace("{pairs}", pairs_text)

    distribution = compute_distributions(items)
    concepts = compute_concept_distribution(cluster_id)

    try:
        resp = requests.post(
            SUMMARY_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        raw = (resp.json().get("message", {}) or {}).get("content", "").strip()
        if not raw:
            raise ValueError("empty response")
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
        first = raw.find("{")
        last = raw.rfind("}")
        if first >= 0 and last > first:
            raw = raw[first : last + 1]
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("response is not a dict")
        result.setdefault("takeaways", [])

        questions = result.pop("questions", [])
        result.setdefault("representative_questions", questions)
        result["distribution"] = distribution
        result["distribution"]["concepts"] = concepts
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "summary": "Could not generate summary.",
            "takeaways": [],
            "representative_questions": [],
            "distribution": {"intents": [], "topics": [], "concepts": []},
        }


def get_cluster_summary(cluster_id: str, model: str = DEFAULT_MODEL) -> dict:
    cluster = get_cluster(cluster_id)
    if not cluster:
        return {"error": "cluster not found"}

    cache_path = STORE_DIR / f"cluster_{cluster_id}_summary.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("_model") == model:
                return cached
        except Exception:
            pass

    result = build_cluster_summary(cluster, model)
    result["_model"] = model
    result["_generated_at"] = time.time()
    try:
        cache_path.write_text(json.dumps(result, indent=2))
    except Exception:
        pass
    return result
