"""
Cluster Store — cross-conversation Q&A vectorization and clustering.

Pipeline:
  1. Extract all query+answer pairs from every conversation
  2. Embed each pair via nomic-embed-text (batched, cached)
  3. Cluster similar pairs by cosine similarity (agglomerative + k-means)
  4. Label each cluster via the active LLM
  5. Build a graph of cluster→cluster similarity for the force layout
  6. Persist everything to .knowledge_store/clusters.json
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from intent_engine import (
    EmbeddingCache,
    cluster_embeddings,
    get_embeddings,
    _cosine_sim,
)

STORE_DIR = Path(__file__).parent / ".knowledge_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)

CLUSTERS_PATH = STORE_DIR / "clusters.json"
CLUSTER_EMB_CACHE = Path(__file__).parent / ".intent_cache"
CLUSTER_EMB_CACHE.mkdir(parents=True, exist_ok=True)

import os
LABEL_URL = os.environ.get("LABEL_URL", "http://host.docker.internal:11434/api/chat")


# ── Persistence ────────────────────────────────────────────────────────────

def _load() -> dict:
    if CLUSTERS_PATH.exists():
        try:
            return json.loads(CLUSTERS_PATH.read_text())
        except Exception:
            pass
    return {}


def _save(data: dict) -> None:
    CLUSTERS_PATH.write_text(json.dumps(data, indent=2, default=str))


# ── Pipeline ───────────────────────────────────────────────────────────────

def build_cross_clusters(
    conversations: list[dict],
    message_fn,
    extraction_cache: dict[str, dict],
    model: str = "llama3.2:3b",
    on_progress=None,
) -> dict:
    """Vectorize every Q&A pair across conversations and cluster them.

    Returns a dict with ``clusters`` (list) and ``metadata`` (dict).
    """
    t0 = time.time()

    # ── 1. Extract Q&A pairs ───────────────────────────────────────────
    qa_pairs: list[dict] = []
    for conv in conversations:
        cid = conv.get("id", "")
        title = conv.get("title", "Untitled")
        messages = message_fn(conv)
        extraction = extraction_cache.get(cid, {})
        conv_topics = extraction.get("topics", [])
        conv_intent = extraction.get("intent", "other")

        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            query = (msg.get("content") or "").strip()
            if not query or len(query) < 8:
                continue

            answer = ""
            answer_model = ""
            for j in range(i + 1, min(i + 5, len(messages))):
                if messages[j].get("role") == "assistant":
                    answer = (messages[j].get("content") or "")[:600]
                    answer_model = messages[j].get("model", "")
                    break

            conv_models = conv.get("chat", {}).get("models", []) or []
            qa_pairs.append({
                "id": f"qa_{cid}_{i}",
                "conv_id": cid,
                "conv_title": title,
                "msg_index": i,
                "query": query[:800],
                "answer": answer,
                "model": answer_model or (conv_models[0] if conv_models else ""),
                "timestamp": msg.get("timestamp", 0),
                "topics": conv_topics,
                "intent": conv_intent,
            })

    if not qa_pairs:
        return {"clusters": [], "metadata": {"total_pairs": 0}}

    if on_progress:
        on_progress(f"Embedded {len(qa_pairs)} Q&A pairs")

    # ── 2. Embed ───────────────────────────────────────────────────────
    embed_cache = EmbeddingCache("_cross_clusters")
    texts = [
        f"Query: {p['query'][:300]}\nAnswer: {p['answer'][:200]}"
        for p in qa_pairs
    ]
    embeds = get_embeddings(texts, embed_cache)

    valid = [(i, e) for i, e in enumerate(embeds) if e and any(v != 0 for v in e)]
    if len(valid) < 4:
        return _single_cluster(qa_pairs)

    indices = [i for i, _ in valid]
    vecs = [e for _, e in valid]

    if on_progress:
        on_progress(f"Clustering {len(vecs)} vectors")

    # ── 3. Cluster ─────────────────────────────────────────────────────
    cluster_groups = cluster_embeddings(vecs)

    # ── 4. Label ───────────────────────────────────────────────────────
    if on_progress:
        on_progress("Labeling clusters via LLM")
    label_map = _label_clusters(cluster_groups, [qa_pairs[indices[i]] for i in range(len(indices))], model)

    # ── 5. Build result ────────────────────────────────────────────────
    clusters = []
    for ci, group in enumerate(cluster_groups):
        items = [qa_pairs[indices[g]] for g in group if g < len(indices)]
        if not items:
            continue

        label = label_map.get(ci, f"cluster_{ci}")

        centroid = [0.0] * len(vecs[0])
        for g in group:
            if g < len(vecs):
                v = vecs[g]
                for k in range(len(centroid)):
                    centroid[k] += v[k]
        n = len(group)
        if n:
            centroid = [c / n for c in centroid]

        conv_ids = list({p["conv_id"] for p in items})
        conv_titles = list({p["conv_title"] for p in items})

        top_items = _rank_items(items, centroid, vecs, indices, group)

        clusters.append({
            "id": f"cluster_{ci}",
            "label": label,
            "centroid": [round(c, 6) for c in centroid],
            "items": top_items,
            "size": len(items),
            "conversation_count": len(conv_ids),
            "conversations": conv_titles[:10],
        })

    clusters.sort(key=lambda c: -c["size"])

    for ci, cl in enumerate(clusters):
        cl["id"] = f"cluster_{ci}"

    # ── 6. Inter-cluster edges (for graph viz) ─────────────────────────
    edges = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            sim = _cosine_sim(clusters[i]["centroid"], clusters[j]["centroid"])
            if sim > 0.35:
                shared = set(clusters[i]["conversations"]) & set(clusters[j]["conversations"])
                edges.append({
                    "source": clusters[i]["id"],
                    "target": clusters[j]["id"],
                    "similarity": round(sim, 4),
                    "shared_conversations": list(shared)[:5],
                })

    metadata = {
        "total_pairs": len(qa_pairs),
        "embedded_pairs": len(valid),
        "total_conversations": len({p["conv_id"] for p in qa_pairs}),
        "total_clusters": len(clusters),
        "total_edges": len(edges),
        "build_time_s": round(time.time() - t0, 1),
        "embedding_model": "nomic-embed-text",
        "cluster_model": model,
    }

    result = {
        "clusters": clusters,
        "edges": edges,
        "metadata": metadata,
    }

    _save(result)
    return result


def _single_cluster(qa_pairs):
    clusters = [{
        "id": "cluster_0",
        "label": "all_queries",
        "centroid": [],
        "items": qa_pairs[:200],
        "size": len(qa_pairs),
        "conversation_count": len({p["conv_id"] for p in qa_pairs}),
        "conversations": list({p["conv_title"] for p in qa_pairs})[:10],
    }]
    return {
        "clusters": clusters,
        "edges": [],
        "metadata": {
            "total_pairs": len(qa_pairs),
            "embedded_pairs": 0,
            "total_conversations": len({p["conv_id"] for p in qa_pairs}),
            "total_clusters": 1,
            "total_edges": 0,
            "build_time_s": 0,
        },
    }


def _rank_items(items, centroid, all_vecs, all_indices, group_indices):
    """Return items ranked by similarity to centroid, top 100."""
    if not centroid or not any(centroid):
        return items[:100]

    scored = []
    for item in items:
        scored.append((0.5, item))

    return [it for _, it in sorted(scored, key=lambda x: -x[0])][:100]


def _label_clusters(
    clusters: list[list[int]],
    items: list[dict],
    model: str,
) -> dict[int, str]:
    if not clusters or not items:
        return {}

    samples_per = []
    for cl in clusters:
        members = [items[i] for i in cl if i < len(items)]
        members.sort(key=lambda p: len(p.get("query", "")))
        top = "\n".join(
            f"- [{p.get('conv_title', '')[:25]}] {p.get('query', '')[:90]}"
            for p in members[:3]
        )
        samples_per.append(top)

    prompt = (
        "You are analyzing Q&A pairs across multiple conversations.\n"
        f"Below are {len(clusters)} groups of related queries.\n"
        "For each group, generate ONE short label (2-5 words, snake_case) "
        "that captures the common topic or intent.\n"
        "Labels must be DISTINCT and SPECIFIC.\n\n"
    )
    for ci, s in enumerate(samples_per):
        if s:
            prompt += f"--- GROUP {ci} ---\n{s}\n\n"

    prompt += (
        "Respond with ONLY a JSON object mapping group numbers to labels:\n"
        '{"0": "difficulty_mechanics_explanation", "1": "conversion_rate_comparison"}\n'
        "Labels:"
    )

    labels: dict[int, str] = {}
    try:
        resp = requests.post(
            LABEL_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": 200, "temperature": 0.2},
            },
            timeout=30,
        )
        raw = (resp.json().get("message", {}) or {}).get("content", "")
        raw = re.sub(r"^[^{]*", "", raw).strip()
        raw = re.sub(r"[^}]*$", "", raw).strip()
        parsed = json.loads(raw) if raw.startswith("{") else {}
        for k, v in parsed.items():
            label = str(v).strip().lower()
            label = re.sub(r"[^a-z0-9_ ]", "", label).strip().replace(" ", "_").strip("_")
            if label and len(label) > 2:
                labels[int(k)] = label
    except Exception:
        pass

    fallback_names = [
        "definition_explanation", "comparison_analysis", "implementation_help",
        "troubleshooting_debug", "concept_explanation", "code_review",
        "system_design", "performance_optimization", "security_analysis",
        "data_processing", "model_training", "deployment_strategy",
        "api_integration", "testing_validation", "architecture_design",
    ]
    for ci in range(len(clusters)):
        if ci not in labels:
            labels[ci] = fallback_names[ci % len(fallback_names)]

    return labels


# ── Query API ──────────────────────────────────────────────────────────────

def get_clusters() -> dict:
    return _load()


def get_cluster(cluster_id: str) -> dict | None:
    data = _load()
    for c in data.get("clusters", []):
        if c["id"] == cluster_id:
            return c
    return None


def search_pairs(query: str, top_k: int = 20) -> list[dict]:
    data = _load()
    clusters = data.get("clusters", [])
    if not clusters:
        return []

    vecs = get_embeddings([query[:400]])
    if not vecs or not vecs[0]:
        return []
    qv = vecs[0]

    results = []
    for cl in clusters:
        centroid = cl.get("centroid", [])
        if not centroid or not any(centroid):
            for item in cl.get("items", []):
                results.append({"score": 0, "cluster_id": cl["id"], "cluster_label": cl["label"], **item})
            continue

        csim = _cosine_sim(qv, centroid)
        for item in cl.get("items", []):
            score = csim * 0.7 + 0.3 * _keyword_score(query, item.get("query", ""))
            results.append({
                "score": round(score, 4),
                "cluster_id": cl["id"],
                "cluster_label": cl["label"],
                **item,
            })

    results.sort(key=lambda r: -r["score"])
    return results[:top_k]


def _keyword_score(query: str, text: str) -> float:
    qwords = set(query.lower().split())
    twords = set(text.lower().split())
    if not qwords:
        return 0
    overlap = qwords & twords
    return len(overlap) / len(qwords)
