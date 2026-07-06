"""
Cluster Knowledge Graph — extract entities and relationships from cluster Q&A pairs.

For a given cluster, we:
  1. Collect all query+answer text across the cluster's items
  2. Extract entities (technical terms, concepts, topics) via regex + known keywords
  3. Build relationships based on co-occurrence within the same Q&A pair
  4. Return a node-link JSON consumable by a D3.js hive plot
"""

from __future__ import annotations

import json
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

KG_CACHE_DIR = Path(__file__).parent / ".knowledge_store"
KG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LLAMA_URL = "http://localhost:11434/api/chat"

# ── Domain-specific technical keywords (hive-mining vocabulary) ────────
TECHNICAL_TERM_PATTERNS = [
    (r"\b[A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+)*\b", "concept"),  # Multi-word capitalized
    (r"\b[A-Z]{2,}\b", "acronym"),                                  # ALL CAPS
    (r"\b\w+(?:-\w+)+\b", "technical_term"),                        # hyphenated terms
]

STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "dare", "ought", "used", "this", "that", "these", "those", "it",
    "its", "they", "them", "their", "we", "us", "our", "you", "your",
    "he", "she", "him", "her", "his", "i", "my", "me", "what", "which",
    "who", "whom", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "because",
    "about", "above", "after", "again", "against", "below", "between",
    "during", "without", "through", "before", "between", "under", "over",
    "out", "off", "up", "down", "then", "once", "here", "there", "please",
    "help", "know", "like", "want", "tell", "ask", "get", "make", "go",
    "see", "come", "take", "think", "look", "give", "find", "use", "say",
    "try", "leave", "call", "thank", "thanks", "yes", "no", "hi", "hello",
    "hey", "oh", "ok", "okay", "sure", "well", "right", "actually",
    "basically", "essentially", "literally", "seriously", "absolutely",
    "definitely", "probably", "maybe", "perhaps", "quite", "rather",
    "somewhat", "really", "pretty", "much", "also", "even", "still",
    "already", "yet", "just", "now", "then", "here", "there", "always",
    "never", "often", "sometimes", "usually", "frequently", "rarely",
    "eventually", "finally", "first", "next", "last", "later", "soon",
    "immediately", "instantly", "suddenly", "quickly", "slowly",
    "carefully", "easily", "simply", "especially", "generally",
    "typically", "specifically", "particularly", "including",
    "regarding", "considering", "following", "previous", "current",
    "previous", "final", "entire", "complete", "total", "partial",
    "certain", "various", "different", "multiple", "several", "many",
})

LEGACY_TOPICS = {
    "price_market", "mining_difficulty", "conversion_rate",
    "network_protocol", "stability_peg", "security_attack",
    "performance_scale", "token_economics", "governance_vote",
    "technical_impl", "definition_explain", "comparison",
    "diagram_visual", "mechanism_how",
}


# ── Helpers ───────────────────────────────────────────────────────────

def _id_hash(text: str) -> str:
    return hashlib.md5(text.lower().encode()).hexdigest()[:12]


def _normalize(text: str) -> str:
    return text.lower().strip().strip('.').strip('"').strip("'")


def _tech_terms_from_text(text: str) -> list[tuple[str, str]]:
    """Extract (normalized_term, entity_type) pairs from text."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Common English words that get falsely capitalized at sentence start
    FALSE_POSITIVES = frozenset({
        "the", "this", "that", "these", "those", "what", "which", "who", "whom",
        "when", "where", "why", "how", "can", "could", "would", "should", "will",
        "shall", "may", "might", "must", "need", "dare", "ought", "used", "let",
        "get", "make", "take", "give", "find", "keep", "know", "want", "see",
        "come", "go", "put", "set", "use", "try", "ask", "tell", "call", "send",
        "run", "move", "show", "start", "stop", "look", "hear", "feel", "think",
        "believe", "understand", "explain", "describe", "define", "list",
        "outline", "summarize", "compare", "contrast", "discuss", "provide",
        "include", "consider", "regarding", "following", "following", "based",
        "also", "even", "still", "already", "yet", "just", "now", "then",
        "here", "there", "always", "never", "often", "usually", "sometimes",
        "finally", "eventually", "basically", "essentially", "actually",
        "please", "thank", "thanks", "hello", "hey", "hi", "yes", "no", "ok",
        "okay", "sure", "well", "right", "great", "good", "nice", "awesome",
        "perfect", "correct", "exactly", "absolutely", "definitely",
        "probably", "maybe", "perhaps", "quite", "very", "extremely",
        "highly", "deeply", "strongly", "fully", "clearly", "obviously",
        "importantly", "interestingly", "notably", "specifically",
        "particularly", "especially", "mostly", "mainly", "primarily",
        "largely", "basically", "essentially", "fundamentally",
        "additionally", "furthermore", "moreover", "however", "therefore",
        "thus", "hence", "consequently", "accordingly", "nevertheless",
        "nonetheless", "instead", "otherwise", "meanwhile", "while", "during",
        "before", "after", "above", "below", "between", "through", "across",
        "against", "without", "within", "about", "around", "behind", "beyond",
        "chapter", "section", "previous", "next", "last", "first", "second",
        "third", "finally", "initially", "currently", "previously",
        "recently", "recent", "original", "specific", "general", "common",
        "typical", "normal", "regular", "standard", "basic", "simple",
        "complex", "advanced", "proper", "important", "necessary",
        "possible", "potential", "likely", "certain", "multiple", "various",
        "different", "similar", "related", "relevant", "specific",
        "particular", "individual", "separate", "additional", "extra",
        "further", "entire", "entire", "complete", "total", "partial",
        "full", "whole", "half", "quarter", "double", "triple",
    })

    for pattern, etype in TECHNICAL_TERM_PATTERNS:
        for m in re.finditer(pattern, text):
            raw = m.group(0)
            norm = _normalize(raw)

            # Basic filters
            if len(norm) < 3 or norm in STOP_WORDS:
                continue
            if re.match(r'^\d+[.\d]*$', norm):
                continue

            # Skip single-word false positives
            words = raw.split()
            if len(words) == 1 and norm in FALSE_POSITIVES:
                continue

            if norm in seen:
                continue
            seen.add(norm)

            found.append((raw, etype))

    return found


def _extract_entities(items: list[dict]) -> list[dict]:
    """Extract typed entities from a cluster's Q&A items.

    Returns list of {id, label, type, frequency, items_indexed: [idx]}.
    """
    entity_counter: dict[str, dict] = {}
    # entity_key -> {label, type, count, source_indices: set}

    seen_global: set[str] = set()

    for idx, item in enumerate(items):
        query = item.get("query", "")
        answer = item.get("answer", "")

        # 1. Topic entities from the item's topic tags
        for topic in item.get("topics", []):
            norm = _normalize(topic)
            if norm and norm not in seen_global:
                seen_global.add(norm)
            key = f"topic::{norm}" if norm else f"topic::{topic}"
            if key not in entity_counter:
                entity_counter[key] = {
                    "label": topic.title() if topic else topic,
                    "type": "topic",
                    "count": 0,
                    "source_indices": set(),
                }
            entity_counter[key]["count"] += 1
            entity_counter[key]["source_indices"].add(idx)

        # 2. Intent entity
        intent = item.get("intent", "")
        if intent:
            norm = _normalize(intent)
            key = f"intent::{norm}"
            if key not in entity_counter:
                entity_counter[key] = {
                    "label": intent.title(),
                    "type": "intent",
                    "count": 0,
                    "source_indices": set(),
                }
            entity_counter[key]["count"] += 1
            entity_counter[key]["source_indices"].add(idx)

        # 3. Technical terms from query + answer
        combined = f"{query} {answer}"
        for term, term_type in _tech_terms_from_text(combined):
            norm = _normalize(term)
            if norm in STOP_WORDS or len(norm) < 3:
                continue
            # Skip pure digits
            if re.match(r'^\d+[.\d]*$', norm):
                continue
            key = f"tech::{norm}"
            if key not in entity_counter:
                entity_counter[key] = {
                    "label": term,
                    "type": "technical_term" if term_type != "acronym" else "acronym",
                    "count": 0,
                    "source_indices": set(),
                }
            entity_counter[key]["count"] += 1
            entity_counter[key]["source_indices"].add(idx)

    # Filter: keep entities that appear in at least 1 item
    # Sort by frequency, take top 60
    sorted_entities = sorted(
        entity_counter.values(),
        key=lambda e: (-e["count"], e["label"]),
    )

    result = []
    for ent in sorted_entities[:60]:
        eid = _id_hash(f"{ent['type']}::{ent['label']}")
        result.append({
            "id": eid,
            "label": ent["label"],
            "type": ent["type"],
            "frequency": ent["count"],
            "source_count": len(ent["source_indices"]),
        })

    return result


def _build_edges(
    entities: list[dict],
    items: list[dict],
) -> list[dict]:
    """Build edges between entities based on co-occurrence.

    Two entities are linked if they appear in the same Q&A item.
    Edge weight = number of shared items.
    """
    # entity_id -> set of item indices
    ent_items: dict[str, set[int]] = {}

    for idx, item in enumerate(items):
        # Collect entity IDs that match this item
        query_lower = (item.get("query", "") + " " + item.get("answer", "")).lower()
        topics_lower = {_normalize(t) for t in item.get("topics", []) if t}
        intent_lower = _normalize(item.get("intent", ""))

        for ent in entities:
            label_lower = ent["label"].lower()
            etype = ent["type"]

            matches = False
            if etype == "topic":
                matches = label_lower in topics_lower
            elif etype == "intent":
                matches = label_lower == intent_lower
            else:
                # Technical term: check if it appears in query/answer text
                matches = label_lower in query_lower

            if matches:
                ent_items.setdefault(ent["id"], set()).add(idx)

    # Build edges from co-occurrence
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    eids = [e["id"] for e in entities]

    for i in range(len(eids)):
        for j in range(i + 1, len(eids)):
            shared = ent_items.get(eids[i], set()) & ent_items.get(eids[j], set())
            if len(shared) > 0:
                edge_counts[(eids[i], eids[j])] = len(shared)

    # Filter: keep edges with weight >= 1, sorted, take top 120
    sorted_edges = sorted(
        edge_counts.items(),
        key=lambda kv: -kv[1],
    )

    result = []
    for (src, tgt), weight in sorted_edges[:120]:
        result.append({
            "source": src,
            "target": tgt,
            "relation": "co_occurs",
            "weight": weight,
        })

    return result


def _label_entities_via_llm(
    entities: list[dict],
    cluster_label: str,
    model: str = "llama3.2:3b",
) -> list[dict]:
    """Use LLM to refine entity labels — merge duplicates, assign better types."""
    if not entities:
        return entities

    # Build a compact summary for the LLM
    lines = []
    for i, ent in enumerate(entities):
        lines.append(f"{i}: [{ent['type']}] {ent['label']} (freq={ent['frequency']})")
    summary = "\n".join(lines)

    prompt = (
        "Below is a list of entities extracted from Q&A pairs about:\n"
        f'"{cluster_label}"\n\n'
        "For each entity, if the label is noisy or too generic, suggest a better label. "
        "If two entities are essentially the same, mark them as duplicates.\n"
        "Respond with ONLY a JSON array of objects:\n"
        '[{"index": 0, "keep": true, "label": "improved label", "type": "topic|concept|technical_term|acronym|intent"}, ...]\n'
        'If an entity should be removed, set "keep": false.\n\n'
        f"{summary}\n"
    )

    try:
        resp = requests.post(
            LLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": 300, "temperature": 0.1},
            },
            timeout=30,
        )
        raw = (resp.json().get("message", {}) or {}).get("content", "")
        raw = re.sub(r"^[^[]*", "", raw).strip()
        raw = re.sub(r"[^]]*$", "", raw).strip()
        if raw.startswith("["):
            parsed = json.loads(raw)
            corrections = {}
            for entry in parsed:
                if isinstance(entry, dict):
                    idx = entry.get("index")
                    if idx is not None and idx < len(entities):
                        corrections[idx] = entry
            for idx, correction in corrections.items():
                if not correction.get("keep", True):
                    entities[idx]["_drop"] = True
                else:
                    new_label = correction.get("label", "").strip()
                    if new_label and len(new_label) > 2:
                        entities[idx]["label"] = new_label
                    new_type = correction.get("type", "")
                    valid_types = {"topic", "concept", "technical_term", "acronym", "intent"}
                    if new_type in valid_types:
                        entities[idx]["type"] = new_type
    except Exception:
        pass

    # Remove dropped entities, regenerate IDs
    result = []
    for ent in entities:
        if ent.get("_drop"):
            continue
        ent.pop("_drop", None)
        ent["id"] = _id_hash(f"{ent['type']}::{ent['label']}")
        result.append(ent)

    return result


# ── Public API ────────────────────────────────────────────────────────

def build_cluster_knowledge_graph(
    cluster: dict,
    model: str = "llama3.2:3b",
    use_llm_refine: bool = True,
) -> dict:
    """Build a knowledge graph from a cluster's Q&A items.

    Returns a node-link dict consumable by D3.js hive plot:
    {
      "nodes": [{id, label, type, frequency, source_count, group}, ...],
      "links": [{source, target, relation, weight}, ...],
      "metadata": {cluster_id, cluster_label, total_entities, total_edges, ...}
    }
    """
    items = cluster.get("items", [])
    cluster_label = cluster.get("label", "unknown")

    if not items:
        return {"nodes": [], "links": [], "metadata": {"cluster_id": cluster.get("id"), "entity_count": 0}}

    # 1. Extract entities
    entities = _extract_entities(items)

    # 2. Optionally refine via LLM
    if use_llm_refine and entities:
        entities = _label_entities_via_llm(entities, cluster_label, model)

    # 3. Build edges
    edges = _build_edges(entities, items)

    # 4. Assign d3 groups based on entity type
    type_groups = {
        "topic": 0,
        "intent": 1,
        "concept": 2,
        "technical_term": 3,
        "acronym": 4,
    }

    for ent in entities:
        ent["group"] = type_groups.get(ent["type"], 5)
        # Ensure id is present
        if "id" not in ent:
            ent["id"] = _id_hash(f"{ent['type']}::{ent['label']}")

    # Remove duplicate nodes (by id)
    seen_ids: set[str] = set()
    unique_nodes = []
    for ent in entities:
        if ent["id"] not in seen_ids:
            seen_ids.add(ent["id"])
            unique_nodes.append(ent)
        else:
            # Merge frequency into existing
            for existing in unique_nodes:
                if existing["id"] == ent["id"]:
                    existing["frequency"] = max(existing.get("frequency", 0), ent.get("frequency", 0))
                    existing["source_count"] = max(existing.get("source_count", 0), ent.get("source_count", 0))
                    break

    # Deduplicate edges
    seen_edges: set[tuple[str, str]] = set()
    unique_edges = []
    for edge in edges:
        key = (edge["source"], edge["target"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(edge)

    return {
        "nodes": unique_nodes,
        "links": unique_edges,
        "metadata": {
            "cluster_id": cluster.get("id"),
            "cluster_label": cluster_label,
            "entity_count": len(unique_nodes),
            "edge_count": len(unique_edges),
            "item_count": len(items),
        },
    }


def get_cluster_knowledge_graph(cluster_id: str, model: str = "llama3.2:3b") -> dict | None:
    """Load a cluster and build its knowledge graph."""
    from cluster_store import get_cluster

    cluster = get_cluster(cluster_id)
    if not cluster:
        return None

    return build_cluster_knowledge_graph(cluster, model)
