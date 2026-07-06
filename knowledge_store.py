"""
Knowledge Store — persistent vector + graph store for conversation intelligence.

Stores:
  - Vector index: query/entity embeddings for semantic search across conversations
  - Graph index: entity-relationship graph for navigation
  - Entity registry: typed entities (topic, concept, intent) with cross-links

All data is JSON-serialized to .knowledge_store/ for zero-dependency operation.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import hashlib

# ── Config ─────────────────────────────────────────────────────────────────

STORE_DIR = Path(__file__).parent / ".knowledge_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_PATH = STORE_DIR / "vectors.json"
GRAPH_PATH = STORE_DIR / "graph.json"
ENTITIES_PATH = STORE_DIR / "entities.json"
SUMMARY_PATH = STORE_DIR / "summary.json"


# ── Helpers ────────────────────────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-10)


def _id_hash(*parts: str) -> str:
    return hashlib.md5(":".join(parts).encode()).hexdigest()[:16]


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


# ── Vector Index ───────────────────────────────────────────────────────────

class VectorIndex:
    """Simple cosine-similarity vector index backed by JSON."""

    def __init__(self):
        self._data: dict[str, dict] = _load_json(VECTOR_PATH)
        # {vec_id: {vector: [...], meta: {type, id, text, conversation_id, timestamp}}}

    def add(self, vec_id: str, vector: list[float], meta: dict) -> None:
        self._data[vec_id] = {"vector": vector, "meta": meta}

    def get(self, vec_id: str) -> dict | None:
        return self._data.get(vec_id)

    def remove(self, vec_id: str) -> None:
        self._data.pop(vec_id, None)

    def search(self, query_vec: list[float], top_k: int = 20,
               type_filter: str | None = None) -> list[dict]:
        """Return top_k nearest neighbors."""
        scored = []
        for vid, entry in self._data.items():
            if type_filter and entry.get("meta", {}).get("type") != type_filter:
                continue
            sim = _cosine_sim(query_vec, entry["vector"])
            scored.append((sim, vid, entry["meta"]))
        scored.sort(key=lambda x: -x[0])
        return [{"id": vid, "score": round(s, 4), "meta": m}
                for s, vid, m in scored[:top_k]]

    def search_by_text(self, text: str, embed_fn, top_k: int = 20,
                       type_filter: str | None = None) -> list[dict]:
        """Embed text then search."""
        from intent_engine import get_embeddings
        vecs = get_embeddings([text[:400]])
        if not vecs:
            return []
        return self.search(vecs[0], top_k, type_filter)

    def count(self) -> int:
        return len(self._data)

    def flush(self) -> None:
        _save_json(VECTOR_PATH, self._data)


# ── Graph Index ────────────────────────────────────────────────────────────

class GraphIndex:
    """Typed entity graph with edges for navigation."""

    def __init__(self):
        self._data: dict = _load_json(GRAPH_PATH)
        # {node_id: {type, label, properties: {}, edges: [{target, relation, weight}]}}
        if "nodes" not in self._data:
            self._data = {"nodes": {}}

    def add_node(self, node_id: str, ntype: str, label: str,
                 props: dict | None = None) -> None:
        if node_id not in self._data["nodes"]:
            self._data["nodes"][node_id] = {
                "type": ntype, "label": label,
                "properties": props or {},
                "edges": [],
            }

    def add_edge(self, source: str, target: str, relation: str,
                 weight: float = 1.0) -> None:
        nodes = self._data["nodes"]
        if source not in nodes or target not in nodes:
            return
        edges = nodes[source]["edges"]
        # Update weight if edge exists
        for e in edges:
            if e["target"] == target and e["relation"] == relation:
                e["weight"] = min(10.0, e["weight"] + weight)
                return
        edges.append({"target": target, "relation": relation, "weight": weight})

    def get_node(self, node_id: str) -> dict | None:
        return self._data["nodes"].get(node_id)

    def get_neighbors(self, node_id: str, relation: str | None = None,
                      max_depth: int = 1) -> list[dict]:
        """BFS traversal from node_id up to max_depth."""
        if node_id not in self._data["nodes"]:
            return []
        visited = {node_id}
        results: list[dict] = []
        queue = [(node_id, 0)]
        while queue:
            current, depth = queue.pop(0)
            if depth > 0:
                node = self._data["nodes"].get(current)
                if node:
                    results.append({"id": current, "node": node, "depth": depth})
            if depth >= max_depth:
                continue
            for edge in self._data["nodes"].get(current, {}).get("edges", []):
                if relation and edge["relation"] != relation:
                    continue
                if edge["target"] not in visited:
                    visited.add(edge["target"])
                    queue.append((edge["target"], depth + 1))
        return results

    def get_nodes_by_type(self, ntype: str) -> list[dict]:
        return [{"id": nid, **nd} for nid, nd in self._data["nodes"].items()
                if nd.get("type") == ntype]

    def count_nodes(self) -> int:
        return len(self._data["nodes"])

    def count_edges(self) -> int:
        return sum(len(nd.get("edges", [])) for nd in self._data["nodes"].values())

    def flush(self) -> None:
        _save_json(GRAPH_PATH, self._data)


# ── Entity Registry ────────────────────────────────────────────────────────

class EntityRegistry:
    """Cross-conversation entity catalog."""

    def __init__(self):
        self._data: dict = _load_json(ENTITIES_PATH)
        # {entity_type: {entity_name: {id, count, conversations: [conv_ids], related: {}}}}
        if "topics" not in self._data:
            self._data = {
                "topics": {},
                "concepts": {},
                "intents": {},
                "tags": {},
            }

    def register(self, etype: str, name: str, conv_id: str) -> str:
        """Register an entity occurrence and return its stable ID."""
        norm = name.lower().strip()
        registry = self._data.get(etype, {})
        if norm not in registry:
            registry[norm] = {
                "id": f"{etype}_{norm}",
                "name": name,
                "count": 0,
                "conversations": [],
                "related": {},
            }
        entry = registry[norm]
        entry["count"] += 1
        if conv_id not in entry["conversations"]:
            entry["conversations"].append(conv_id)
        self._data[etype] = registry
        return entry["id"]

    def get_by_type(self, etype: str, min_count: int = 1) -> list[dict]:
        return [v for v in self._data.get(etype, {}).values() if v["count"] >= min_count]

    def all_entities(self, min_count: int = 1) -> list[dict]:
        result = []
        for etype, registry in self._data.items():
            for norm, entry in registry.items():
                if entry["count"] >= min_count:
                    result.append({**entry, "entity_type": etype})
        return result

    def link_entities(self, etype_a: str, name_a: str,
                       etype_b: str, name_b: str) -> None:
        """Record a co-occurrence relationship between two entities."""
        norm_a = name_a.lower().strip()
        norm_b = name_b.lower().strip()
        for etype, a_name, b_name in [(etype_a, norm_a, norm_b),
                                       (etype_b, norm_b, norm_a)]:
            registry = self._data.get(etype, {})
            if a_name in registry:
                rel = registry[a_name]["related"]
                key = f"{etype_b}:{b_name}"
                rel[key] = rel.get(key, 0) + 1

    def flush(self) -> None:
        _save_json(ENTITIES_PATH, self._data)


# ── Relationship Extraction ────────────────────────────────────────────────

RELATION_KEYWORDS = {
    "cites": ["paper", "arxiv", "according to", "as shown in", "reference"],
    "introduces": ["introduce", "propose", "present", "novel", "new approach"],
    "extends": ["extend", "build upon", "based on", "derived from", "variant"],
    "compares": ["compare", "vs", "versus", "difference", "better than", "worse"],
    "contrasts": ["contrast", "however", "unlike", "whereas", "on the other hand"],
    "improves": ["improve", "better", "enhance", "outperform", "superior"],
    "uses": ["use", "utilize", "apply", "employ", "leverage"],
    "related_to": ["related", "similar", "connected", "associated", "pertain"],
}


def extract_relations(text: str) -> list[tuple[str, str]]:
    """Extract (relation, target_concept) pairs from text."""
    found = []
    lower = text.lower()
    for relation, keywords in RELATION_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                # Try to extract what comes after the keyword
                idx = lower.find(kw)
                after = text[idx + len(kw):idx + len(kw) + 60]
                # Get noun phrase
                target = re.sub(r'^[^a-zA-Z]+', '', after)
                target = re.split(r'[.,;!?]', target)[0].strip()
                if target and len(target) > 3 and len(target) < 50:
                    found.append((relation, target))
                    break
    return found[:3]


# ── Summary stats ──────────────────────────────────────────────────────────

def save_summary(stats: dict) -> None:
    _save_json(SUMMARY_PATH, stats)


def load_summary() -> dict:
    return _load_json(SUMMARY_PATH)


# ── Build the full knowledge store ─────────────────────────────────────────

def build_knowledge_store(conversations: list[dict],
                           message_fn,
                           extraction_cache: dict[str, dict],
                           embed_fn,
                           model: str = "llama3.2:3b") -> dict:
    """Build vector + graph store from all conversations.

    Returns stats dict.
    """
    vi = VectorIndex()
    gi = GraphIndex()
    er = EntityRegistry()

    t0 = time.time()
    total_queries = 0

    for conv in conversations:
        conv_id = conv.get("id", "")
        title = conv.get("title", "Untitled")
        messages = message_fn(conv)
        extraction = extraction_cache.get(conv_id, {})

        # Extract conversation-level entities
        conv_topics = [t.lower().strip() for t in
                      extraction.get("topics", []) if isinstance(t, str)]
        conv_intents = [extraction.get("intent", "other")]

        # Register conversation node
        gi.add_node(conv_id, "conversation", title,
                    {"timestamp": conv.get("created_at", 0)})

        # Track registered entities for this conversation
        conv_entity_ids = {"topics": set(), "intents": set(), "concepts": set()}

        for t in conv_topics:
            eid = er.register("topics", t, conv_id)
            gi.add_node(eid, "topic", t)
            gi.add_edge(conv_id, eid, "has_topic")
            conv_entity_ids["topics"].add(eid)

        for intent in conv_intents:
            eid = er.register("intents", intent, conv_id)
            gi.add_node(eid, "intent", intent)
            gi.add_edge(conv_id, eid, "has_intent")
            conv_entity_ids["intents"].add(eid)

        # Process each user query
        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not content:
                continue

            query_id = f"q_{conv_id}_{i}"
            total_queries += 1

            # Register query node
            gi.add_node(query_id, "query", content[:60])
            gi.add_edge(conv_id, query_id, "contains")

            # Embed for vector search
            query_text = content[:400]
            embeds = embed_fn([query_text]) if query_text else []
            if embeds and embeds[0]:
                vi.add(query_id, embeds[0], {
                    "type": "query",
                    "text": content[:200],
                    "conversation_id": conv_id,
                    "conversation_title": title,
                    "index": i,
                })

            # Link to answer if available
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                answer_id = f"a_{conv_id}_{i}"
                answer_text = (messages[i + 1].get("content") or "")[:60]
                gi.add_node(answer_id, "answer", answer_text)
                gi.add_edge(query_id, answer_id, "has_answer")

                # Extract relations from answer
                relations = extract_relations(messages[i + 1].get("content", ""))
                for rel, target in relations:
                    target_eid = er.register("concepts", target, conv_id)
                    gi.add_node(target_eid, "concept", target)
                    gi.add_edge(query_id, target_eid, rel)
                    conv_entity_ids["concepts"].add(target_eid)

            # Link query to intents
            for iid in conv_entity_ids["intents"]:
                gi.add_edge(query_id, iid, "expresses")

            # Link query to topics via keyword matching
            for tid in conv_entity_ids["topics"]:
                topic_node = gi.get_node(tid)
                if topic_node and topic_node.get("label", "").lower() in content.lower():
                    gi.add_edge(query_id, tid, "about")

        # Cross-link related entities
        topics_list = list(conv_entity_ids["topics"])
        for i in range(len(topics_list)):
            for j in range(i + 1, len(topics_list)):
                gi.add_edge(topics_list[i], topics_list[j], "related_to", 0.5)

    # Save everything
    vi.flush()
    gi.flush()
    er.flush()

    elapsed = time.time() - t0
    stats = {
        "conversations": len(conversations),
        "queries_indexed": total_queries,
        "vector_entries": vi.count(),
        "graph_nodes": gi.count_nodes(),
        "graph_edges": gi.count_edges(),
        "entities": len(er.all_entities()),
        "build_time_s": round(elapsed, 1),
    }
    save_summary(stats)
    return stats
