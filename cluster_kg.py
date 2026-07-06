"""
Cluster Knowledge Graph — build HiveGraph instances from cluster Q&A pairs.

For a given cluster we:
  1. Collect all query+answer text across the cluster's items
  2. Extract entities (technical terms, concepts, topics) via regex
  3. Build co-occurrence edges with typed relations
  4. Return a HiveGraph serialized as Node-Link JSON

Each cluster becomes a Hive (topic-specific knowledge graph) where:
  - entities → Node(type=CONCEPT) with entity-type stored in properties
  - co-occurrences → Edge(relation=related_to|uses|introduces)
  - inter-cluster references → Node(type=GRAPH_REF)
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from hive_datatype import HiveGraph, Node, NodeType, Edge

# ── Patterns ──────────────────────────────────────────────────────────
TECHNICAL_TERM_PATTERNS = [
    (r"\b[A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+)*\b", "concept"),
    (r"\b[A-Z]{2,}\b", "acronym"),
    (r"\b\w+(?:-\w+)+\b", "technical_term"),
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

# Single words that are false positives when capitalized
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
    # Generic query framing words
    "short answer", "step-by-step", "step by step", "real-world",
    "real world", "in-depth", "high-level", "deep dive", "follow-up",
    "follow up", "quick start", "hands-on", "built-in", "built in",
    "use case", "use-case", "first-class", "first class",
    "open-source", "open source", "front-end", "front end",
    "back-end", "back end", "proof of concept", "proof-of-concept",
    "best practice", "best-practice", "ground truth", "ground-truth",
    "quick overview", "brief explanation", "short summary",
    # Prepositional sentence fragments
    "in", "of", "at", "by", "for", "to", "with", "from", "into",
    "onto", "upon", "within", "without", "throughout", "along",
    "among", "between", "beyond", "under", "over", "above", "below",
    "across", "against", "around", "behind", "beneath", "beside",
    "besides", "toward", "towards", "via",
})

# Entity type → D3 group for frontend coloring
ENTITY_TYPE_GROUP = {
    "topic": 0,
    "concept": 2,
    "technical_term": 3,
    "acronym": 4,
}


# ── Helpers ───────────────────────────────────────────────────────────

def _id_hash(text: str) -> str:
    return hashlib.md5(text.lower().encode()).hexdigest()[:12]


def _normalize(text: str) -> str:
    return text.lower().strip().strip('.').strip('"').strip("'")


def _term_from_text(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pattern, etype in TECHNICAL_TERM_PATTERNS:
        for m in re.finditer(pattern, text):
            raw = m.group(0)
            norm = _normalize(raw)
            if len(norm) < 3 or norm in STOP_WORDS:
                continue
            if re.match(r'^\d+[.\d]*$', norm):
                continue
            if norm in FALSE_POSITIVES:
                continue
            # Skip multi-word phrases where any component word is a stop word
            words = raw.split()
            if len(words) > 1 and any(w.lower() in STOP_WORDS for w in words):
                continue
            if norm in seen:
                continue
            seen.add(norm)
            found.append((raw, etype))
    return found


def _build_hive(cluster: dict) -> HiveGraph:
    """Convert a cluster into a HiveGraph."""
    items = cluster.get("items", [])
    cluster_id = cluster.get("id", "unknown")
    cluster_label = cluster.get("label", "unknown")

    hive = HiveGraph(id=f"hive_{cluster_id}")

    # ── 1. Count entities across all items ────────────────────────────
    entity_data: dict[str, dict] = {}
    # key -> {label, etype, count, source_indices: set}

    for idx, item in enumerate(items):
        query = item.get("query", "")
        answer = item.get("answer", "")

        # Topics from item metadata
        for topic in item.get("topics", []):
            norm = _normalize(topic)
            key = f"topic::{norm}"
            if key not in entity_data:
                entity_data[key] = {"label": topic.title(), "etype": "topic", "count": 0, "indices": set()}
            entity_data[key]["count"] += 1
            entity_data[key]["indices"].add(idx)

        # Technical terms from query + answer
        combined = f"{query} {answer}"
        for term, term_type in _term_from_text(combined):
            norm = _normalize(term)
            key = f"tech::{norm}"
            etype = "acronym" if term_type == "acronym" else "technical_term"
            if key not in entity_data:
                entity_data[key] = {"label": term, "etype": etype, "count": 0, "indices": set()}
            entity_data[key]["count"] += 1
            entity_data[key]["indices"].add(idx)

    # ── 2. Sort by frequency, take top 25 (min freq 2) ────────────────
    sorted_entities = sorted(entity_data.values(), key=lambda e: (-e["count"], e["label"]))
    top_entities = []
    for ent in sorted_entities:
        if ent["count"] < 2:
            continue
        top_entities.append(ent)
        if len(top_entities) >= 25:
            break

    # ── 3. Add nodes to hive ──────────────────────────────────────────
    for ent in top_entities:
        eid = _id_hash(f"{ent['etype']}::{ent['label']}")
        definition = f"{ent['etype']} mentioned {ent['count']} times across {len(ent['indices'])} Q&A pairs"
        node = Node(
            id=eid,
            type=NodeType.CONCEPT,
            label=ent["label"],
            definition=definition,
            concept_type=ent["etype"],
        )
        hive.nodes.append(node)

    # ── 4. Build co-occurrence edges ──────────────────────────────────
    ent_items: dict[str, set[int]] = {}
    for ent in top_entities:
        eid = _id_hash(f"{ent['etype']}::{ent['label']}")
        ent_items[eid] = ent["indices"]

    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    eids = list(ent_items.keys())
    for i in range(len(eids)):
        for j in range(i + 1, len(eids)):
            shared = ent_items[eids[i]] & ent_items[eids[j]]
            if len(shared) > 0:
                edge_counts[(eids[i], eids[j])] = len(shared)

    sorted_edges = sorted(edge_counts.items(), key=lambda kv: -kv[1])

    for (src, tgt), weight in sorted_edges[:50]:
        hive.edges.append(Edge(
            source=src,
            target=tgt,
            relation="related_to",
        ))

    return hive


def build_cluster_knowledge_graph(
    cluster: dict,
    model: str = "llama3.2:3b",
) -> dict:
    """Build a HiveGraph from a cluster's Q&A items.

    Returns the serialized HiveGraph (Node-Link JSON) with extra metadata.
    """
    items = cluster.get("items", [])
    cluster_id = cluster.get("id", "unknown")
    cluster_label = cluster.get("label", "unknown")

    if not items:
        return {
            "directed": True, "multigraph": True, "graph": {},
            "graph_id": f"hive_{cluster_id}",
            "nodes": [], "links": [],
            "metadata": {"cluster_id": cluster_id, "entity_count": 0},
        }

    hive = _build_hive(cluster)
    result = hive.to_node_link_dict()

    # Extract frequency and entity type from definition field for frontend
    for nd in result["nodes"]:
        defn = nd.get("definition", "")
        # definition format: "{etype} mentioned {count} times across {src_count} Q&A pairs"
        m = re.match(r"(\w+) mentioned (\d+) times across (\d+)", defn)
        if m:
            nd["frequency"] = int(m.group(2))
            nd["concept_type"] = m.group(1)
        else:
            nd["frequency"] = 1
            nd["concept_type"] = "concept"
        nd.pop("definition", None)

    # Inject metadata
    result["metadata"] = {
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "entity_count": len(hive.nodes),
        "edge_count": len(hive.edges),
        "item_count": len(items),
    }

    return result


def get_cluster_knowledge_graph(cluster_id: str, model: str = "llama3.2:3b") -> dict | None:
    from cluster_store import get_cluster
    cluster = get_cluster(cluster_id)
    if not cluster:
        return None
    return build_cluster_knowledge_graph(cluster, model)
