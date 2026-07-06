"""
Intent Engine — vectorised query intent discovery with persistent caching.

Pipeline:
  1. Embed each user query via nomic-embed-text (batched, cached to disk)
  2. Cluster similar queries by cosine similarity (two-pass: agglomerative + k-means)
  3. Label each cluster via the selected LLM (batch prompt, /api/chat)
  4. Optionally cross-cluster across conversations for unified intent views
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

# ── Config ─────────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / ".intent_cache"
EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
LABEL_URL = os.environ.get("LABEL_URL", "http://localhost:11434/api/chat")
LABEL_MODEL = os.environ.get("LABEL_MODEL", "llama3.2:3b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CLUSTER_SIM = float(os.environ.get("CLUSTER_SIM", "0.55"))
MIN_CLUSTER_PCT = float(os.environ.get("MIN_CLUSTER_PCT", "0.005"))
MAX_QUERIES = int(os.environ.get("MAX_INTENT_QUERIES", "500"))

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-10)


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()[:400]).hexdigest()[:12]


# ── Embedding Cache ────────────────────────────────────────────────────────

class EmbeddingCache:
    """Persistent cache for query embeddings, keyed by conversation + query hash."""

    def __init__(self, conv_id: str):
        self.conv_id = conv_id
        self._path = CACHE_DIR / f"emb_{conv_id}.json"
        self._data: dict[str, list[float]] = {}
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except Exception:
                pass

    def get(self, text: str) -> list[float] | None:
        key = _content_hash(text)
        return self._data.get(key)

    def put(self, text: str, vec: list[float]) -> None:
        key = _content_hash(text)
        self._data[key] = vec

    def flush(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2))

    def hit_count(self, texts: list[str]) -> int:
        return sum(1 for t in texts if self.get(t) is not None)


# ── Embedding (batched) ────────────────────────────────────────────────────

def _fetch_batch(batch: list[tuple[int, str]]) -> list[tuple[int, list[float], str]]:
    """Fetch embeddings for a batch with timeout."""
    texts_batch = [t for _, t in batch]
    try:
        resp = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": texts_batch}, timeout=15)
        vectors = resp.json().get("embeddings", [])
    except Exception:
        vectors = [[0.0] * 768 for _ in texts_batch]
    return [(idx, vec, raw) for (idx, raw), vec in zip(batch, vectors)]


def get_embeddings(texts: list[str], cache: EmbeddingCache | None = None) -> list[list[float]]:
    """Get embeddings, using cache when available and batching uncached texts in parallel."""
    if not texts:
        return []

    texts = [t[:8000] for t in texts]
    result: list[list[float]] = []

    to_fetch: list[tuple[int, str]] = []
    for i, t in enumerate(texts):
        cached = cache.get(t) if cache else None
        if cached is not None:
            result.append(cached)
        else:
            result.append([])
            to_fetch.append((i, t))

    if not to_fetch:
        return result

    # Parallel batches
    batch_size = 50
    batches = [to_fetch[i:i + batch_size] for i in range(0, len(to_fetch), batch_size)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_fetch_batch, b) for b in batches]
        for f in as_completed(futures):
            for idx, vec, raw_text in f.result():
                result[idx] = vec
                if cache and vec:
                    cache.put(raw_text, vec)

    if cache:
        cache.flush()

    return result


# ── Clustering ─────────────────────────────────────────────────────────────

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


def cluster_embeddings(embeds: list[list[float]]) -> list[list[int]]:
    """Two-pass clustering returning list of index groups.

    For small datasets (≤200) uses agglomerative merging.
    For larger datasets uses k-means primary.
    If numpy is available, uses vectorized operations.
    """
    n = len(embeds)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    if n <= 200:
        return _cluster_agglomerative(embeds, n)

    if HAS_NUMPY:
        mat = np.array(embeds, dtype=np.float32)
        return _cluster_kmeans_np(mat, n)
    return _cluster_kmeans_py(embeds, n)


def _cosine_sim_matrix(a, b):
    """Cosine similarity matrix — numpy rows × numpy rows."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    if b.ndim == 1:
        return a_norm @ (b / (np.linalg.norm(b) + 1e-10))
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


# ── Pure-Python path ──────────────────────────────────────────────────────

def _cluster_agglomerative(embeds: list[list[float]], n: int) -> list[list[int]]:
    """Original agglomerative — fine for small n."""
    clusters = [[i] for i in range(n)]
    centroids = [list(e) for e in embeds]
    threshold = CLUSTER_SIM

    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            if i >= len(clusters):
                break
            best_j = -1
            best_sim = threshold
            for j in range(len(clusters)):
                if i == j or j >= len(clusters):
                    continue
                sim = _cosine_sim(centroids[i], centroids[j])
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            if best_j >= 0:
                clusters[i].extend(clusters[best_j])
                m = len(clusters[i])
                centroids[i] = [
                    (centroids[i][k] * (m - len(clusters[best_j])) + centroids[best_j][k] * len(clusters[best_j])) / m
                    for k in range(len(embeds[0]))
                ]
                clusters.pop(best_j)
                centroids.pop(best_j)
                changed = True
                break

    if not clusters:
        return [[i for i in range(n)]]
    return _kmeans_refine_py(embeds, n, clusters, centroids, iters=3)


def _cluster_kmeans_py(embeds: list[list[float]], n: int) -> list[list[int]]:
    """K-means primary — pure-Python fallback for when numpy is unavailable."""
    import math, random
    dim = len(embeds[0])
    k = max(8, min(80, int(math.sqrt(n))))

    centroids = [list(embeds[random.randint(0, n - 1)]) for _ in range(k)]

    for _ in range(8):
        clusters: list[list[int]] = [[] for _ in range(k)]
        for idx in range(n):
            best_c = 0
            best_sim = -1.0
            for ci in range(k):
                sim = _cosine_sim(embeds[idx], centroids[ci])
                if sim > best_sim:
                    best_sim = sim
                    best_c = ci
            clusters[best_c].append(idx)
        for ci in range(k):
            if clusters[ci]:
                centroids[ci] = [0.0] * dim
                for idx in clusters[ci]:
                    for d in range(dim):
                        centroids[ci][d] += embeds[idx][d]
                sz = len(clusters[ci])
                for d in range(dim):
                    centroids[ci][d] /= sz

    pairs = [(cl, ce) for cl, ce in zip(clusters, centroids) if cl]
    clusters = [p[0] for p in pairs]
    centroids = [p[1] for p in pairs]

    min_size = max(3, int(n * MIN_CLUSTER_PCT))
    large: list[list[int]] = []
    large_centroids: list[list[float]] = []
    small: list[tuple[list[int], list[float]]] = []

    for cl, ce in zip(clusters, centroids):
        if len(cl) >= min_size:
            large.append(cl)
            large_centroids.append(ce)
        elif cl:
            small.append((cl, ce))

    for cl, ce in small:
        if large:
            best = max(range(len(large)), key=lambda li: _cosine_sim(ce, large_centroids[li]))
            large[best].extend(cl)
            m = len(large[best])
            large_centroids[best] = [
                (large_centroids[best][d] * (m - len(cl)) + ce[d] * len(cl)) / m
                for d in range(dim)
            ]

    result = large if large else [list(range(n))]
    return _kmeans_refine_py(embeds, n, result, large_centroids if large else centroids, iters=3)


def _kmeans_refine_py(embeds, n, clusters, centroids, iters=3):
    """K-means refinement — pure Python."""
    dim = len(embeds[0])
    for _ in range(iters):
        new_clusters: list[list[int]] = [[] for _ in range(len(clusters))]
        new_centroids = [[0.0] * dim for _ in range(len(clusters))]
        for idx in range(n):
            best_c = max(range(len(centroids)), key=lambda ci: _cosine_sim(embeds[idx], centroids[ci]))
            new_clusters[best_c].append(idx)
            for d in range(dim):
                new_centroids[best_c][d] += embeds[idx][d]
        for ci in range(len(new_centroids)):
            if new_clusters[ci]:
                sz = len(new_clusters[ci])
                for d in range(dim):
                    new_centroids[ci][d] /= sz
        clusters = [c for c in new_clusters if c]
        centroids = [c for c, cl in zip(new_centroids, new_clusters) if cl]
    return clusters


# ── NumPy-accelerated path ────────────────────────────────────────────────

def _cluster_kmeans_np(mat: np.ndarray, n: int) -> list[list[int]]:
    """K-means primary — numpy-accelerated for large n."""
    import random
    dim = mat.shape[1]
    k = max(8, min(80, int(n ** 0.5)))

    # K-means++ init
    centroids = np.zeros((k, dim), dtype=np.float32)
    centroids[0] = mat[random.randint(0, n - 1)]
    for ci in range(1, k):
        sims = _cosine_sim_matrix(mat, centroids[:ci])
        max_sims = sims.max(axis=1)
        dists = np.maximum(1.0 - max_sims, 0.0) + 1e-10
        probs = dists / dists.sum()
        idx = int(np.random.choice(n, p=probs))
        centroids[ci] = mat[idx]

    # 8 k-means iterations
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(8):
        labels = _cosine_sim_matrix(mat, centroids).argmax(axis=1)
        for ci in range(k):
            mask = labels == ci
            if mask.any():
                centroids[ci] = mat[mask].mean(axis=0)

    # Build cluster lists
    clusters: list[list[int]] = [[] for _ in range(k)]
    for idx in range(n):
        clusters[labels[idx]].append(idx)

    # Drop empty, merge small
    min_size = max(3, int(n * MIN_CLUSTER_PCT))
    large: list[list[int]] = []
    large_centroids: list[np.ndarray] = []

    for cl in clusters:
        if len(cl) >= min_size:
            large.append(cl)
            large_centroids.append(centroids[labels[cl[0]]].copy())
        elif cl:
            # Absorb into nearest large
            if large:
                sims = _cosine_sim_matrix(centroids[labels[cl[0]]].reshape(1, -1), np.array(large_centroids))[0]
                best = int(sims.argmax())
                large[best].extend(cl)
                m = len(large[best])
                large_centroids[best] = (large_centroids[best] * (m - len(cl)) + centroids[labels[cl[0]]] * len(cl)) / m

    result = large if large else [list(range(n))]
    final_centroids = np.array(large_centroids) if large else centroids
    return _kmeans_refine_np(mat, n, result, final_centroids, iters=3)


def _kmeans_refine_np(mat: np.ndarray, n: int, clusters: list[list[int]], centroids: np.ndarray, iters=3):
    """K-means refinement — numpy."""
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, -1)

    for _ in range(iters):
        labels = _cosine_sim_matrix(mat, centroids).argmax(axis=1)
        new_clusters: list[list[int]] = [[] for _ in range(len(centroids))]
        for idx in range(n):
            new_clusters[labels[idx]].append(idx)
        new_centroids = np.zeros_like(centroids)
        for ci in range(len(centroids)):
            if new_clusters[ci]:
                new_centroids[ci] = mat[new_clusters[ci]].mean(axis=0)
        keep = [ci for ci in range(len(new_centroids)) if new_clusters[ci]]
        clusters = [new_clusters[ci] for ci in keep]
        centroids = new_centroids[keep]
    return clusters


# ── LLM Labeling ───────────────────────────────────────────────────────────

def label_clusters(clusters: list[list[int]], queries: list[dict],
                   conv_title: str, model: str = "") -> dict[int, str]:
    """Batch-label all clusters via the labeling LLM.  Returns {cluster_idx: label}."""
    model = model or LABEL_MODEL
    if not clusters:
        return {}

    # Build representative samples for each cluster
    samples_per: list[str] = []
    for cl in clusters:
        # Sort by length (shorter = more focused) and take up to 3
        members = [(queries[i].get("content", "")[:200], i) for i in cl if i < len(queries)]
        members.sort(key=lambda x: len(x[0]))
        top = "\n".join(f"- {t}" for t, _ in members[:3])
        samples_per.append(top)

    prompt = (
        f"You are analyzing a conversation about \"{conv_title[:80]}\".\n"
        f"Below are {len(clusters)} groups of related user queries. "
        "For each group, generate ONE short intent label "
        "(2-5 words, snake_case) that captures what users are trying to do. "
        "Labels must be DISTINCT and SPECIFIC — avoid generic single words. "
        "Examples: understand_difficulty_mechanics, compare_conversion_rates, "
        "ask_protocol_definition, request_visual_explanation.\n\n"
    )
    for ci, samples in enumerate(samples_per):
        if samples:
            prompt += f"--- GROUP {ci} ---\n{samples}\n\n"

    prompt += (
        "Respond with ONLY a JSON object mapping group numbers to labels:\n"
        '{"0": "understand_difficulty_mechanics", "1": "compare_conversion_rates"}\n'
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

    return labels


# ── Main entry point ───────────────────────────────────────────────────────

def discover_intents(queries: list[dict], conv_id: str = "",
                     conv_title: str = "", model: str = "llama3.2:3b"
                     ) -> dict[str, list[dict]]:
    """Full intent discovery pipeline: embed → cluster → label → return groups.

    Returns {label: [{index, content, model, topics, answer}, ...]}
    """
    if not queries:
        return {"other": []}

    # Cap for performance
    if len(queries) > MAX_QUERIES:
        queries = queries[:MAX_QUERIES]

    texts = [q.get("content", "")[:400] for q in queries]
    cache = EmbeddingCache(conv_id) if conv_id else None
    embeds = get_embeddings(texts, cache)

    if not embeds or len(embeds) != len(queries):
        return {"other": list(queries)}

    # Cluster
    cluster_indices = cluster_embeddings(embeds)

    # Label (always uses fast model for speed)
    label_map = label_clusters(cluster_indices, queries, conv_title)

    # Build result
    result: dict[str, list[dict]] = {}
    for ci, indices in enumerate(cluster_indices):
        label = label_map.get(ci, "")
        if not label or label == "other":
            # Keyword fallback
            topic_counts: dict[str, int] = {}
            for idx in indices:
                if idx < len(queries):
                    for t in _extract_topics(queries[idx].get("content", "")):
                        topic_counts[t] = topic_counts.get(t, 0) + 1
            sorted_t = sorted(topic_counts.items(), key=lambda x: -x[1])
            label = sorted_t[0][0] if sorted_t else f"group_{ci}"

        if label in result:
            suffix = 2
            while f"{label}_{suffix}" in result:
                suffix += 1
            label = f"{label}_{suffix}"

        result[label] = []
        for idx in sorted(indices):
            if idx < len(queries):
                q = dict(queries[idx])
                q["topics"] = _extract_topics(q.get("content", ""))
                result[label].append(q)

    if not result:
        result["discussion"] = list(queries)

    return result


def discover_intents_deep(queries: list[dict], conv_id: str = "",
                          conv_title: str = "", model: str = "llama3.2:3b",
                          max_sub_cluster: int = 40
                          ) -> dict[str, dict | list]:
    """Like discover_intents but sub-clusters groups larger than max_sub_cluster.

    Returns {label: {queries: [...], sub_groups: {sub_label: [queries,...]}}}
    """
    top_groups = discover_intents(queries, conv_id, conv_title, model)
    result: dict[str, dict | list] = {}
    for label, qs in top_groups.items():
        if len(qs) > max_sub_cluster:
            subs = discover_intents(qs, conv_id + f"_{label}", conv_title, model)
            result[label] = {"queries": qs, "sub_groups": subs, "count": len(qs)}
        else:
            result[label] = {"queries": qs, "sub_groups": {}, "count": len(qs)}
    return result


def _extract_topics(text: str) -> list[str]:
    """Simple keyword-based topic extraction."""
    topics = []
    keywords = {
        "price_market": ["price", "cost", "market", "value", "incentive"],
        "mining_difficulty": ["miner", "mine", "difficulty", "hash", "reward"],
        "conversion_rate": ["convert", "rate", "ratio", "exchange", "swap"],
        "network_protocol": ["network", "node", "protocol", "consensus"],
        "stability_peg": ["stable", "stability", "peg", "anchor", "volatil"],
        "security_attack": ["security", "attack", "vulnerability", "risk", "sybil"],
        "performance_scale": ["performance", "speed", "latency", "throughput", "scal"],
        "token_economics": ["econom", "supply", "demand", "tokenomic", "inflation"],
        "governance_vote": ["govern", "vote", "proposal", "dao", "decision"],
        "technical_impl": ["implement", "code", "api", "sdk", "deploy", "architect"],
        "definition_explain": ["what is", "define", "explain", "meaning", "purpose"],
        "comparison": ["compare", "vs", "versus", "difference", "better"],
        "diagram_visual": ["diagram", "graph", "plot", "chart", "visual", "sketch", "draw"],
        "mechanism_how": ["how does", "how to", "mechanism", "process", "workflow"],
    }
    lower = text.lower()
    for topic, words in keywords.items():
        if any(w in lower for w in words):
            topics.append(topic)
    return topics[:2]


# ── Cross-conversation grouping ────────────────────────────────────────────

def cross_cluster_intents(per_conv_groups: dict[str, dict[str, list[dict]]],
                          model: str = "llama3.2:3b"
                          ) -> dict[str, list[dict]]:
    """Group intent clusters across multiple conversations into meta-intents.

    per_conv_groups: {conv_title: {intent_label: [queries, ...], ...}}
    Returns: {meta_intent_label: [{conv, orig_label, queries}, ...]}
    """
    # Collect all cluster centroids by embedding representative queries
    all_clusters: list[dict] = []  # {conv, orig_label, queries, centroid}
    for conv_title, groups in per_conv_groups.items():
        for label, qs in groups.items():
            if not qs:
                continue
            texts = [q.get("content", "")[:400] for q in qs[:10]]
            embeds = get_embeddings(texts, None)
            if embeds:
                centroid = [sum(v) / len(embeds) for v in zip(*embeds)]
            else:
                centroid = [0.0] * 768
            all_clusters.append({
                "conv": conv_title,
                "orig_label": label,
                "queries": qs,
                "centroid": centroid,
            })

    if not all_clusters:
        return {}

    # Cluster the centroids
    centroids = [c["centroid"] for c in all_clusters]
    meta_indices = cluster_embeddings(centroids)

    # Label meta-groups
    meta_samples = []
    for mi in meta_indices:
        samples = "\n".join(
            f"[{all_clusters[i]['conv']}] {all_clusters[i]['orig_label']}"
            for i in mi[:5]
        )
        meta_samples.append(samples)

    conv_names = list({c["conv"] for c in all_clusters})
    prompt = (
        f"Below are groups of conversation intent labels from these conversations: "
        f"{', '.join(conv_names[:5])}.\n"
        "For each meta-group, generate ONE short label (2-4 words, snake_case) "
        "that captures the common intent theme.\n\n"
    )
    for mi, (indices, samples) in enumerate(zip(meta_indices, meta_samples)):
        if samples:
            prompt += f"--- META GROUP {mi} ---\n{samples}\n\n"

    prompt += (
        "Respond with ONLY a JSON object mapping group numbers to labels:\n"
        '{"0": "mining_difficulty_discussion", "1": "token_economics_debate"}\n'
        "Labels:"
    )

    meta_labels: dict[int, str] = {}
    try:
        resp = requests.post(
            LABEL_URL,
            json={
                "model": LABEL_MODEL,
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
                meta_labels[int(k)] = label
    except Exception:
        pass

    result: dict[str, list[dict]] = {}
    for mi, indices in enumerate(meta_indices):
        label = meta_labels.get(mi, f"meta_group_{mi}")
        result[label] = []
        for idx in indices:
            if idx < len(all_clusters):
                result[label].append(all_clusters[idx])

    return result
