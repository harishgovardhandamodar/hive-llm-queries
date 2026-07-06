"""
Hive LLM Queries Dashboard
A knowledge-graph dashboard for exploring LLM conversation history.
Uses Ollama for semantic extraction of topics, concepts, and intents.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import threading
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, render_template, request

from hive_datatype import HiveGraph, Node, NodeType, Edge

from intent_engine import discover_intents_deep

app = Flask(__name__)

CHAT_HISTORY_PATH = Path(__file__).parent / "chatHistory"
CACHE_PATH = Path(__file__).parent / ".extraction_cache.json"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
AVAILABLE_MODELS = ["llama3.2:3b", "qwen3.6:35b-mlx"]
_current_model: str = OLLAMA_MODEL
MAX_WORKERS = 6

# ── Logging ─────────────────────────────────────────────────────────────────

_log_lock = threading.Lock()
_log_buffer: collections.deque = collections.deque(maxlen=500)
_log_seq: int = 0


def log(msg: str, level: str = "info", source: str = "") -> None:
    """Append a log entry.  Levels: info, success, warn, error."""
    global _log_seq
    with _log_lock:
        _log_seq += 1
        _log_buffer.append({
            "seq": _log_seq,
            "ts": time.time(),
            "level": level,
            "source": source,
            "msg": msg,
        })


def get_logs(since: int = 0) -> list[dict]:
    with _log_lock:
        if since:
            return [e for e in _log_buffer if e["seq"] > since]
        return list(_log_buffer)


# ── Helpers ────────────────────────────────────────────────────────────────

def load_chat_data() -> list[dict]:
    all_convs: list[dict] = []
    seen_ids: set[str] = set()
    for f in sorted(CHAT_HISTORY_PATH.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        convs = data.get("data", []) if isinstance(data, dict) else data
        for c in convs:
            cid = c.get("id") or c.get("_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                all_convs.append(c)
            elif not cid:
                all_convs.append(c)
    return all_convs


def get_user_queries_and_answers(conv: dict) -> list[dict]:
    messages = conv.get("chat", {}).get("messages", {})
    if isinstance(messages, dict):
        messages = list(messages.values())
    pairs = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            pairs.append({
                "role": "user",
                "content": msg.get("content", ""),
                "model": (msg.get("models") or [None])[0],
                "timestamp": msg.get("timestamp", 0),
            })
        elif role == "assistant":
            content_list = msg.get("content_list") or []
            answer_content = ""
            for item in content_list:
                if item.get("phase") == "answer" and item.get("content"):
                    answer_content = item["content"]
                    break
            pairs.append({
                "role": "assistant",
                "content": answer_content,
                "model": msg.get("model") or msg.get("modelName", ""),
                "timestamp": msg.get("timestamp", 0),
            })
    return pairs


# ── Ollama extraction ──────────────────────────────────────────────────────

EXTRACTION_PROMPT = """{{"topics":["<1-4 topics>"],"concepts":["<3-8 concepts>"],"intent":"<category>","tags":["<2-6 tags>"],"summary":"<1 sentence>"}}

Valid intents: build, debug, explain, compare, design, optimize, explore, integrate, deploy, troubleshoot, analyze, research, other

Title: {title}
Messages:
{messages}

Return ONLY valid JSON (no other text)."""


def _normalize_str_items(items) -> list[str]:
    if not items:
        return []
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            val = item.get("name") or item.get("label") or item.get("text") or str(item)
            result.append(val if isinstance(val, str) else str(val))
    return result


def extract_with_ollama(conv: dict) -> tuple[str, dict]:
    conv_id = conv.get("id", "")
    messages = get_user_queries_and_answers(conv)
    title = conv.get("title", "Untitled")
    msg_text = "\n".join(
        f"[{m['role'].upper()}] {m['content'][:500]}"
        for m in messages[:10]
    )

    prompt = EXTRACTION_PROMPT.format(title=title, messages=msg_text)

    try:
        chat_url = OLLAMA_URL.replace("/api/generate", "/api/chat")
        resp = requests.post(
            chat_url,
            json={
                "model": _current_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        raw = (resp.json().get("message", {}) or {}).get("content", "")
        raw = re.sub(r"^```(?:json)?\s*", "", raw).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            result = decoder.raw_decode(raw)[0]
        result["topics"] = _normalize_str_items(result.get("topics"))
        result["concepts"] = _normalize_str_items(result.get("concepts"))
        result["tags"] = _normalize_str_items(result.get("tags"))
        raw_intent = result.get("intent", "other")
        if isinstance(raw_intent, str):
            raw_intent = raw_intent.lower().strip()
        elif isinstance(raw_intent, list):
            raw_intent = (raw_intent[0] or "").lower().strip() if raw_intent else "other"
        else:
            raw_intent = str(raw_intent).lower().strip()
        VALID_INTENTS = {"build","debug","explain","compare","design","optimize","explore","integrate","deploy","troubleshoot","analyze","research","other"}
        result["intent"] = raw_intent if raw_intent in VALID_INTENTS else "other"
        return conv_id, result
    except Exception as e:
        return conv_id, {
            "topics": [],
            "concepts": [],
            "intent": "other",
            "tags": [],
            "summary": title,
        }


# ── Vector-based intent discovery ──────────────────────────────────────────

EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CLUSTER_SIM_THRESHOLD = float(os.environ.get("CLUSTER_SIM_THRESHOLD", "0.6"))
MAX_EMBED_QUERIES = 3000


def build_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


# ── Knowledge Graph Builder ─────────────────────────────────────────────────

def build_knowledge_graph(convs: list[dict]) -> dict:
    from notes_store import load_notes
    cache = build_cache()
    changed = False

    # Determine which conversations need extraction
    to_extract = []
    for conv in convs:
        conv_id = conv.get("id", "")
        if conv_id not in cache:
            to_extract.append(conv)
        elif not changed:
            changed = True

    # Parallel extraction for uncached conversations
    if to_extract:
        print(f"  Extracting entities from {len(to_extract)} conversations (workers={MAX_WORKERS})...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(extract_with_ollama, conv): conv for conv in to_extract}
            done = 0
            for f in as_completed(futures):
                conv_id, result = f.result()
                cache[conv_id] = result
                done += 1
                if done % 5 == 0:
                    print(f"    {done}/{len(to_extract)} done ({time.time()-t0:.0f}s)")
        print(f"  Extraction complete ({time.time()-t0:.0f}s)")
        save_cache(cache)
        changed = True

    # Entity registries
    topics: dict[str, dict] = {}
    concepts: dict[str, dict] = {}
    intents: dict[str, dict] = {}
    models: dict[str, dict] = {}
    conversations: list[dict] = []

    for conv in convs:
        conv_id = conv.get("id", "")
        title = conv.get("title", "Untitled")
        created = conv.get("created_at", 0)
        conv_models = conv.get("chat", {}).get("models", []) or []
        if isinstance(conv_models, str):
            conv_models = [conv_models]

        extraction = cache.get(conv_id, {})

        conv_topics = _normalize_str_items(extraction.get("topics", []))
        conv_concepts = _normalize_str_items(extraction.get("concepts", []))
        conv_intent = extraction.get("intent", "other")
        if isinstance(conv_intent, list):
            conv_intent = conv_intent[0] if conv_intent else "other"
        elif not isinstance(conv_intent, str):
            conv_intent = str(conv_intent)
        conv_tags = _normalize_str_items(extraction.get("tags", []))
        conv_summary = extraction.get("summary", title)

        for t in conv_topics:
            t_key = t.lower().strip()
            if t_key not in topics:
                topics[t_key] = {"id": f"topic_{t_key}", "label": t, "count": 0}
            topics[t_key]["count"] += 1

        for c in conv_concepts:
            c_key = c.lower().strip()
            if c_key not in concepts:
                concepts[c_key] = {"id": f"concept_{c_key}", "label": c, "count": 0}
            concepts[c_key]["count"] += 1

        i_key = conv_intent.lower().strip()
        if i_key not in intents:
            intents[i_key] = {"id": f"intent_{i_key}", "label": conv_intent, "count": 0}
        intents[i_key]["count"] += 1

        for m in conv_models:
            m_key = m.lower().strip()
            if m_key not in models:
                models[m_key] = {"id": f"model_{m_key}", "label": m, "count": 0}
            models[m_key]["count"] += 1

        msg_count = len(get_user_queries_and_answers(conv))

        conversations.append({
            "id": conv_id,
            "title": title,
            "summary": conv_summary,
            "timestamp": created,
            "model": conv_models[0] if conv_models else "unknown",
            "models": conv_models,
            "tags": conv_tags,
            "intent": conv_intent,
            "topics": conv_topics,
            "concepts": conv_concepts,
            "message_count": msg_count,
        })

    # Build HiveGraph
    hg = HiveGraph(id="llm-queries-dashboard")

    for c in conversations:
        nid = c["id"]
        hg.nodes.append(Node(
            id=nid,
            type=NodeType.PAPER,
            label=(c["title"] or "")[:60],
            published=str(c.get("timestamp") or ""),
            abstract=((c.get("summary") or "")[:300]),
        ))

    topic_node_ids = {}
    for t_key, t_info in topics.items():
        nid = t_info["id"]
        topic_node_ids[t_key] = nid
        hg.nodes.append(Node(
            id=nid, type=NodeType.CONCEPT, label=t_info["label"][:60], concept_type="topic",
        ))

    concept_node_ids = {}
    for c_key, c_info in concepts.items():
        nid = c_info["id"]
        concept_node_ids[c_key] = nid
        hg.nodes.append(Node(
            id=nid, type=NodeType.CONCEPT, label=c_info["label"][:60], concept_type="concept",
        ))

    intent_node_ids = {}
    for i_key, i_info in intents.items():
        nid = i_info["id"]
        intent_node_ids[i_key] = nid
        hg.nodes.append(Node(
            id=nid, type=NodeType.CONCEPT, label=i_info["label"][:60], concept_type="intent",
        ))

    model_node_ids = {}
    for m_key, m_info in models.items():
        nid = m_info["id"]
        model_node_ids[m_key] = nid
        hg.nodes.append(Node(
            id=nid, type=NodeType.CONCEPT, label=m_info["label"][:60], concept_type="model",
        ))

    edge_key = 0
    for c in conversations:
        cid = c["id"]
        for t in c.get("topics", []):
            tid = topic_node_ids.get(t.lower().strip())
            if tid:
                hg.edges.append(Edge(source=cid, target=tid, relation="related_to", key=edge_key))
                edge_key += 1
        for cpt in c.get("concepts", []):
            cid2 = concept_node_ids.get(cpt.lower().strip())
            if cid2:
                hg.edges.append(Edge(source=cid, target=cid2, relation="related_to", key=edge_key))
                edge_key += 1
        iid = intent_node_ids.get(c.get("intent", "").lower().strip())
        if iid:
            hg.edges.append(Edge(source=cid, target=iid, relation="related_to", key=edge_key))
            edge_key += 1
        for m in c.get("models", []):
            mid = model_node_ids.get(m.lower().strip())
            if mid:
                hg.edges.append(Edge(source=cid, target=mid, relation="uses", key=edge_key))
                edge_key += 1

    # Attach journal notes to the graph
    from notes_store import attach_notes_to_graph
    attach_notes_to_graph(hg)

    result = hg.to_node_link_dict()
    result["conversations"] = conversations
    result["stats"] = {
        "conversations": len(conversations),
        "topics": len(topics),
        "concepts": len(concepts),
        "intents": len(intents),
        "models": list(models.keys()),
        "notes": len(load_notes()),
        "total_messages": sum(c["message_count"] for c in conversations),
    }
    return result


# ── Flask Routes ───────────────────────────────────────────────────────────

_graph_data: dict | None = None
_build_in_progress = False


@app.route("/api/logs")
def api_logs():
    since = int(request.args.get("since", "0"))
    return jsonify(get_logs(since))


@app.route("/api/model", methods=["GET", "POST"])
def api_model():
    """Get or set the active Ollama model."""
    global _current_model, _graph_data
    if request.method == "POST":
        model = (request.json or {}).get("model", "")
        if model and model in AVAILABLE_MODELS:
            _current_model = model
            _graph_data = None
            return jsonify({"status": "ok", "model": _current_model, "available": AVAILABLE_MODELS})
        return jsonify({"error": f"Model must be one of {AVAILABLE_MODELS}", "available": AVAILABLE_MODELS}), 400
    return jsonify({"model": _current_model, "available": AVAILABLE_MODELS})


def get_graph_data(force: bool = False) -> dict:
    global _graph_data, _build_in_progress
    if _graph_data is not None and not force:
        return _graph_data
    if _build_in_progress:
        return _graph_data or {"error": "building", "nodes": [], "links": [], "conversations": [], "stats": {}}
    _build_in_progress = True
    try:
        convs = load_chat_data()
        _graph_data = build_knowledge_graph(convs)
    finally:
        _build_in_progress = False
    return _graph_data


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/graph")
def api_graph():
    force = request.args.get("force", "").lower() in ("1", "true")
    return jsonify(get_graph_data(force))


@app.route("/api/conversation/<conv_id>")
def api_conversation(conv_id: str):
    convs = load_chat_data()
    for conv in convs:
        if conv.get("id") == conv_id:
            messages = get_user_queries_and_answers(conv)
            conv_models = conv.get("chat", {}).get("models", []) or []
            return jsonify({
                "id": conv_id,
                "title": conv.get("title", "Untitled"),
                "messages": messages,
                "created_at": conv.get("created_at", 0),
                "models": conv_models,
            })
    return jsonify({"error": "not found"}), 404


@app.route("/api/conversation-subgraph/<conv_id>")
def api_conversation_subgraph(conv_id: str):
    """Return a focused subgraph for a single conversation."""
    convs = load_chat_data()
    conv = None
    for c in convs:
        if c.get("id") == conv_id:
            conv = c
            break
    if not conv:
        return jsonify({"error": "not found"}), 404

    cache = build_cache()
    extraction = cache.get(conv_id, {})
    conv_topics = _normalize_str_items(extraction.get("topics", []))
    conv_concepts = _normalize_str_items(extraction.get("concepts", []))
    conv_intent = extraction.get("intent", "other")
    conv_tags = _normalize_str_items(extraction.get("tags", []))
    conv_summary = extraction.get("summary", conv.get("title", ""))
    conv_models = conv.get("chat", {}).get("models", []) or []

    messages = get_user_queries_and_answers(conv)
    title = conv.get("title", "Untitled")

    hg = HiveGraph(id=f"subgraph_{conv_id}")

    def add_node(nid: str, ntype: str, label: str, ctype: str = "") -> None:
        hg.nodes.append(Node(
            id=nid, type=ntype, label=label[:60],
            concept_type=ctype,
        ))

    def add_edge(src: str, tgt: str, rel: str = "related_to") -> None:
        hg.edges.append(Edge(source=src, target=tgt, relation=rel))

    # Center conversation node
    conv_nid = f"conv_{conv_id}"
    add_node(conv_nid, "graph_paper", title)

    # Summary
    if conv_summary:
        sum_nid = f"summary_{conv_id}"
        add_node(sum_nid, "concept", conv_summary[:60], "summary")
        add_edge(conv_nid, sum_nid, "describes")

    # Topics
    for t in conv_topics:
        tid = f"topic_{t.lower().strip()}"
        add_node(tid, "concept", t, "topic")
        add_edge(conv_nid, tid, "related_to")

    # Concepts
    for cpt in conv_concepts:
        cid = f"concept_{cpt.lower().strip()}"
        add_node(cid, "concept", cpt, "concept")
        add_edge(conv_nid, cid, "related_to")

    # Intent
    if conv_intent:
        iid = f"intent_{conv_intent.lower().strip()}"
        add_node(iid, "concept", conv_intent, "intent")
        add_edge(conv_nid, iid, "has_intent")

    # Models
    for m in conv_models:
        mid = f"model_{m.lower().strip()}"
        add_node(mid, "concept", m, "model")
        add_edge(conv_nid, mid, "uses")

    # Messages (queries and answers)
    for i, msg in enumerate(messages):
        mid = f"msg_{conv_id}_{i}"
        role = msg.get("role", "user")
        content = (msg.get("content") or "")[:80]
        label = f"[{role.upper()}] {content}"
        add_node(mid, "paper" if role == "user" else "graph_paper", label, role)

        if role == "user":
            add_edge(conv_nid, mid, "contains")
            # Connect user query to topics
            for t in conv_topics:
                add_edge(mid, f"topic_{t.lower().strip()}", "about")
            # Connect user query to intent
            if conv_intent:
                add_edge(mid, f"intent_{conv_intent.lower().strip()}", "expresses")
        else:
            # Connect answer to the previous user query
            if i > 0 and messages[i - 1].get("role") == "user":
                prev_mid = f"msg_{conv_id}_{i - 1}"
                add_edge(prev_mid, mid, "answers")
            add_edge(conv_nid, mid, "contains")

    # Tags as nodes
    for tag in conv_tags:
        tid = f"tag_{tag.lower().strip().replace(' ', '_')}"
        add_node(tid, "concept", tag, "tag")
        add_edge(conv_nid, tid, "tagged")

    # Vector-based intent discovery via intent_engine
    user_queries = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            user_queries.append({
                "index": i,
                "content": msg.get("content", ""),
                "model": msg.get("model", ""),
                "intent": "",
                "topics": [],
                "conv_title": title,
                "answer": messages[i + 1].get("content", "") if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant" else "",
            })

    message_groups = discover_intents_deep(user_queries, conv_id=conv_id, conv_title=title, model=_current_model, max_sub_cluster=30)

    result = hg.to_node_link_dict()
    result["conversation"] = {
        "id": conv_id,
        "title": title,
        "summary": conv_summary,
        "intent": conv_intent,
        "models": conv_models,
        "topics": conv_topics,
        "concepts": conv_concepts,
        "tags": conv_tags,
        "message_count": len(messages),
    }
    result["messages"] = messages
    result["message_groups"] = message_groups
    return jsonify(result)


@app.route("/api/refresh")
def api_refresh():
    get_graph_data(force=True)
    return jsonify({"status": "ok"})


@app.route("/api/import-json", methods=["POST"])
def api_import_json():
    """Import a chat-history JSON file (upload or paste)."""
    try:
        raw = None
        submitted_filename = None

        # File upload via multipart
        if request.files:
            file = request.files.get("file")
            if file:
                raw = file.read().decode("utf-8")
                submitted_filename = file.filename or "upload.json"

        # Pasted JSON via request body
        if raw is None:
            raw = request.get_data(as_text=True)
            submitted_filename = "pasted.json"

        if not raw:
            return jsonify({"error": "No JSON data provided"}), 400

        data = json.loads(raw)

        # Validate — expect a dict with a "data" key (list) or a top-level list
        if isinstance(data, dict) and "data" in data:
            if not isinstance(data["data"], list):
                return jsonify({"error": "Expected 'data' to be a list"}), 400
        elif isinstance(data, list):
            data = {"success": True, "data": data}
        else:
            return jsonify({"error": "Expected a JSON object with a 'data' array, or a top-level array"}), 400

        # Save to chatHistory
        stem = Path(submitted_filename).stem
        timestamp = int(time.time())
        out_path = CHAT_HISTORY_PATH / f"chat-export-{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        # Invalidate graph cache and rebuild in background
        global _graph_data, _build_in_progress
        _graph_data = None
        _build_in_progress = True

        def rebuild():
            global _graph_data, _build_in_progress
            log("Graph rebuild started after import", "info", "graph")
            try:
                _graph_data = build_knowledge_graph(load_chat_data())
                log("Graph rebuild complete", "success", "graph")
            finally:
                _build_in_progress = False

        conv_count = sum(1 for c in (data.get("data", []) if isinstance(data, dict) else data))
        log(f"Imported {conv_count} conversations from {out_path.name}", "success", "import")
        threading.Thread(target=rebuild, daemon=True).start()

        return jsonify({
            "status": "imported",
            "filename": out_path.name,
            "conversations_added": conv_count,
            "rebuilding": True,
        })

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat-files")
def api_chat_files():
    """List available chat history files."""
    files = []
    for f in sorted(CHAT_HISTORY_PATH.glob("*.json"), reverse=True):
        files.append({
            "name": f.name,
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime,
        })
    return jsonify(files)


@app.route("/api/cross-intents")
def api_cross_intents():
    """Cross-conversation intent grouping."""
    from intent_engine import cross_cluster_intents, discover_intents_deep
    convs = load_chat_data()
    per_conv: dict[str, dict[str, list[dict]]] = {}
    for conv in convs[:20]:
        cid = conv.get("id", "")
        title = conv.get("title", "Untitled")
        messages = get_user_queries_and_answers(conv)
        user_qs = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                user_qs.append({
                    "index": i,
                    "content": msg.get("content", ""),
                    "model": msg.get("model", ""),
                    "conv_title": title,
                })
        if user_qs:
            groups = discover_intents_deep(user_qs, conv_id=cid, conv_title=title, model=_current_model)
            if groups:
                per_conv[title] = groups

    meta_groups = cross_cluster_intents(per_conv, model=_current_model)
    return jsonify({
        "total_conversations": len(per_conv),
        "meta_groups": {k: [{"conv": c["conv"], "orig_label": c["orig_label"], "count": len(c["queries"])}
                           for c in v] for k, v in meta_groups.items()},
    })


@app.route("/api/knowledge-store/status")
def api_ks_status():
    """Knowledge store build status and stats."""
    from knowledge_store import load_summary
    return jsonify(load_summary() or {"status": "not_built"})


@app.route("/api/knowledge-store/build", methods=["POST"])
def api_ks_build():
    """Build/rebuild the full knowledge store."""
    from knowledge_store import build_knowledge_store
    from intent_engine import get_embeddings

    log("Knowledge store build started", "info", "knowledge-store")
    convs = load_chat_data()
    cache = build_cache()
    stats = build_knowledge_store(
        convs, get_user_queries_and_answers, cache, get_embeddings, _current_model
    )
    log(
        f"Knowledge store complete: {stats.get('vector_entries',0)} vectors, "
        f"{stats.get('graph_nodes',0)} nodes in {stats.get('build_time_s',0)}s",
        "success", "knowledge-store",
    )
    return jsonify(stats)


@app.route("/api/knowledge-store/search")
def api_ks_search():
    """Semantic search across all indexed queries."""
    from knowledge_store import VectorIndex
    from intent_engine import get_embeddings

    q = request.args.get("q", "")
    top_k = int(request.args.get("top_k", "20"))
    if not q:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    vi = VectorIndex()
    results = vi.search_by_text(q, get_embeddings, top_k=top_k)
    return jsonify({"query": q, "results": results, "total_indexed": vi.count()})


@app.route("/api/knowledge-store/entity/<entity_type>")
def api_ks_entities(entity_type: str):
    """List entities of a given type (topics, concepts, intents)."""
    from knowledge_store import EntityRegistry
    er = EntityRegistry()
    min_count = int(request.args.get("min_count", "1"))
    entities = er.get_by_type(entity_type, min_count)
    return jsonify({"type": entity_type, "count": len(entities), "entities": entities})


@app.route("/api/knowledge-store/graph/<node_id>")
def api_ks_graph_node(node_id: str):
    """Get a node and its neighbors from the knowledge graph."""
    from knowledge_store import GraphIndex
    gi = GraphIndex()
    depth = int(request.args.get("depth", "1"))
    node = gi.get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    neighbors = gi.get_neighbors(node_id, max_depth=depth)
    return jsonify({
        "node": {"id": node_id, **node},
        "neighbors": neighbors,
        "neighbor_count": len(neighbors),
    })


@app.route("/api/knowledge-store/related")
def api_ks_related():
    """Find related entities across conversations."""
    from knowledge_store import EntityRegistry
    from intent_engine import get_embeddings

    etype = request.args.get("type", "topics")
    name = request.args.get("name", "")
    if not name:
        return jsonify({"error": "Missing 'name'"}), 400

    er = EntityRegistry()
    all_entities = er.get_by_type(etype)
    target = [e for e in all_entities if e.get("name", "").lower() == name.lower()]
    if not target:
        return jsonify({"error": "entity not found"}), 404

    related = target[0].get("related", {})
    return jsonify({
        "entity": target[0],
        "related_count": len(related),
        "related": related,
    })


# ── Cross-conversation Q&A Clustering ─────────────────────────────────────

_clusters_building = False


@app.route("/api/clusters/build", methods=["POST"])
def api_clusters_build():
    """Build cross-conversation Q&A clusters in background."""
    global _clusters_building
    if _clusters_building:
        return jsonify({"status": "already_building"})

    convs = load_chat_data()
    cache = build_cache()

    def build():
        global _clusters_building
        _clusters_building = True
        log("Cluster build started", "info", "clusters")
        try:
            from cluster_store import build_cross_clusters
            result = build_cross_clusters(
                convs, get_user_queries_and_answers, cache, _current_model,
                on_progress=lambda m: log(m, "info", "clusters"),
            )
            meta = result.get("metadata", {})
            log(
                f"Cluster build complete: {meta.get('total_clusters',0)} clusters, "
                f"{meta.get('total_pairs',0)} pairs in {meta.get('build_time_s',0)}s",
                "success", "clusters",
            )
        except Exception as e:
            log(f"Cluster build error: {e}", "error", "clusters")
            import traceback
            traceback.print_exc()
        finally:
            _clusters_building = False

    threading.Thread(target=build, daemon=True).start()
    return jsonify({"status": "building"})


@app.route("/api/clusters/build/status")
def api_clusters_build_status():
    return jsonify({"building": _clusters_building})


@app.route("/api/clusters")
def api_clusters():
    from cluster_store import get_clusters
    return jsonify(get_clusters())


@app.route("/api/clusters/<cluster_id>")
def api_cluster_detail(cluster_id: str):
    from cluster_store import get_cluster
    c = get_cluster(cluster_id)
    if c:
        return jsonify(c)
    return jsonify({"error": "not found"}), 404


@app.route("/api/clusters/search")
def api_clusters_search():
    from cluster_store import search_pairs
    q = request.args.get("q", "")
    top_k = int(request.args.get("top_k", "20"))
    if not q:
        return jsonify({"error": "Missing 'q' parameter"}), 400
    return jsonify({"query": q, "results": search_pairs(q, top_k)})


@app.route("/api/clusters/<cluster_id>/knowledge-graph")
def api_cluster_kg(cluster_id: str):
    from cluster_kg import get_cluster_knowledge_graph
    kg = get_cluster_knowledge_graph(cluster_id, request.args.get("model", "llama3.2:3b"))
    if kg is not None:
        return jsonify(kg)
    return jsonify({"error": "cluster not found"}), 404


@app.route("/api/clusters/<cluster_id>/summary")
def api_cluster_summary(cluster_id: str):
    from cluster_summary import get_cluster_summary
    result = get_cluster_summary(cluster_id, request.args.get("model", "llama3.2:3b"))
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


# ── Notes API ─────────────────────────────────────────────────────────────────

@app.route("/api/notes", methods=["GET", "POST"])
def api_notes():
    if request.method == "POST":
        from notes_store import create_note
        data = request.json or {}
        title = (data.get("title") or "").strip()
        content = (data.get("content") or "").strip()
        if not title or not content:
            return jsonify({"error": "title and content required"}), 400
        links = data.get("links", [])
        note = create_note(title, content, links, model=_current_model)
        return jsonify(note), 201
    from notes_store import load_notes
    notes = load_notes()
    return jsonify(notes)


@app.route("/api/notes/<note_id>", methods=["GET", "DELETE"])
def api_note_detail(note_id: str):
    from notes_store import get_note, delete_note
    if request.method == "DELETE":
        if delete_note(note_id):
            return jsonify({"status": "deleted"})
        return jsonify({"error": "not found"}), 404
    note = get_note(note_id)
    if note:
        return jsonify(note)
    return jsonify({"error": "not found"}), 404


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser

    # Load chat data in background so server starts immediately
    _build_in_progress = True  # prevent race: set BEFORE thread starts

    def _startup_build():
        global _graph_data, _build_in_progress
        log("Knowledge graph build started", "info", "graph")
        try:
            data = build_knowledge_graph(load_chat_data())
            _graph_data = data
            s = data.get("stats", {})
            log(
                f"Graph complete: {s.get('conversations',0)} conversations, "
                f"{s.get('topics',0)} topics, {s.get('concepts',0)} concepts, "
                f"{len(data.get('nodes',[]))} nodes, {len(data.get('links',[]))} edges",
                "success", "graph",
            )
        except Exception as e:
            log(f"Graph build error: {e}", "error", "graph")
            import traceback
            traceback.print_exc()
            _graph_data = {"error": str(e), "nodes": [], "links": [], "conversations": [], "stats": {}}
        finally:
            _build_in_progress = False

    bg = threading.Thread(target=_startup_build, daemon=True)
    bg.start()

    print("Starting dashboard at http://127.0.0.1:5001")
    print("  Graph is building in background — check back in a moment")
    webbrowser.open("http://127.0.0.1:5001")
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    app.run(host=host, debug=debug_mode, port=5001, use_reloader=False)
