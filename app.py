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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, "/Users/harishgovardhandamodar/codebase/hive-datatype")
from hive_datatype import HiveGraph, Node, NodeType, Edge

from intent_engine import discover_intents as ie_discover_intents

app = Flask(__name__)

CHAT_HISTORY_PATH = Path(__file__).parent / "chatHistory"
CACHE_PATH = Path(__file__).parent / ".extraction_cache.json"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
AVAILABLE_MODELS = ["llama3.2:3b", "qwen3.6:35b-mlx"]
_current_model: str = OLLAMA_MODEL
MAX_WORKERS = 6

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

EXTRACTION_PROMPT = """Analyze this LLM conversation and extract structured information.

Conversation title: {title}
Messages:
{messages}

Return a JSON object (no markdown, no extra text) with these keys:
- "topics": array of 1-4 broad topic areas (e.g. ["machine learning", "web development"])
- "concepts": array of 3-8 key technical concepts or entities mentioned
- "intent": the primary intent category (one of: build, debug, explain, compare, design, optimize, explore, integrate, deploy, troubleshoot, analyze, research, other)
- "tags": array of 2-6 short descriptive tags
- "summary": one sentence summary of what this conversation is about
"""


def _normalize_str_items(items: list) -> list[str]:
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
        result = json.loads(raw)
        result["topics"] = _normalize_str_items(result.get("topics", []))
        result["concepts"] = _normalize_str_items(result.get("concepts", []))
        result["tags"] = _normalize_str_items(result.get("tags", []))
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

    result = hg.to_node_link_dict()
    result["conversations"] = conversations
    result["stats"] = {
        "conversations": len(conversations),
        "topics": len(topics),
        "concepts": len(concepts),
        "intents": len(intents),
        "models": list(models.keys()),
        "total_messages": sum(c["message_count"] for c in conversations),
    }
    return result


# ── Flask Routes ───────────────────────────────────────────────────────────

_graph_data: dict | None = None
_build_in_progress = False


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

    message_groups = ie_discover_intents(user_queries, conv_id=conv_id, conv_title=title, model=_current_model)

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
            try:
                _graph_data = build_knowledge_graph(load_chat_data())
            finally:
                _build_in_progress = False

        conv_count = sum(1 for c in (data.get("data", []) if isinstance(data, dict) else data))
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
    from intent_engine import cross_cluster_intents
    convs = load_chat_data()
    per_conv: dict[str, dict[str, list[dict]]] = {}
    for conv in convs[:20]:  # max 20 convs for performance
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
            groups = ie_discover_intents(user_qs, conv_id=cid, conv_title=title, model=_current_model)
            if groups:
                per_conv[title] = groups

    meta_groups = cross_cluster_intents(per_conv, model=_current_model)
    return jsonify({
        "total_conversations": len(per_conv),
        "meta_groups": {k: [{"conv": c["conv"], "orig_label": c["orig_label"], "count": len(c["queries"])}
                           for c in v] for k, v in meta_groups.items()},
    })


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser

    # Load chat data in background so server starts immediately
    def _startup_build():
        global _graph_data, _build_in_progress
        _build_in_progress = True
        try:
            data = build_knowledge_graph(load_chat_data())
            _graph_data = data
            s = data.get("stats", {})
            print(f"  Conversations: {s.get('conversations', 0)}")
            print(f"  Topics: {s.get('topics', 0)}")
            print(f"  Concepts: {s.get('concepts', 0)}")
            print(f"  Intents: {s.get('intents', 0)}")
            print(f"  Models: {', '.join(s.get('models', []))}")
            print(f"  Nodes: {len(data.get('nodes', []))}")
            print(f"  Edges: {len(data.get('links', []))}")
            print("  Background build complete")
        finally:
            _build_in_progress = False

    bg = threading.Thread(target=_startup_build, daemon=True)
    bg.start()

    print("Starting dashboard at http://127.0.0.1:5001")
    print("  Graph is building in background — check back in a moment")
    webbrowser.open("http://127.0.0.1:5001")
    app.run(debug=True, port=5001, use_reloader=False)
