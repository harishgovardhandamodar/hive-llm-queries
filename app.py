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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, "/Users/harishgovardhandamodar/codebase/hive-datatype")
from hive_datatype import HiveGraph, Node, NodeType, Edge

app = Flask(__name__)

CHAT_HISTORY_PATH = Path(__file__).parent / "chatHistory"
CACHE_PATH = Path(__file__).parent / ".extraction_cache.json"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
MAX_WORKERS = 6

# ── Helpers ────────────────────────────────────────────────────────────────

def load_chat_data() -> list[dict]:
    for f in CHAT_HISTORY_PATH.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        return data.get("data", [])
    return []


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
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        raw = resp.json().get("response", "")
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
        if isinstance(conv_intent, dict):
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

        # Invalidate graph cache so it reloads all chat files
        global _graph_data
        _graph_data = None
        graph = get_graph_data(force=True)

        return jsonify({
            "status": "ok",
            "filename": out_path.name,
            "conversations": len(graph.get("conversations", [])),
            "stats": graph.get("stats", {}),
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


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser
    print("Loading chat history and extracting entities via Ollama...")
    data = get_graph_data()
    if data.get("error"):
        print("  Graph build in progress, starting server anyway...")
    else:
        s = data["stats"]
        print(f"  Conversations: {s['conversations']}")
        print(f"  Topics: {s['topics']}")
        print(f"  Concepts: {s['concepts']}")
        print(f"  Intents: {s['intents']}")
        print(f"  Models: {', '.join(s['models'])}")
        print(f"  Nodes: {len(data['nodes'])}")
        print(f"  Edges: {len(data['links'])}")
    print(f"\n  Dashboard at http://127.0.0.1:5001")
    webbrowser.open("http://127.0.0.1:5001")
    app.run(debug=True, port=5001, use_reloader=False)
