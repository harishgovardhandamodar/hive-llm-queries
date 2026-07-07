"""Journal Notes — CRUD, persistence, and Ollama extraction for dated notes."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

import requests

from hive_datatype import HiveGraph, Node, NodeType, Edge

STORE_DIR = Path(__file__).parent / ".notes_store"
NOTES_PATH = STORE_DIR / "notes.json"
UPLOADS_DIR = STORE_DIR / "uploads"
STORE_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

NOTES_EXTRACTION_PROMPT = """{{"topics":["<1-4 topics>"],"concepts":["<3-8 concepts>"],"intent":"<category>","tags":["<2-6 tags>"],"summary":"<1 sentence>"}}

Valid intents: build, debug, explain, compare, design, optimize, explore, integrate, deploy, troubleshoot, analyze, research, other

Title: {title}
Content:
{content}

Return ONLY valid JSON (no other text)."""


def load_notes() -> list[dict]:
    if NOTES_PATH.exists():
        try:
            return json.loads(NOTES_PATH.read_text())
        except Exception:
            pass
    return []


def save_notes(notes: list[dict]) -> None:
    NOTES_PATH.write_text(json.dumps(notes, indent=2))


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


def extract_note_entities(title: str, content: str, model: str = DEFAULT_MODEL) -> dict:
    prompt = NOTES_EXTRACTION_PROMPT.format(title=title, content=content)
    chat_url = OLLAMA_URL.replace("/api/generate", "/api/chat")
    try:
        resp = requests.post(
            chat_url,
            json={
                "model": model,
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
        return result
    except Exception:
        return {
            "topics": [], "concepts": [], "intent": "other",
            "tags": [], "summary": title,
        }


def create_note(title: str, content: str, links: list[str] | None = None,
                image_filenames: list[str] | None = None,
                model: str = DEFAULT_MODEL) -> dict:
    notes = load_notes()
    note_id = f"note_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    now = time.time()
    extraction = extract_note_entities(title, content, model)
    note = {
        "id": note_id,
        "title": title,
        "content": content,
        "date": time.strftime("%Y-%m-%d", time.localtime(now)),
        "created_at": now,
        "updated_at": now,
        "links": links or [],
        "images": image_filenames or [],
        "extraction": extraction,
    }
    notes.append(note)
    save_notes(notes)
    return note


def get_note(note_id: str) -> dict | None:
    for n in load_notes():
        if n["id"] == note_id:
            return n
    return None


def delete_note(note_id: str) -> bool:
    notes = load_notes()
    filtered = [n for n in notes if n["id"] != note_id]
    if len(filtered) == len(notes):
        return False
    save_notes(filtered)
    return True


def attach_notes_to_graph(hg: HiveGraph) -> None:
    notes = load_notes()
    if not notes:
        return

    existing_ids = {n.id for n in hg.nodes}

    for note in notes:
        nid = note["id"]
        if nid in existing_ids:
            continue

        extraction = note.get("extraction", {})
        hg.nodes.append(Node(
            id=nid,
            type=NodeType.CONCEPT,
            label=(note.get("title") or "Untitled Note")[:60],
            concept_type="note",
            definition=(note.get("content") or "")[:200],
            published=str(note.get("date", "")),
        ))
        existing_ids.add(nid)

        topics = _normalize_str_items(extraction.get("topics", []))
        concepts = _normalize_str_items(extraction.get("concepts", []))
        intent = extraction.get("intent", "")

        edge_key = len(hg.edges)

        for t in topics:
            tk = t.lower().strip()
            tid = None
            for node in hg.nodes:
                if node.concept_type == "topic" and node.label.lower().strip() == tk:
                    tid = node.id
                    break
            if tid:
                hg.edges.append(Edge(source=nid, target=tid, relation="related_to", key=edge_key))
                edge_key += 1

        for c in concepts:
            ck = c.lower().strip()
            cid = None
            for node in hg.nodes:
                if node.concept_type == "concept" and node.label.lower().strip() == ck:
                    cid = node.id
                    break
            if cid:
                hg.edges.append(Edge(source=nid, target=cid, relation="related_to", key=edge_key))
                edge_key += 1

        if intent:
            ik = intent.lower().strip()
            iid = None
            for node in hg.nodes:
                if node.concept_type == "intent" and node.label.lower().strip() == ik:
                    iid = node.id
                    break
            if iid:
                hg.edges.append(Edge(source=nid, target=iid, relation="related_to", key=edge_key))
