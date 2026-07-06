"""Pre-build the extraction cache so the dashboard starts fast."""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, "/Users/harishgovardhandamodar/codebase/hive-datatype")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

CHAT_FILE = Path("chatHistory/chat-export-1783327480401.json")
CACHE_PATH = Path(".extraction_cache.json")

# Load data
with open(CHAT_FILE) as f:
    data = json.load(f)
convs = data["data"]
print(f"Loaded {len(convs)} conversations")

# Load existing cache
cache = {}
if CACHE_PATH.exists():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    print(f"Loaded cache with {len(cache)} entries")

# Find uncached
to_extract = [c for c in convs if c.get("id") not in cache]
print(f"Need to extract: {len(to_extract)} conversations")

if not to_extract:
    print("All cached, no extraction needed")
    sys.exit(0)


def extract_one(conv):
    conv_id = conv.get("id", "")
    title = conv.get("title", "Untitled")
    messages = conv.get("chat", {}).get("messages", {})
    if isinstance(messages, dict):
        messages = list(messages.values())
    msg_text = ""
    for m in messages[:10]:
        role = m.get("role", "?")
        content = m.get("content", "")[:500]
        if content:
            msg_text += f"[{role.upper()}] {content[:500]}\n"

    prompt = (
        "Analyze this LLM conversation and extract structured information.\n\n"
        f"Conversation title: {title}\n"
        "Messages:\n"
        f"{msg_text}\n"
        "Return a JSON object (no markdown, no extra text) with these keys:\n"
        '- "topics": array of 1-4 broad topic areas\n'
        '- "concepts": array of 3-8 key technical concepts\n'
        '- "intent": primary intent (build/debug/explain/compare/design/optimize/explore/integrate/deploy/troubleshoot/analyze/research/other)\n'
        '- "tags": array of 2-6 short descriptive tags\n'
        '- "summary": one sentence summary'
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        raw = resp.json().get("response", "")
        raw = re.sub(r"^```(?:json)?\s*", "", raw).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
        return conv_id, json.loads(raw)
    except Exception:
        return conv_id, {
            "topics": [],
            "concepts": [],
            "intent": "other",
            "tags": [],
            "summary": title,
        }


t0 = time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    futures = {ex.submit(extract_one, conv): conv for conv in to_extract}
    done = 0
    for f in as_completed(futures):
        conv_id, result = f.result()
        cache[conv_id] = result
        done += 1
        if done % 5 == 0:
            print(f"  {done}/{len(to_extract)} ({time.time()-t0:.0f}s)")

with open(CACHE_PATH, "w") as f:
    json.dump(cache, f, indent=2)
print(f"Done in {time.time()-t0:.0f}s, cached {len(cache)} entries")
