"""Pre-build the extraction cache so the dashboard starts fast."""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/generate").replace("/api/generate", "/api/chat")
OLLAMA_MODEL = "llama3.2:3b"

CACHE_PATH = Path(".extraction_cache.json")

EXTRACTION_PROMPT = """{{"topics":["<1-4 topics>"],"concepts":["<3-8 concepts>"],"intent":"<category>","tags":["<2-6 tags>"],"summary":"<1 sentence>"}}

Valid intents: build, debug, explain, compare, design, optimize, explore, integrate, deploy, troubleshoot, analyze, research, other

Title: {title}
Messages:
{messages}

Return ONLY valid JSON (no other text)."""

# Load the newest JSON file from chatHistory
chat_files = sorted(Path("chatHistory").glob("*.json"), reverse=True)
if not chat_files:
    print("No chat history files found in chatHistory/")
    sys.exit(1)
CHAT_FILE = chat_files[0]
print(f"Loading {CHAT_FILE.name}")

with open(CHAT_FILE) as f:
    data = json.load(f)
convs = data.get("data", []) if isinstance(data, dict) else data
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


def extract_one(conv, retries=3):
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

    prompt = EXTRACTION_PROMPT.format(title=title, messages=msg_text)
    for attempt in range(retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
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
            if not isinstance(result.get("topics"), list):
                result["topics"] = []
            if not isinstance(result.get("concepts"), list):
                result["concepts"] = []
            if not isinstance(result.get("tags"), list):
                result["tags"] = []
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
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
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
