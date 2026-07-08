"""Microsoft OneNote import — OAuth 2.0 API client + .onepkg file parser."""

from __future__ import annotations

import io
import json
import re
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    from msal import ConfidentialClientApplication, PublicClientApplication
    HAS_MSAL = True
except ImportError:
    HAS_MSAL = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

from notes_store import STORE_DIR, UPLOADS_DIR, save_notes, load_notes, extract_note_entities
from notes_store import DEFAULT_MODEL

ONENOTE_DIR = STORE_DIR / "onenote"
ONENOTE_DIR.mkdir(parents=True, exist_ok=True)

TOKENS_PATH = ONENOTE_DIR / "tokens.json"
IMPORT_PROGRESS_PATH = ONENOTE_DIR / "import_progress.json"

# Microsoft Graph OAuth config (set ONENOTE_CLIENT_ID env var, or hardcode below)
import os as _os
CLIENT_ID = _os.environ.get("ONENOTE_CLIENT_ID", "YOUR_CLIENT_ID")
CLIENT_SECRET = _os.environ.get("ONENOTE_CLIENT_SECRET", "")
REDIRECT_URI = _os.environ.get("ONENOTE_REDIRECT_URI", "http://localhost:5001/api/onenote/auth/callback")
SCOPES = ["Notes.Read", "offline_access", "User.Read"]
AUTHORITY = "https://login.microsoftonline.com/common"
GRAPH_URL = "https://graph.microsoft.com/v1.0"


# ── Token Management ───────────────────────────────────────────────────────

def load_tokens() -> dict | None:
    if TOKENS_PATH.exists():
        try:
            return json.loads(TOKENS_PATH.read_text())
        except Exception:
            pass
    return None


def save_tokens(tokens: dict) -> None:
    TOKENS_PATH.write_text(json.dumps(tokens, indent=2))


def clear_tokens() -> None:
    if TOKENS_PATH.exists():
        TOKENS_PATH.unlink()


def get_access_token() -> str | None:
    tokens = load_tokens()
    if not tokens:
        return None
    if tokens.get("expires_at", 0) > time.time() + 60:
        return tokens["access_token"]
    return refresh_access_token(tokens)


def refresh_access_token(tokens: dict) -> str | None:
    if not HAS_MSAL or not tokens.get("refresh_token"):
        return None
    try:
        app = PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
        result = app.acquire_token_by_refresh_token(
            tokens["refresh_token"], scopes=SCOPES
        )
        if "access_token" in result:
            tokens["access_token"] = result["access_token"]
            tokens["expires_at"] = time.time() + result.get("expires_in", 3600)
            if "refresh_token" in result:
                tokens["refresh_token"] = result["refresh_token"]
            save_tokens(tokens)
            return result["access_token"]
    except Exception:
        pass
    return None


# ── Auth Routes Helpers ────────────────────────────────────────────────────

def build_auth_url() -> str | None:
    if not HAS_MSAL:
        return None
    app = PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
    return app.get_authorization_request_url(
        SCOPES,
        redirect_uri=REDIRECT_URI,
        state=str(uuid.uuid4()),
    )


def exchange_code(code: str) -> dict | None:
    if not HAS_MSAL:
        return None
    try:
        app = PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
        result = app.acquire_token_by_authorization_code(
            code, scopes=SCOPES, redirect_uri=REDIRECT_URI
        )
        if "access_token" in result:
            tokens = {
                "access_token": result["access_token"],
                "refresh_token": result.get("refresh_token", ""),
                "expires_at": time.time() + result.get("expires_in", 3600),
                "user": result.get("id_token_claims", {}),
            }
            save_tokens(tokens)
            return tokens
    except Exception:
        pass
    return None


# ── Microsoft Graph API Client ─────────────────────────────────────────────

def _graph_get(path: str, params: dict | None = None) -> dict | None:
    token = get_access_token()
    if not token:
        return None
    url = f"{GRAPH_URL}{path}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 401:
        # Token expired, try refresh once
        clear_tokens()
        token = get_access_token()
        if not token:
            return None
        headers["Authorization"] = f"Bearer {token}"
        resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 200:
        return resp.json()
    return None


def get_user_info() -> dict | None:
    return _graph_get("/me")


def list_notebooks() -> list[dict]:
    data = _graph_get("/me/onenote/notebooks")
    return (data or {}).get("value", [])


def list_notebook_sections(notebook_id: str) -> list[dict]:
    data = _graph_get(f"/me/onenote/notebooks/{notebook_id}/sections")
    return (data or {}).get("value", [])


def list_section_pages(section_id: str, top: int = 100) -> list[dict]:
    data = _graph_get(f"/me/onenote/sections/{section_id}/pages", {"top": top})
    return (data or {}).get("value", [])


def get_page_content(page_id: str) -> str:
    token = get_access_token()
    if not token:
        return ""
    url = f"{GRAPH_URL}/me/onenote/pages/{page_id}/content"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 200:
        return resp.text
    return ""


# ── Content Extraction ─────────────────────────────────────────────────────

def extract_page_content(page_html: str, note_id: str) -> dict:
    """Parse OneNote page HTML to extract text, images, and links."""
    if not page_html:
        return {"content": "", "images": [], "links": []}

    text = page_html
    images = []
    links = []

    if HAS_BS4:
        soup = BeautifulSoup(page_html, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style", "meta", "head"]):
            tag.decompose()

        # Extract images
        for i, img in enumerate(soup.find_all("img")):
            src = img.get("src", "")
            if src and not src.startswith("data:"):
                filename = f"{note_id}_img{i}.png"
                try:
                    download_image(src, UPLOADS_DIR / filename)
                    images.append(filename)
                except Exception:
                    pass

        # Extract links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href and not href.startswith("https://graph.microsoft.com"):
                links.append(href)

        # Get text content
        text = soup.get_text(separator="\n").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)

    return {"content": text[:10000], "images": images, "links": links[:20]}


def download_image(url: str, dest: Path) -> None:
    token = get_access_token()
    if not token:
        raise ValueError("No auth token")
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def extract_onenote_text_from_html(html: str) -> str:
    """Simple text extraction from OneNote HTML without BeautifulSoup."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:10000]


# ── .onepkg File Parser ────────────────────────────────────────────────────

def parse_onepkg(file_bytes: bytes) -> list[dict]:
    """Parse a .onepkg file (ZIP archive) and extract text content."""
    notes = []
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            for name in z.namelist():
                if name.lower().endswith(".one"):
                    content = z.read(name)
                    extracted = parse_one_file(content, name)
                    if extracted:
                        notes.append(extracted)
    except zipfile.BadZipFile:
        pass
    return notes


def parse_one_file(data: bytes, filename: str) -> dict | None:
    """Extract readable text from a .one (FSSHTTP) binary section file."""
    try:
        text = data.decode("utf-16-le", errors="replace")
    except Exception:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = ""

    # Extract meaningful strings (2+ chars, not control characters)
    words = re.findall(r"[\w\s\-.,!?;:'\"()]+", text)
    content = " ".join(w.strip() for w in words if len(w.strip()) > 3)
    content = re.sub(r"\s+", " ", content).strip()

    if len(content) < 20:
        return None

    section_name = Path(filename).stem.replace("_", " ").title()

    return {
        "title": section_name,
        "content": content[:10000],
        "links": [],
        "images": [],
        "source": f".onepkg/{filename}",
    }


# ── Import Logic ───────────────────────────────────────────────────────────

_import_progress: dict = {}


def get_import_progress() -> dict:
    return _import_progress


def import_section(section_id: str, section_name: str,
                   limit: int = 100, model: str = DEFAULT_MODEL) -> dict:
    """Import all pages from a OneNote section into notes store."""
    pages = list_section_pages(section_id, limit)
    if not pages:
        return {"imported": 0, "total": 0, "section": section_name}

    notes = load_notes()
    existing_ids = {n["id"] for n in notes}
    imported = 0
    total = len(pages)

    _import_progress.update({
        "status": "importing",
        "total": total,
        "completed": 0,
        "current": f"{section_name} — 0/{total}",
    })

    def process_page(page: dict) -> dict | None:
        page_id = page.get("id", "")
        page_title = page.get("title", "Untitled Page")
        created = page.get("createdTime", "")
        links = page.get("links", {})
        web_url = (links or {}).get("oneNoteWebUrl", {}).get("href", "")

        html = get_page_content(page_id)
        extracted = extract_page_content(html, f"onenote_{page_id[:8]}")
        content = extracted["content"]
        if not content:
            content = page_title

        note_id = f"note_on_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        extraction = extract_note_entities(page_title, content, model)
        imgs = extracted.get("images", [])
        urls = extracted.get("links", [])
        if web_url:
            urls.insert(0, web_url)

        return {
            "id": note_id,
            "title": page_title[:100],
            "content": content,
            "date": created[:10] if created else time.strftime("%Y-%m-%d"),
            "created_at": time.time(),
            "updated_at": time.time(),
            "links": urls[:10],
            "images": imgs,
            "source": f"onenote_section_{section_id}",
            "extraction": extraction,
        }

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(process_page, p): p for p in pages}
        for f in as_completed(futures):
            result = f.result()
            if result and result["id"] not in existing_ids:
                notes.append(result)
                existing_ids.add(result["id"])
                imported += 1
            _import_progress["completed"] = imported
            _import_progress["current"] = f"{section_name} — {imported}/{total}"

    save_notes(notes)
    _import_progress.update({
        "status": "done",
        "imported": imported,
        "total": total,
    })
    return {"imported": imported, "total": total, "section": section_name}


def import_onepkg(file_bytes: bytes, model: str = DEFAULT_MODEL) -> dict:
    """Import notes from a .onepkg file into notes store."""
    parsed = parse_onepkg(file_bytes)
    if not parsed:
        return {"imported": 0, "total": 0, "error": "No readable content found in .onepkg"}

    notes = load_notes()
    existing_ids = {n["id"] for n in notes}
    imported = 0
    total = len(parsed)

    _import_progress.update({
        "status": "importing",
        "total": total,
        "completed": 0,
        "source": ".onepkg",
    })

    for entry in parsed:
        note_id = f"note_onpkg_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        if note_id in existing_ids:
            continue
        extraction = extract_note_entities(entry["title"], entry["content"], model)
        note = {
            "id": note_id,
            "title": entry["title"],
            "content": entry["content"],
            "date": time.strftime("%Y-%m-%d"),
            "created_at": time.time(),
            "updated_at": time.time(),
            "links": entry.get("links", []),
            "images": entry.get("images", []),
            "source": entry.get("source", ".onepkg"),
            "extraction": extraction,
        }
        notes.append(note)
        existing_ids.add(note_id)
        imported += 1
        _import_progress["completed"] = imported

    save_notes(notes)
    _import_progress.update({"status": "done", "imported": imported, "total": total})
    return {"imported": imported, "total": total}
