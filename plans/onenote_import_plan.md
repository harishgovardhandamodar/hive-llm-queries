# Microsoft OneNote Import — Feature Plan

## Overview
Import notebooks, sections, and pages from Microsoft OneNote into the app's journal notes system. Imported notes go through the same Ollama extraction pipeline (topics, concepts, intent) and appear as note nodes in the knowledge graph with the pink `#ff9bce` color.

---

## 1. Authentication (OAuth 2.0 with Microsoft Graph)

### Registration
- Register an app in the [Azure Portal](https://portal.azure.com) → App Registrations
- Redirect URI: `http://localhost:5001/api/onenote/auth/callback`
- Required API permissions (delegated):
  - `Notes.Read` — read OneNote notebooks, sections, pages
  - `Notes.Read.All` — optional, for reading shared notebooks
  - `offline_access` — for refresh tokens (long-lived sessions)
- No app-only auth supported by OneNote API — only delegated (user must sign in)

### Auth Flow (Authorization Code + PKCE)
```
Frontend                          Backend                          Microsoft
   │                                │                                │
   │ 1. Click "Import from OneNote" │                                │
   │───────────────────────────────>│                                │
   │                                │ 2. Store PKCE challenge        │
   │                                │ 3. Return auth URL             │
   │  <redirect to Microsoft>       │                                │
   │──────────────────────────────────────────────────────────────>  │
   │                                │                                │
   │  <user logs in, authorizes>    │                                │
   │                                │                                │
   │  <redirect to callback>        │                                │
   │───────────────────────────────>│                                │
   │                                │ 4. Exchange code for token     │
   │                                │──────────────────────────────> │
   │                                │  <access_token + refresh_token>│
   │                                │ 5. Store encrypted tokens      │
   │                                │ 6. Fetch notebooks list        │
   │  <show notebooks to select>    │                                │
   │<───────────────────────────────│                                │
```

### Token Storage
- Store `access_token`, `refresh_token`, `expires_at` in `.notes_store/onenote_tokens.json` (encrypted via Flask's `itsdangerous` or simple Fernet encryption)
- On startup, check if token is expired — if so, use refresh token
- If refresh fails (revoked), user must re-authenticate

### Backend Routes
```
GET  /api/onenote/auth          → returns Microsoft auth URL (redirect user here)
GET  /api/onenote/auth/callback → OAuth callback, exchanges code for token
GET  /api/onenote/auth/status   → check if authenticated (returns user info or false)
POST /api/onenote/auth/logout   → clear stored tokens
```

### Frontend Auth UI
- "Connect OneNote" button in the sidebar Notes section (below the + New Note button)
- Shows profile picture + email when connected
- "Disconnect" button to clear tokens

---

## 2. OneNote API Integration

### Endpoints Used
| Resource | Endpoint | Purpose |
|---|---|---|
| Notebooks | `GET /me/onenote/notebooks` | List all notebooks |
| Sections | `GET /me/onenote/sections` | List all sections (or per notebook) |
| Section Groups | `GET /me/onenote/sectionGroups` | List section groups for hierarchy |
| Pages | `GET /me/onenote/pages?top=100` | List recent pages |
| Page Content | `GET /me/onenote/pages/{id}/content` | Get page HTML content |
| Notebook Sections | `GET /me/onenote/notebooks/{id}/sections` | Sections within a notebook |
| Section Pages | `GET /me/onenote/sections/{id}/pages` | Pages within a section |

### Data Model Mapping
```
OneNote                   → Our Notes Store
─────────────────────────────────────────────
Notebook                  → (parent grouping, not stored)
Section                   → tag/label
Page                      → note
  Page.title              → note.title
  Page.content (HTML)     → note.content (plain text extracted)
  Page.createdTime        → note.date / note.created_at
  Page.lastModifiedTime   → note.updated_at
  Page.links (oneNoteWebUrl) → note.links[0]
  Page.images             → note.images (downloaded & stored in .notes_store/uploads/)
```

### Content Extraction (`onenote_import.py`)
Parse OneNote page HTML to extract:

1. **Text content** — strip HTML, preserve paragraph breaks, headings, lists, code blocks
2. **Images** — download from OneNote's content URLs, save to `.notes_store/uploads/<note_id>_<n>.png`, add to note.images
3. **Links** — extract hyperlinks, add to note.links
4. **Tables** — flatten to markdown-style text (optional)
5. **Metadata** — created time, last modified time, section name

```python
def extract_page_content(page_html: str, note_id: str) -> dict:
    """Parse OneNote page HTML, extract text + images + links."""
    soup = BeautifulSoup(page_html, 'html.parser')
    
    # Remove script/style elements
    for tag in soup(['script', 'style', 'meta']):
        tag.decompose()
    
    # Extract images before removing them
    images = []
    for i, img in enumerate(soup.find_all('img')):
        src = img.get('src', '')
        if src:
            filename = f"{note_id}_img{i}.png"
            download_image(src, UPLOADS_DIR / filename)
            images.append(filename)
    
    # Extract text content
    text = soup.get_text(separator='\n').strip()
    
    # Extract hyperlinks
    links = list(set(
        a['href'] for a in soup.find_all('a', href=True)
        if not a['href'].startswith('https://graph.microsoft.com')
    ))
    
    return {"content": text, "images": images, "links": links}
```

### Image Download
```python
def download_image(url: str, dest: Path) -> None:
    """Download image from OneNote content URL using auth token."""
    headers = {"Authorization": f"Bearer {get_token()}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
```

---

## 3. Notebook Browser UI

### New Modal: "Import from OneNote"
A multi-step modal (wizard-like) using the existing modal pattern:

```
Step 1: Connect
  [Connect to OneNote] button → Microsoft login
  Shows: "Connected as user@example.com"

Step 2: Select Notebooks
  □ Notebook 1 (12 sections)
  □ Notebook 2 (5 sections)
  □ Notebook 3 (8 sections)
  [Select All] [Next]

Step 3: Select Sections
  □ Section 1.1 (24 pages)
  □ Section 1.2 (15 pages)
  ...
  [Import Selected] [Back]
```

### Import Progress
- Show progress bar during import
- Display per-page status: "✓ Section 1.1 — Page 3 of 24"
- On completion: "Imported 47 pages from 3 sections"
- Notes appear in the Notes tab of the sidebar

### Backend Routes for Browser
```
GET  /api/onenote/notebooks     → list notebooks (requires auth)
GET  /api/onenote/notebooks/{id}/sections → list sections in notebook
GET  /api/onenote/sections/{id}/pages     → list pages in section (preview)
POST /api/onenote/import        → import selected sections/pages
```

### POST /api/onenote/import
```json
Request:
{
  "section_ids": ["section_id_1", "section_id_2"],
  "page_limit": 100,
  "model": "llama3.1:8b"
}

Response:
{
  "status": "importing",
  "total": 47,
  "completed": 0
}

Progress polling:
GET /api/onenote/import/status → {"total": 47, "completed": 12, "current": "Section 1.1"}
```

---

## 4. Implementation Order

### Phase 1 — Auth Backend
1. Add `msal` (Microsoft Authentication Library) to requirements
2. Create `onenote_import.py` with:
   - OAuth configuration (client_id, client_secret, redirect_uri, scopes)
   - Token storage & refresh logic
   - Auth endpoint handlers
3. Add API routes to `app.py`:
   - `GET /api/onenote/auth`
   - `GET /api/onenote/auth/callback`
   - `GET /api/onenote/auth/status`
   - `POST /api/onenote/auth/logout`

### Phase 2 — OneNote API Client
4. Add to `onenote_import.py`:
   - `list_notebooks()` → fetch and return notebook list
   - `list_notebook_sections(notebook_id)` → fetch sections
   - `list_section_pages(section_id, top=50)` → fetch page metadata
   - `get_page_content(page_id)` → fetch page HTML
   - `extract_page_content(html, note_id)` → parse HTML to text + images + links
   - `download_image(url, dest)` → download via authenticated GET
   - `import_section(section_id, limit, model)` → import all pages in a section

### Phase 3 — Import API
5. Add API routes:
   - `GET /api/onenote/notebooks`
   - `GET /api/onenote/notebooks/{id}/sections`
   - `GET /api/onenote/sections/{id}/pages`
   - `POST /api/onenote/import` — background import with progress tracking
   - `GET /api/onenote/import/status`

### Phase 4 — Frontend UI
6. Add "Connect OneNote" button to sidebar Notes section
7. Add OneNote import modal (multi-step wizard: connect → select notebooks → select sections → import)
8. Add import progress tracking (poll GET /api/onenote/import/status)
9. Show imported count in notes-badge

### Phase 5 — Content Quality
10. Handle HTML edge cases (tables, lists, code blocks, ink drawings)
11. Batch imports (rate limiting, pagination for large notebooks)
12. Incremental imports (skip pages already imported, check lastModifiedTime)

---

## 5. Key Decisions

### Why MSAL library instead of raw OAuth?
`msal` handles token caching, refresh, PKCE, and auth code exchange automatically. Reduces boilerplate and security bugs.

### Why BeautifulSoup for HTML parsing?
OneNote page content is HTML. BeautifulSoup is the standard Python HTML parser — handles malformed markup, image extraction, text normalization.

### Why separate import progress API?
Importing 100+ pages with Ollama extraction takes minutes. Background import with polling prevents HTTP timeouts and gives user feedback.

### Why not app-only auth?
Microsoft Graph OneNote API doesn't support app-only (client_credentials) authentication. Only delegated auth (user signed in) is supported, so each user must connect their own OneNote account.

### How to handle large notebooks?
- Paginate page lists: `GET /pages?top=100&skip=100`
- Rate limiting: 10k requests per 10 minutes per tenant
- Batch extraction: process pages in parallel (ThreadPoolExecutor, max_workers=3)
- Progress saved to `.notes_store/onenote_import_progress.json` for resumability

---

## 6. Files to Create/Modify

| File | Action |
|---|---|
| `onenote_import.py` | **Create** — auth, API client, content extraction |
| `app.py` | **Modify** — add OneNote API routes |
| `notes_store.py` | **Modify** — maybe add import tracking methods |
| `templates/dashboard.html` | **Modify** — add OneNote connect UI + import modal |
| `Dockerfile` | **Modify** — `pip install beautifulsoup4 msal` |
| `requirements.txt` | **Create or modify** — beautifulsoup4, msal, lxml |
| `.notes_store/` | Used for token storage, import progress |
| `plans/onenote_import_plan.md` | This file |
