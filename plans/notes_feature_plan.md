# Dated Journal & Notes — Feature Plan

## Overview
Add a journal/notes system where users can save dated text entries (with optional web links, screenshots, pictures). On save, the note is analyzed by Ollama (same extraction pipeline as conversations) and attached to the knowledge graph as a new node type with a distinct color.

---

## 1. Data Model

### Note (Python dataclass, stored in `.notes_store/notes.json`)
```python
{
  "id": "note_20260707_001",
  "title": "Meeting notes on Quai stability",
  "content": "Full text of the note...",
  "date": "2026-07-07",
  "created_at": 1783372800.0,
  "updated_at": 1783372800.0,
  "links": ["https://example.com/article"],
  "images": ["note_20260707_001_img0.png"],   # filenames in .notes_store/uploads/
  "extraction": {
    "topics": ["Quai Stability", "Mining Difficulty"],
    "concepts": ["Proof-of-Work", "Exchange Rate Controller"],
    "intent": "research",
    "tags": ["blockchain", "crypto"],
    "summary": "Notes discussing Quai stability mechanisms..."
  }
}
```

### Storage
- **File:** `.notes_store/notes.json` — array of note objects
- **Uploads:** `.notes_store/uploads/` — timestamped image/screenshot files
- **Persistence:** JSON file-based, same pattern as `.extraction_cache.json`

---

## 2. Backend — New Files & Changes

### New file: `notes_store.py`
Role: CRUD for notes, Ollama extraction, graph integration.

```
Functions:
  load_notes() -> list[dict]
  save_notes(notes: list[dict]) -> None
  create_note(title, content, links, images) -> dict
  get_note(note_id) -> dict | None
  delete_note(note_id) -> bool
  extract_note_entities(note: dict) -> dict    # calls Ollama
  attach_note_to_graph(note: dict, hg: HiveGraph) -> None   # adds node + edges
```

### Changes to `app.py`
```
New routes:
  POST /api/notes              — create a note (multipart: text + images)
  GET  /api/notes              — list all notes (with optional ?date= filter)
  GET  /api/notes/<id>         — get single note detail
  DELETE /api/notes/<id>       — delete a note
  POST /api/notes/<id>/reprocess — re-run extraction on a note

Modified routes:
  GET /api/graph               — include note nodes in the graph data

New extraction prompt (shorter, for notes):
  NOTES_EXTRACTION_PROMPT = """Same JSON template as EXTRACTION_PROMPT
  but adapted for note/journal content instead of conversation messages."""
```

### Extraction Pipeline (reuses existing pattern)
```
1. create_note() saves to .notes_store/notes.json
2. Calls extract_with_ollama_note() using NOTES_EXTRACTION_PROMPT
3. Result stored in note["extraction"]
4. On next graph build, attach_note_to_graph() adds:
   - A Node(type=CONCEPT, concept_type="note", label=title)
   - Edges to matching topic/concept nodes (by name matching)
```

### Graph Integration (`attach_note_to_graph`)
```
For each note in notes_store:
  1. Create a note node: Node(id="note_<id>", type=CONCEPT, concept_type="note", label=title)
  2. For each topic in note.extraction.topics:
     - Find existing topic node by name
     - Create Edge(source=note_id, target=topic_id, relation="related_to")
  3. For each concept in note.extraction.concepts:
     - Find existing concept node by name
     - Create Edge(source=note_id, target=concept_id, relation="related_to")
  4. For intent:
     - Find existing intent node
     - Create Edge(source=note_id, target=intent_id, relation="related_to")
```

Note nodes connect to **existing** concept/topic/intent nodes. If an extracted concept doesn't exist yet, a new concept node is created (same as how conversations work).

---

## 3. Frontend — Changes to `dashboard.html`

### New node type: "note"
```javascript
// In getNodeType (add before return 'concept'):
if (n.concept_type === 'note') return 'note';

// In getNodeColor:
case 'note': return '#ff9bce';  // pink/magenta — distinct from all other types

// In getNodeRadius:
case 'note': return 10;

// New CSS class:
.node-note { fill: #ff9bce; background: #ff9bce; }

// Legend entry (before controls-hint):
<div class="legend-item"><span class="legend-dot node-note"></span>Notes</div>
```

### Notes UI in Sidebar
Insert a new `div.filters` section between the Analysis Model selector and the conversation list:

```
┌─ Journal Notes ─────────────────────┐
│  📝 [Title input...]                │
│  📄 [Content textarea...]            │
│  🔗 [Link input...]                  │
│  🖼 [Drop zone or upload button]     │
│  [💾 Save Note]                      │
│  ─────────────────────────────────   │
│  📌 Jul 7 — Meeting notes (3)       │  ← clickable note list
│  📌 Jul 6 — Research links (1)      │
│  📌 Jul 5 — Screenshot notes (0)    │
└──────────────────────────────────────┘
```

### Note Creation Modal
Using the existing modal pattern (same as Import Modal):
```
Title input
Content textarea (main body)
Link input (with +Add button, renders as tag list)
Image drop zone (drag & drop screenshots)
Save button
```

### Notes List
- Show last 5-10 notes in the sidebar section
- Each entry shows: date + title + count of links/images
- Clicking a note opens it in a detail view (inline or in the graph)
- Click date to filter graph to show only that date's notes

### Graph Filtering
- Notes appear in the graph only when the "Conversations" or "All" tab is active
- A "Notes" toggle could be added to the filter bar to show/hide note nodes
- Filtering by topic/intent also filters notes (they share the same extracted fields)

---

## 4. Image Handling

### Upload flow
```
1. User drags image onto drop zone or clicks to browse
2. File read client-side, preview shown
3. On save, images sent as multipart to POST /api/notes
4. Server saves to .notes_store/uploads/<note_id>_<n>.<ext>
5. Only filename stored in note["images"], actual filesystem path resolved on read
```

### Supported formats
- PNG, JPG, GIF, WebP (screenshots, pictures)
- Max file size: 10MB per image (configurable)
- Images are NOT embedded in the graph — they're attached to the note node

### Serving images
```
New route: GET /api/notes/<id>/images/<filename>
Returns the image file with proper Content-Type.
Frontend can reference: <img src="/api/notes/<id>/images/<filename>">
```

---

## 5. Web Links Handling

- Links stored as strings in `note["links"]`
- On save, links are included in the extraction prompt context (e.g., "Content references: https://...")
- Links displayed as clickable tags in the note detail view
- No automatic fetching/content extraction from links (privacy/scope concern)

---

## 6. Implementation Order

### Phase 1 — Core CRUD + Extraction
1. Create `notes_store.py` with load/save/create/get/delete
2. Create `NOTES_EXTRACTION_PROMPT` in `app.py`
3. Create `extract_with_ollama_note()` in `app.py`
4. Add `POST /api/notes`, `GET /api/notes`, `GET /api/notes/<id>`, `DELETE /api/notes/<id>`
5. Add `attach_note_to_graph()` called during `build_knowledge_graph()`

### Phase 2 — Frontend Note Creation
6. Add note node type to `getNodeType`, `getNodeColor`, `getNodeRadius`, CSS
7. Add "Journal Notes" section to sidebar (between Analysis Model and conv-list)
8. Implement note creation form with title + content
9. Implement notes list in sidebar

### Phase 3 — Images & Links
10. Add link input with tag rendering
11. Add image upload with drag-and-drop
12. Add `GET /api/notes/<id>/images/<filename>` route
13. Add image preview in note detail

### Phase 4 — Graph Integration & Polish
14. Notes appear in graph with pink color
15. Notes filtered by topic/intent filters
16. Click note node → note detail view
17. Date-based filtering
18. Note editing (PUT endpoint)

---

## 7. Key Decisions

### Why not store images in JSON?
Images as base64 in JSON would bloat the file and make parsing slow. Using filesystem storage with filenames in JSON keeps the notes file lightweight and images separately cacheable.

### Why reuse existing extraction pipeline?
The existing `EXTRACTION_PROMPT` pattern is proven with llama3.1:8b — 94% success rate. Adapting it for notes (shorter text, no conversation structure) is simpler than building a separate extraction system.

### Why pink for notes?
Existing colors: blue(conversation), green(topic), yellow(concept), red(intent), purple(model), orange(query), light blue(answer). Pink (#ff9bce) is not used by any existing type and provides high contrast against both dark and light themes.

### Should notes be in the overview graph or clusters?
Default: Notes appear in the overview graph (attached to existing concept/topic nodes). They do NOT appear in cluster views (clusters are conversation-only). A future enhancement could cluster notes separately.
