# Hive LLM Queries Dashboard

A knowledge-graph dashboard for exploring LLM conversation history. Uses Ollama for semantic extraction, vector embeddings for intent discovery, and D3.js for interactive graph visualization.

## Features

- **Knowledge Graph** — Force-directed graph of all conversations, organized by topics, concepts, intents, and models
- **Subgraph Drill-Down** — Three-level navigation: Overview → Intent → Sub-Intent → Messages
- **Timeline View** — Chronological flow of query-answer pairs, sized by intent group, with follow-up edges
- **Intent Discovery** — Vector-based clustering of queries using `nomic-embed-text` + LLM labeling
- **Cross-Conversation Search** — Semantic search across all indexed queries via embedding similarity
- **Persistent Knowledge Store** — Vector index + graph index + entity registry, JSON-serialized to `.knowledge_store/`
- **Dark/Light Theme** — Toggle in sidebar, persisted to localStorage
- **Dynamic Model Switching** — Switch between fast (`llama3.2:3b`) and large (`qwen3.6:35b-mlx`) models from the UI
- **JSON Import** — Drag-and-drop or paste conversation exports
- **Docker Support** — Compose file with host Ollama access

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally with models:
  - `llama3.2:3b` (for intent labeling)
  - `nomic-embed-text` (for query embeddings)
  - `qwen3.6:35b-mlx` (optional, for deeper extraction)

### Install & Run

```bash
# Install dependencies
pip install flask requests

# Pre-build extraction cache (optional, speeds first startup)
python3 prebuild_cache.py

# Start the dashboard
python3 app.py
```

Open http://127.0.0.1:5001

### Docker

```bash
docker compose up --build
```

Make sure Ollama is running on the host at port 11434.

## Usage

### Dashboard Sidebar

| Section | Description |
|---------|-------------|
| Stats | Conversation, topic, concept, model counts |
| Search | Filter conversations by title/summary |
| Intent Filters | Filter conversations by extracted intent |
| Model Filters | Filter conversations by LLM model used |
| Analysis Model | Switch between fast/large Ollama models |
| Conversation List | Click any conversation to open its subgraph |

### Graph Views

| View | How to Access | Description |
|------|---------------|-------------|
| Overview | Default tab | Full knowledge graph of all conversations |
| Conversation Subgraph | Click a conversation node | Force graph of that conversation's queries grouped by intent |
| Timeline | Click "Timeline" in tab bar | Chronological flow of query-answer pairs |
| Intent Drill-Down | Click intent node | Focused view of a single intent group |
| Sub-Intent | Click sub-intent node | Further grouped queries within a large intent |

### Filter Bar (Subgraph)

Each subgraph view has a filter bar with:
- **Search input** — Filter messages by text content
- **Topic chips** — Click to show only messages with that topic
- **Count badge** — Number of matching messages

### Timeline Node Detail

Click any message node in timeline view to open a floating overlay with three tabs:
- **Concept** — Query text, answer preview, extracted concepts, tags
- **Intent** — Intent category and conversation title
- **Topic** — Extracted topics and model info

## Architecture

```
app.py                  — Flask server, API routes, graph builder
intent_engine.py        — Embedding → clustering → labeling pipeline
knowledge_store.py      — Persistent vector index + graph store
hive_datatype.py        — HiveGraph data model (Node, Edge, HiveGraph)
prebuild_cache.py       — Pre-compute extraction cache for faster startup
templates/dashboard.html — Full D3.js frontend
```

### Data Flow

```
Chat Export JSON → load_chat_data() → build_knowledge_graph()
                                         ↓
                                  HiveGraph (nodes + edges)
                                         ↓
                                  /api/graph → D3.js force graph
                                         ↓
                                  Click conversation → /api/conversation-subgraph/
                                         ↓
                                  intent_engine.discover_intents_deep()
                                    → embeddings (nomic-embed-text)
                                    → clustering (cosine similarity)
                                    → labeling (llama3.2:3b via /api/chat)
                                         ↓
                                  Subgraph tabs (overview → intent → message)
```

### Storage

| Path | Contents |
|------|----------|
| `.extraction_cache.json` | Cached LLM extractions per conversation |
| `.intent_cache/` | Per-conversation query embeddings (keyed by content hash) |
| `.knowledge_store/` | Cross-conversation vector index + graph + entity registry |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/api/graph` | GET | Full knowledge graph data |
| `/api/conversation/<id>` | GET | Conversation detail with messages |
| `/api/conversation-subgraph/<id>` | GET | Subgraph with intent groups |
| `/api/model` | GET/POST | Get or set analysis model |
| `/api/import-json` | POST | Import chat history JSON |
| `/api/chat-files` | GET | List available chat files |
| `/api/refresh` | GET | Force graph rebuild |
| `/api/cross-intents` | GET | Cross-conversation intent groups |
| `/api/knowledge-store/build` | POST | Build/rebuild knowledge store |
| `/api/knowledge-store/status` | GET | Knowledge store stats |
| `/api/knowledge-store/search?q=...` | GET | Semantic search across queries |
| `/api/knowledge-store/entity/<type>` | GET | List entities by type |
| `/api/knowledge-store/graph/<id>` | GET | Node + neighbors from graph store |
| `/api/knowledge-store/related` | GET | Related entities across conversations |

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint |
| `OLLAMA_MODEL` | `llama3.2:3b` | Default model for extraction |
| `EMBED_URL` | `http://localhost:11434/api/embed` | Ollama embed endpoint |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `LABEL_URL` | `http://localhost:11434/api/chat` | Ollama chat endpoint |
| `LABEL_MODEL` | `llama3.2:3b` | Model for intent labeling (always fast) |
| `CLUSTER_SIM` | `0.55` | Cosine similarity threshold for clustering |
| `MAX_INTENT_QUERIES` | `500` | Max queries per intent pass |
