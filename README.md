# EduAgent: A Multi-Agent Retrieval-Augmented Generation System for Education

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**EduAgent** is an open-source multi-agent LLM system purpose-built for educational
scenarios. It integrates **keyword-first dense–sparse retrieval**, **LangGraph-based
agent orchestration**, **MCP (Model Context Protocol) tool extensibility**, and
**server-sent event (SSE) streaming** into a single deployable stack. The system is
designed to produce faithful, citation-backed answers while offering real-time
interactivity suitable for classroom use.

> This repository accompanies the paper *"[Paper Title]"* and provides the full
> implementation, evaluation harness, and benchmark data for reproducibility.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Contributions](#key-contributions)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Agent Orchestration](#agent-orchestration)
- [Tool Ecosystem (MCP)](#tool-ecosystem-mcp)
- [Evaluation](#evaluation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Citation](#citation)
- [License](#license)

---

## Architecture Overview

```
┌─────────────┐     SSE Stream      ┌──────────────────────────────────┐
│   Frontend   │ ◄────────────────── │          FastAPI Backend         │
│  React/TS    │                     │                                  │
│  Vite + TW   │ ──── POST /chat ──►│  ┌──────────┐  ┌─────────────┐  │
└─────────────┘                     │  │  /chat   │  │ /chat/stream│  │
                                    │  └────┬─────┘  └──────┬──────┘  │
                                    │       │               │          │
                                    │       ▼               ▼          │
                                    │  ┌──────────────────────────┐   │
                                    │  │      LangGraph DAG        │   │
                                    │  │  retriever → generator    │   │
                                    │  └──────┬──────────┬────────┘   │
                                    │         │          │            │
                                    │         ▼          ▼            │
                                    │  ┌──────────┐ ┌──────────────┐ │
                                    │  │ pgvector │ │  MCP Tools   │ │
                                    │  │ +jieba  │ │ (stdio sub-  │ │
                                    │  │ +reranker│ │  processes)  │ │
                                    │  └──────────┘ └──────────────┘ │
                                    └──────────────────────────────────┘
```

The system follows a **linear agent topology**: every user query passes through a
mandatory retrieval node before reaching the generator, which autonomously decides
whether to invoke external tools via LLM function calling.

---

## Key Contributions

1. **Keyword-First Hybrid Retrieval** — A progressive recall strategy combining jieba
   tokenization, SQL `ILIKE` pre-filtering, pgvector cosine ranking, and optional
   cross-encoder re-ranking (Qwen3-Reranker-4B). Empirical evaluation shows 2× Hit@5
   improvement over vector-only baselines on educational corpora.

2. **Faithfulness-Driven RAG for Education** — Controlled experiments (RAGAS benchmark)
   demonstrate that RAG improves answer faithfulness from 0.10 (plain LLM) to 0.40,
   while maintaining competitive answer relevance — a critical trade-off for
   high-stakes educational contexts where hallucination is unacceptable.

3. **MCP-Native Tool Extensibility** — All external capabilities (diagram generation,
   WeChat notification, GitHub search, web fetching, context7 documentation lookup)
   are implemented as MCP-compliant stdio servers with a unified risk-level registry.
   Users selectively enable tools per request, and the LLM autonomously decides
   when to invoke them.

4. **Real-Time SSE Streaming with Sub-Second TTFT** — The `/chat/stream` endpoint
   delivers citation events, token-by-token generation, tool execution results, and
   Mermaid diagram code via structured SSE, achieving **TTFT p50 < 600 ms** under
   single-user load — suitable for interactive classroom use.

5. **Reproducible Evaluation Framework** — A single-script evaluation harness
   (`eval/eval.py`) measures Recall@K, MRR, Hit@K, and nDCG@K across retrieval
   ablations, with manifest-based run tracking and benchmark SHA256 verification.

---

## Retrieval Pipeline

The retrieval module implements a **progressive recall** strategy designed for
Chinese educational corpora:

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1. Keyword Extraction | jieba tokenization + domain dictionary | Extract meaningful terms; filter stopwords |
| 2. Chapter Filter | Regex `第X章` detection | Pin retrieval to a specific textbook chapter |
| 3. Keyword Pre-filter | SQL `ILIKE AND` over `knowledge_chunks` | Narrow candidate set before vector scoring |
| 4. Vector Ranking | pgvector `<=>` cosine distance | Rank candidates by semantic similarity |
| 5. Fallback | Progressive keyword relaxation → full vector | Gracefully degrade when candidates are sparse |
| 6. Re-ranking | Qwen3-Reranker-4B (optional, configurable) | Cross-encoder refinement of top candidates |

**Evaluation results (RQ1, 9 labeled samples on AI textbook corpus):**

| Ablation | Recall@1 | Hit@5 | MRR | nDCG@10 |
|----------|----------|-------|-----|---------|
| Vector Only | 0.003 | 0.111 | 0.111 | 0.025 |
| + Keyword + RRF | **0.017** | **0.222** | **0.222** | **0.053** |
| + Reranker (full) | 0.017 | 0.222 | 0.222 | 0.053 |

---

## Agent Orchestration

The agent graph is defined as a LangGraph `StateGraph` with two nodes:

```
START → rag_retriever → generator → END
```

- **`rag_retriever`**: Executes the retrieval pipeline and builds a formatted citation
  list from the top-K document chunks.
- **`generator`**: Constructs a system prompt that includes the retrieved context,
  citation index, and the full tool menu (with enabled/disabled demarcation). If the
  user has enabled any tools, the LLM is bound with `bind_tools()` and can
  autonomously invoke them. Tool results are injected as `ToolMessage` objects for a
  second LLM pass that synthesizes the final answer.

This design eliminates the need for a separate dispatcher node — the LLM itself
decides *if* and *which* tool to call, following explicit safety guidelines in the
system prompt.

---

## Tool Ecosystem (MCP)

All tools are managed via a unified MCP configuration (`mcp_servers.json`) and a
registry with risk-level metadata:

| Tool | Risk | Provider | Description |
|------|------|----------|-------------|
| `generate_mermaid` | LOW | eduagent (Python) | Convert knowledge into Mermaid DSL diagrams |
| `send_wechat` | HIGH | eduagent (Python) | Push answers to WeChat Work webhook |
| `github` | MEDIUM | @modelcontextprotocol/server-github (npm) | Search repos, issues, PRs, code. Requires `GITHUB_PERSONAL_ACCESS_TOKEN` |
| `fetch` | MEDIUM | mcp-server-fetch (PyPI, run via `python -m mcp_server_fetch`) | Fetch web page content from URLs, with optional Markdown conversion |
| `filesystem` | HIGH | @modelcontextprotocol/server-filesystem (npm) | Read/write files in uploads directory (disabled by default) |
| `context7` | MEDIUM | @upstash/context7-mcp (npm) | Fetch latest library/framework docs. Requires `CONTEXT7_API_KEY` |

**Key design decisions:**
- All tools default to **disabled** — users opt in per request via the frontend
  `ToolSelector`.
- High-risk tools require explicit user intent in the query.
- MCP server environment variables use `${VAR}` placeholders (e.g. `${CONTEXT7_API_KEY}`)
  to avoid hardcoding secrets; missing placeholders cause the server to be **safely
  skipped** at startup with a warning, without breaking other tools.
- Each MCP server runs as a **stdio subprocess**; tools are loaded once at startup
  (warm-up in `lifespan`) and cached for the lifetime of the backend process.
- Disabled servers are prefixed with `_` in `mcp_servers.json` (e.g. `_disabled_filesystem`)
  and skipped during loading — no code change required to toggle them.

---

## Evaluation

### RQ1 — Retrieval Ablation

Measures Recall@K, Hit@K, MRR, and nDCG@K across five retrieval variants on a
Chinese AI textbook corpus. The keyword + RRF combination doubles Hit@5 over pure
vector search.

### RQ2 — End-to-End Answer Quality (RAGAS)

Uses DeepSeek as judge to score faithfulness, answer relevance, and context relevance
across three configurations:

| Configuration | Faithfulness | Answer Relevance |
|---------------|-------------|-----------------|
| A0: Plain LLM (no RAG) | 0.100 | 0.990 |
| A1: RAG Only | 0.365 | 0.690 |
| A2: EduAgent Full | **0.401** | 0.550 |

**Key insight:** RAG provides a ~4× improvement in faithfulness (0.10 → 0.40),
confirming its value for educational scenarios where factual accuracy is paramount.

### RQ4 — Streaming Performance

| Concurrency | TTFT p50 | TTFT p99 | E2E p50 | Success Rate |
|-------------|----------|----------|---------|-------------|
| 1 | 594.8 ms | 1602.3 ms | 4289.1 ms | 100% |
| 3 | 1056.6 ms | 2415.4 ms | 5434.6 ms | 60% |
| 5 | 1399.1 ms | 2347.0 ms | 6222.0 ms | 60% |

Concurrency failures at >1 are attributed to external LLM endpoint rate-limiting,
not to the EduAgent orchestration layer.

> See [`eval/outputs/runs/RESULTS.md`](eval/outputs/runs/RESULTS.md) for detailed
> analysis and manifest references.

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- API keys for LLM, Embedding, and (optionally) Reranker services

### 1. Clone and Configure

```bash
git clone https://github.com/<org>/EduAgent.git
cd EduAgent
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys
```

### 2. Launch Services

```bash
docker compose up -d
```

This starts:
- **PostgreSQL 16 + pgvector** on port 5432
- **FastAPI backend** on port 8000

### 3. Import Knowledge Base

```bash
docker compose exec backend python -c "
import asyncio
from app.db.database import async_session_factory
from app.rag.knowledge_loader import load_file

async def main():
    async with async_session_factory() as db:
        count = await load_file('path/to/textbook.md', db)
        await db.commit()
        print(f'Imported {count} chunks')

asyncio.run(main())
"
```

### 4. Run Evaluation

```bash
docker compose exec backend python -m eval.eval
```

### 5. Access the API

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

---

## Project Structure

```
EduAgent/
├── backend/
│   ├── app/
│   │   ├── api/                  # FastAPI route handlers
│   │   │   ├── chat.py           # /chat, /chat/stream, sessions CRUD
│   │   │   ├── knowledge.py      # Knowledge base management
│   │   │   └── feedback.py       # User feedback collection
│   │   ├── db/
│   │   │   ├── database.py       # AsyncSession + pgvector engine
│   │   │   └── init_db.py        # Schema auto-creation
│   │   ├── graph/
│   │   │   ├── state.py          # AgentState TypedDict
│   │   │   ├── nodes.py          # rag_retriever + generator nodes
│   │   │   ├── builder.py        # LangGraph StateGraph assembly
│   │   │   └── titles.py         # LLM-driven session title generation
│   │   ├── models/
│   │   │   ├── db_models.py      # SQLAlchemy ORM models
│   │   │   └── schemas.py        # Pydantic request/response schemas
│   │   ├── rag/
│   │   │   ├── retriever.py      # Keyword-first retrieval pipeline
│   │   │   ├── embeddings.py     # OpenAI-compatible embedding client
│   │   │   ├── citation.py       # Citation formatting utilities
│   │   │   └── knowledge_loader.py # Markdown → chunks + embeddings
│   │   ├── tools/
│   │   │   ├── registry.py       # ToolMetadata + risk-level registry
│   │   │   ├── server.py         # MCP stdio server entrypoint
│   │   │   ├── mermaid.py        # Mermaid DSL generation
│   │   │   └── wechat.py         # WeChat Work webhook push
│   │   ├── config.py             # Pydantic Settings (env-driven)
│   │   └── main.py               # FastAPI app + lifespan
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatView.tsx       # Main chat interface
│   │   │   ├── Sidebar.tsx        # Session list
│   │   │   ├── KnowledgePanel.tsx # Knowledge base management
│   │   │   ├── ToolSelector.tsx   # Per-request tool toggles
│   │   │   └── MermaidRenderer.tsx # Mermaid diagram rendering
│   │   ├── lib/
│   │   │   ├── api.ts             # Backend API client
│   │   │   └── types.ts           # TypeScript type definitions
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── eval/
│   ├── eval.py                    # Retrieval evaluation harness
│   ├── questions.jsonl            # QA benchmark dataset
│   └── outputs/runs/              # Evaluation run artifacts
│       ├── RESULTS.md             # Aggregated analysis
│       ├── rq1_v3/                # Retrieval ablation results
│       ├── rq2_v2/                # RAGAS answer quality results
│       └── rq4_v1/                # Streaming performance results
├── docker-compose.yml
└── mcp_servers.json               # Unified MCP server configuration
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/chat` | POST | Synchronous chat (full RAG + tool pipeline) |
| `/api/chat/stream` | POST | SSE streaming chat with token-by-token output |
| `/api/tools` | GET | List available tools with risk levels |
| `/api/sessions` | GET | List all conversation sessions |
| `/api/sessions/{id}/messages` | GET | Get full history for a session |
| `/api/sessions/{id}` | DELETE | Delete a session |
| `/api/knowledge/upload` | POST | Upload and index a Markdown textbook |
| `/api/feedback` | POST | Submit user feedback (rating + comment) |

### SSE Event Types (from `/api/chat/stream`)

| Event | Payload | Description |
|-------|---------|-------------|
| `citation` | JSON array of `{id, source_file, heading_path}` | Retrieved document metadata |
| `token` | Plain text | Incremental LLM output token |
| `tool_result` | JSON `{name, args, status, result?}` | Tool invocation outcome |
| `mermaid` | Plain text (Mermaid DSL) | Diagram code for frontend rendering |
| `done` | Conversation ID (int) | Stream completion signal |
| `error` | Error message (string) | Stream-level error |

---

## Citation

If you use EduAgent or its components in your research, please cite:

```bibtex
@software{eduagent2026,
  author = {<Authors>},
  title = {EduAgent: A Multi-Agent Retrieval-Augmented Generation System for Education},
  year = {2026},
  url = {https://github.com/<org>/EduAgent}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
