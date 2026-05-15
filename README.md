# EduAgent — 教育领域 RAG + MCP 多 Agent 系统

> 基于 LangGraph 编排 + 混合检索 RAG + MCP 工具生态的教育智能体。
> 后端 FastAPI(异步 / SSE 流式),向量库 PostgreSQL+pgvector,LLM 默认 ModelScope Qwen3-30B-A3B,Embedding 默认 SiliconFlow bge-m3。

---

## 目录

- [一、核心特性](#一核心特性)
- [二、整体架构](#二整体架构)
- [三、技术选型](#三技术选型)
- [四、项目结构](#四项目结构)
- [五、后端文件详解](#五后端文件详解)
- [六、关键功能处理流程](#六关键功能处理流程)
- [七、数据模型](#七数据模型)
- [八、API 参考](#八api-参考)
- [九、MCP 工具生态](#九mcp-工具生态)
- [十、配置与环境变量](#十配置与环境变量)
- [十一、快速启动](#十一快速启动)
- [十二、前端(简略)](#十二前端简略)
- [十三、运维与排障](#十三运维与排障)
- [十四、路线图](#十四路线图)
- [十五、License](#十五license)

---

## 一、核心特性

- **教学场景 RAG**:Markdown 教材按标题切分 → bge-m3 向量化 → pgvector HNSW 索引 → **混合检索(向量召回 + 关键词召回 + RRF 融合 + 可选 reranker)**,回答自动附 `[1][2]` 来源引用。
- **LangGraph 编排**:`rag_retriever → generator → dispatcher` 三节点,条件边按需路由,统一同步 / 流式两套接口。
- **真·SSE token 流式**:从 RAG 命中到工具调用,逐事件推送(`citation` / `token` / `mermaid` / `tool_result` / `done` / `error`),首字延迟即 LLM 的 TTFT。
- **MCP 工具生态**:
  - 内置 `generate_mermaid`(LLM 生成图表 DSL,带语法兜底自动修复)、`send_wechat`(企业微信 Webhook 推送)。
  - 外部任意 MCP server 通过 `mcp_servers.json` 一行接入(`@modelcontextprotocol/server-github`、`fetch`、`filesystem` 等)。
  - 配置文件 `env` / `args` 支持 `${VAR}` 占位符,密钥从环境变量解析,**缺失时安全跳过该 server 并告警**。
  - LLM `bind_tools` 自主决定调用哪些外部工具。
- **每次工具调用单独 SSE 推送**:前端按 `{name, args, status, result|error}` 渲染折叠卡片,展示调用名、参数、返回摘要。
- **可视化 / 推送 / 外部工具三个开关**:聊天框上端并排,默认外部工具开启、其他关闭。
- **持久化工件**:`mermaid_code` 与 `tool_invocations` 落盘到 `conversations` 表,会话切走再切回仍能完整重放。
- **会话标题自动生成**:首轮对话完成后 fire-and-forget 调用 LLM,12 字内主题概括。
- **MCP 工具预热**:lifespan 启动时 fork stdio 子进程并拉取工具列表,首请求省 1-3 秒。
- **Embedding 单 query TTL+LRU 缓存**:重复查询零外部 API 开销。
- **启发式 RAG 路由**:闲聊 / 元问题 / 短查询自动跳过 RAG,零额外 LLM 调用。
- **章节定向检索**:查询中带"第三章"等关键词时,WHERE 子句过滤源文件名,提高召回相关性。

---

## 二、整体架构

```
                       ┌───────────────────────────┐
                       │  前端 React + Vite + TS   │
                       │  (SSE 解析 / Mermaid 渲染)│
                       └─────────────┬─────────────┘
                                     │ /api/*  (Vite 代理或同源部署)
                                     ▼
┌────────────────────────────────────────────────────────────────┐
│                       FastAPI 后端 (Uvicorn)                   │
│                                                                │
│   ┌──────────────┐   ┌──────────────────────────────────────┐  │
│   │ API 路由层   │   │  LangGraph 状态图                    │  │
│   │ chat / kb /  │──►│  rag_retriever → generator → ?       │  │
│   │ feedback     │   │            ?: dispatcher (LLM 工具)  │  │
│   └──────┬───────┘   └──────┬──────────────┬────────────────┘  │
│          │                  │              │                   │
│   ┌──────▼───────┐  ┌───────▼──────┐ ┌─────▼─────────────┐     │
│   │  RAG 模块    │  │  Embedding   │ │   MCP 客户端       │    │
│   │  混合检索 +  │  │  bge-m3 +    │ │  (langchain-mcp-   │    │
│   │  RRF + rerank│  │  TTL 缓存    │ │   adapters)        │    │
│   └──────┬───────┘  └──────────────┘ └─────┬─────────────┘     │
│          │                                 │                   │
└──────────┼─────────────────────────────────┼───────────────────┘
           ▼                                 ▼
   ┌───────────────┐                  ┌─────────────────────┐
   │ PostgreSQL 16 │                  │ 内置 MCP Server     │
   │  + pgvector   │                  │ (stdio 子进程)      │
   │   HNSW 索引   │                  │  - generate_mermaid │
   │  3 张表       │                  │  - send_wechat      │
   └───────────────┘                  └─────────────────────┘
                                                ▲
                                                │ stdio
                                      ┌─────────┴─────────────┐
                                      │ 外部 MCP servers      │
                                      │  - github             │
                                      │  - fetch              │
                                      │  - filesystem ...     │
                                      └───────────────────────┘
```

请求流向(以 `/api/chat/stream` 为例,**统一工具注册表**架构):

```
client ──POST {query, session_id, enabled_tools}─► /chat/stream
                 │
                 ├─ Stage 1  并行:RAG 检索(可选) + 写 user 消息到 DB
                 ├─ Stage 2  推 SSE event: citation
                 ├─ Stage 3  prompt 注入"本轮启用的工具清单"(含 LOW/MEDIUM/HIGH 风险标签)
                 ├─ Stage 4  llm.bind_tools(enabled).astream() 第一轮流式
                 │            ├─ 逐 token 推 SSE event: token
                 │            └─ AIMessageChunk 累加抓 tool_calls
                 ├─ Stage 5  顺序执行 tool_calls(每个独立推 tool_result event,
                 │             status ∈ success / error / rejected)
                 ├─ Stage 6  若有 tool_call → 带 ToolMessage 回灌再 astream 第二轮 → 继续推 token
                 ├─ Stage 7  推 SSE event: mermaid(若 generate_mermaid 命中)
                 └─ Stage 8  落库 → SSE event: done(含 conversation_id)
```

**注:** 不再有 dispatcher 节点 / `need_visualization` / `need_dispatch` 等显式开关,
所有工具(内置 + 外部)统一由 LLM 基于系统提示自主决策。

---

## 三、技术选型

### 后端

| 类别 | 选型 | 理由 |
|---|---|---|
| Web 框架 | **FastAPI 0.115** | 原生 async / 自动 OpenAPI / Pydantic 校验 / 性能优秀 |
| ASGI 服务器 | Uvicorn 0.30 (`--reload`) | 开发热重载,生产可换 gunicorn+uvicorn worker |
| 校验/配置 | Pydantic 2.9 + pydantic-settings 2.5 | 类型安全的 schema 和 `.env` 加载 |
| ORM | SQLAlchemy 2.0 async + asyncpg 0.29 | 现代异步 ORM,与 FastAPI 异步链路一致 |
| 向量库 | **PostgreSQL 16 + pgvector 0.3.5** | 单库即可,HNSW 索引 O(log N) 检索,运维心智成本低于专用向量库 |
| LLM 客户端 | langchain-openai 0.2 | 兼容任意 OpenAI 协议端点(ModelScope / SiliconFlow / DeepSeek / OpenAI 原厂) |
| 编排框架 | **LangGraph 0.2.39** | 显式状态机比隐式 agent 链路可调试,条件边天然支持 RAG 路由 |
| MCP 客户端 | **langchain-mcp-adapters 0.0.3** | 一行 `MultiServerMCPClient` 接入多 server,与 LangChain Tool 接口打通 |
| MCP Server | **FastMCP** (`mcp` 官方 SDK) | stdio 传输,Python 进程间隔离,工具热插拔 |
| SSE | sse-starlette 2.1 | 真·token 流式,自动处理 CRLF / multiline data |
| HTTP 客户端 | httpx 0.27 | 原生 async,reranker / wechat 推送复用 |
| 文件上传 | python-multipart 0.0.12 | FastAPI 默认依赖 |

### 前端

| 类别 | 选型 |
|---|---|
| UI 框架 | React 18 + TypeScript 5.6 |
| 构建 | Vite 5.4(开发代理 `/api → :8000`) |
| 样式 | Tailwind CSS 3.4 + 自定义 cream/terracotta/ink 调色板 |
| Markdown | react-markdown + remark-gfm |
| Mermaid 渲染 | mermaid 11.4(SSR 关闭,按需 init) |
| 图标 | lucide-react |

### 模型默认配置

| 角色 | 模型 | 端点 | 选型理由 |
|---|---|---|---|
| LLM | Qwen/Qwen3-30B-A3B-Instruct-2507 | ModelScope | 30B MoE,推理快、免费、教学问答稳定 |
| Embedding | BAAI/bge-m3 (1024 维) | SiliconFlow | 中英文双语 SOTA,与 db_models `Vector(1024)` 强一致 |
| Reranker(可选) | BAAI/bge-reranker-v2-m3 | SiliconFlow | 候选集二次排序,留空则不启用 |

---

## 四、项目结构

```
EduAgent/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI 入口、CORS、lifespan 预热
│   │   ├── config.py           # Pydantic Settings 单例
│   │   ├── api/
│   │   │   ├── chat.py         # 对话:同步/流式/会话管理/GET /tools
│   │   │   ├── knowledge.py    # 知识库上传/列表/删除
│   │   │   └── feedback.py     # 反馈提交
│   │   ├── tools/
│   │   │   ├── registry.py     # 统一工具注册表(metadata + 启用集合过滤)
│   │   │   └── __init__.py
│   │   ├── graph/
│   │   │   ├── builder.py      # StateGraph 编译(线性:rag_retriever → generator → END)
│   │   │   ├── state.py        # AgentState TypedDict
│   │   │   ├── nodes.py        # rag_retriever / generator(内部 bind_tools 自主决策)
│   │   │   ├── edges.py        # 已废弃占位(线性图不需要条件边)
│   │   │   └── titles.py       # 会话标题异步生成
│   │   ├── rag/
│   │   │   ├── retriever.py    # 混合检索:向量+关键词+RRF+rerank
│   │   │   ├── embeddings.py   # bge-m3 + TTL+LRU 缓存
│   │   │   ├── knowledge_loader.py # Markdown → 标题切分 → embedding → 入库
│   │   │   └── citation.py     # 引用格式化
│   │   ├── models/
│   │   │   ├── db_models.py    # SQLAlchemy ORM: KnowledgeChunk/Conversation/Session/Feedback
│   │   │   └── schemas.py      # Pydantic 请求/响应
│   │   └── db/
│   │       ├── database.py     # async engine / session factory
│   │       └── init_db.py      # 启用 pgvector / 创建表 / 幂等 ALTER / 建索引
│   ├── Dockerfile              # Python 3.11 + Node.js 22 (npx 启动 npm-based MCP)
│   └── requirements.txt
├── mcp_server/                 # 内置 MCP server (stdio 子进程)
│   ├── server.py               # FastMCP 入口
│   └── tools/
│       ├── mermaid.py          # generate_mermaid:LLM 生成 DSL + 危险字符自动修复
│       └── wechat.py           # send_wechat:企业微信 Webhook
├── mcp_servers.json            # 外部 MCP server 配置 (${VAR} 占位符)
├── frontend/                   # React SPA
│   └── src/
│       ├── App.tsx
│       ├── components/{ChatView,Sidebar,MermaidRenderer,KnowledgePanel}.tsx
│       └── lib/{api,types}.ts
├── docker-compose.yml          # postgres + backend (volume 挂载源码/MCP)
└── README.md                   (本文档)
```

---

## 五、后端文件详解

### 5.1 `app/main.py`(入口)

- 创建 `FastAPI` 实例,挂载 CORS 中间件(开发期 `*`,关闭 credentials)、注册三个路由(`chat` / `knowledge` / `feedback`)、`/` 与 `/health` 健康检查。
- `lifespan` 启动阶段做三件事:
  1. `init_database()`:启用 `vector` 扩展、`Base.metadata.create_all`、幂等 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`(用于灰度发布新字段)、建 HNSW 与复合索引。
  2. `_get_embeddings_model()` 同步构造客户端,避免首请求阻塞 event loop。
  3. `_get_mcp_tools()` fork stdio 子进程并 `ListToolsRequest`,把工具表加载到全局缓存,首请求省 1-3s。
- 关闭阶段记录日志即可,无显式连接清理(SQLAlchemy / httpx 客户端自动 GC)。

### 5.2 `app/config.py`(配置)

- `pydantic_settings.BaseSettings` 从 `.env` 加载,`extra="ignore"` 容忍多余键。
- `@lru_cache` 的 `get_settings()` 让全应用共享一个 `Settings` 实例。
- 关键字段:`LLM_*` / `EMBEDDING_*` / `RERANKER_*`(可选) / `DATABASE_URL` / `WECHAT_WEBHOOK_URL` / `MCP_SERVER_COMMAND` / `MCP_SERVER_ARGS` / `RAG_OVERFETCH_MULTIPLIER`(默认 4) / `RAG_RRF_K`(默认 60) / `UPLOAD_DIR`。

### 5.3 `app/db/database.py`(连接管理)

- `create_async_engine(DATABASE_URL, pool_size=10, max_overflow=20)`。
- `async_sessionmaker(expire_on_commit=False)`,**这点很关键**:`/chat/stream` 在 `done` 之后会二次 commit 写 `mermaid_code` / `tool_invocations_json`,需要 ORM 对象在 commit 后仍可用。
- `get_db()` 是 FastAPI 依赖,**不可用于 SSE 端点**(FastAPI 会在 `return EventSourceResponse(...)` 时立刻关 session,但 generator 才刚开始迭代)。流式端点改在 generator 内部用 `async with async_session_factory()`。

### 5.4 `app/db/init_db.py`(启动初始化)

- 启用 pgvector 扩展。
- `Base.metadata.create_all`:只在表不存在时建表。
- 幂等列补齐(本项目未接 Alembic,以 `ADD COLUMN IF NOT EXISTS` 兜底,适合早期单人项目):
  ```sql
  ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS mermaid_code TEXT,
    ADD COLUMN IF NOT EXISTS tool_invocations_json TEXT;
  ```
- 建 HNSW 向量索引(cosine ops)与 `(session_id, created_at DESC)` 复合索引(覆盖 `_load_history` 与 `list_sessions`)。

### 5.5 `app/models/db_models.py`(ORM 模型)

| 表 | 主要字段 | 备注 |
|---|---|---|
| `knowledge_chunks` | `id` / `source_file` / `heading_path` / `chunk_index` / `original_text` / `embedding Vector(1024)` / `created_at` | HNSW 索引在 `embedding` 上 |
| `sessions` | `session_id PK` / `title` / `created_at` / `updated_at` | 标题由 `titles.py` 异步写入 |
| `conversations` | `id` / `session_id` / `role` / `content` / `citations_json` / `mermaid_code` / `tool_invocations_json` / `created_at` | 后两列承载工件持久化 |
| `feedbacks` | `id` / `conversation_id` / `rating(±1)` / `comment` / `prompt` / `response` / `created_at` | 同时落盘 prompt 与 response 便于微调采样 |

### 5.6 `app/models/schemas.py`(Pydantic Schema)

- `ChatRequest`:`query` / `session_id` / `enabled_tools: dict[str, bool]`(本轮要启用的工具,缺失键回退到 registry 默认值)。
- `ChatResponse`:`conversation_id` / `content` / `citations` / `mermaid_code` / `tool_results`(后者为 `{name: invocation}` 字典)。
- `ToolConfig` / `ToolsListResponse`:`GET /api/tools` 响应模型,前端 `ToolSelector` 用它渲染开关。
- `SessionListItem`:`title` / `last_message`(80 字预览) / `last_role` / `message_count` / `updated_at`。
- `MessageItem`:历史消息,含 `mermaid_code` 与 `tool_invocations`(用于切回会话时重放工件)。
- `FeedbackRequest`:`conversation_id` / `rating(1|-1)` / `comment`。

### 5.7 `app/tools/registry.py`(**统一工具注册表**)

整个项目唯一的工具元数据中心,删除了内置 / 外部的二分法。

- `ToolCategory`:`internal` / `external`(只为前端分组渲染,后端调度不区分)。
- `RiskLevel`:`LOW`(只读) / `MEDIUM`(只读 + 外部网络) / `HIGH`(改写远端/本地状态)。
- `ToolMetadata`:`@dataclass(frozen=True)`,字段 `id / display_name / description / category / risk_level / default_enabled / mcp_server`。
- `TOOLS_REGISTRY`:静态字典,当前收录 `generate_mermaid` / `send_wechat` / `github` / `fetch` / `filesystem` 五个工具,前两个 `default_enabled=True`,其余默认禁用。
- `get_available_tools(loaded_tools, enabled_tools)`:把"MCP client 实际加载到的工具"与"本次请求启用集合"做交集,返回可 `bind_tools()` 的列表。注册但 MCP server 没起来的工具(如未填 `GITHUB_PAT`)静默跳过。
- `get_default_enabled_map()`:生成默认启用字典,供前端首次拉取使用。

### 5.8 `app/graph/state.py`

`AgentState` 是 `TypedDict(total=False)`,LangGraph 节点共享:`messages` / `query` / `retrieved_docs` / `citations` / `generated_content` / `tool_invocations` / `mermaid_code` / `enabled_tools` / `session_id`。**不再有 `need_visualization` / `need_dispatch` 字段**。

### 5.9 `app/graph/nodes.py`(两大节点)

- `rag_retriever_node`:调 `retriever.retrieve(query, top_k=5)`,失败时返回空 docs 继续往下(不中断)。
- `generator_node`:
  - 用 `build_system_prompt(retrieved_docs, enabled_tools)` 拼系统提示,提示词会把本轮启用的工具按 `display_name / 风险等级 / 描述` 全部列出来,并明确告诉 LLM "高风险工具必须有用户明确意图才能调用"。
  - 通过 `get_available_tools()` 过滤出本轮可用工具,`llm.bind_tools(available)` 一次性 ainvoke。
  - 若 LLM 输出 `tool_calls`,顺序执行每一个,把 `ToolMessage` 回灌后再跑一次 `_get_llm().ainvoke(...)` 拿到最终正文。
  - 状态产出:`generated_content` / `tool_invocations` / `mermaid_code`(若 `generate_mermaid` 命中则单独提取)。
- 同模块还有:
  - `_get_llm()`:延迟初始化的 `ChatOpenAI` 单例。
  - `_get_mcp_tools()`:聚合内置 `eduagent` MCP + 外部 MCP 的工具,缓存为 `{name: tool}`。
  - `_load_external_mcp_servers()`:从 `mcp_servers.json` 读外部 server 配置,过滤 `_` 前缀,**展开 `${VAR}` 占位符**;变量缺失时跳过该 server 并打印警告。
  - `_extract_text(result)`:把 MCP 返回的 `list[TextContent]` / dict / str 统一拍平为纯字符串。
  - `_content_to_text(content)`:把 LangChain AIMessage `content`(可能是 list[dict])拍平成 str。

### 5.10 `app/graph/edges.py`(已废弃)

- 线性图(`rag_retriever → generator → END`)不需要任何条件路由,本文件保留为空占位,避免历史 import 路径报错。

### 5.11 `app/graph/builder.py`

- 装配 `StateGraph(AgentState)`,只注册 `rag_retriever` 与 `generator` 两个节点,边: `rag_retriever → generator → END`,**没有条件边**。
- 模块加载时即 `app_graph = build_graph()`,API 层 `await app_graph.ainvoke(state)` 启动。

### 5.12 `app/graph/titles.py`(会话标题)

- `_TITLE_SYSTEM_PROMPT`:让 LLM 用 ≤12 字概括,直接输出标题。
- `_clean_title`:去引号 / 标点 / 空白,截到 24 字。
- `schedule_title_generation(session_id, first_query)`:**fire-and-forget**,`asyncio.create_task`。失败仅记日志,不阻塞主请求。在两个 chat 端点中,**仅当 `is_first_turn=True`** 时调度。

### 5.13 `app/rag/embeddings.py`

- `_get_embeddings_model()` 延迟初始化 `OpenAIEmbeddings`。
- `embed_text(text)`:**自带 OrderedDict TTL+LRU 缓存**(`maxsize=512`、`ttl=600s`)。同样 query 命中 0 外部 API。
- `embed_batch(texts, batch_size=32)`:按批调 `aembed_documents`,避免单次过大被 API 拒。

### 5.14 `app/rag/knowledge_loader.py`

- `_split_by_headings`:扫描 `#` / `##`,按一/二级标题切语义段,`heading_path` 写成 `第一章 > 1.2 牛顿定律`。
- `_secondary_split`:若段落 > 512 字符,按 `\n\n` 二次切分,带 64 字符 overlap;若仍超长则按字符硬切。
- `load_file(file_path, db)`:read → split → `embed_batch` → `db.add_all`。返回 chunk 数。

### 5.15 `app/rag/retriever.py`(混合检索核心)

- **章节定向**:`_detect_chapter_filter` 匹配 `第X章`,把 LIKE 条件作为 SQL WHERE 加进去。
- **向量层(over-fetch)**:`top_k * RAG_OVERFETCH_MULTIPLIER`(默认 4)个候选,`ORDER BY embedding <=> :q_emb`。
- **关键词层**:`_extract_keywords` 抽 ≥2 字符 CJK / ASCII 单词,去停用片段;`_keyword_score` 按命中次数累加,候选集内重排。
- **RRF 融合**:`_rrf_fuse` 给两条 rank 列表打 `1/(k+rank)` 分,以文档 id 为去重键,默认 `k=60`。
- **可选 Reranker**:`RERANKER_API_KEY` 配置后调用 SiliconFlow `bge-reranker-v2-m3`,失败回退 RRF 顺序。
- 返回 `[{id, text, source_file, heading_path, chunk_index, score}]`。

### 5.16 `app/rag/citation.py`

- `format_context`:把 docs 拼成 `[1] {text}\n\n[2] {text}` 喂给 LLM。
- `format_citations`:拼 `[1] {source_file} > {heading_path}` 文本块。
- `build_citation_list`:产出 API 响应用的结构化字典列表。

### 5.17 `app/api/chat.py`(核心)

划分为五层:

1. **启发式判定**
   - `_needs_rag`:闲聊 / 元问题 / 极短查询 → 跳过 RAG,零额外 LLM 调用。
   - 不再有"可视化关键词自动开关"——是否调 `generate_mermaid` 完全由 LLM 看 prompt 决定。
2. **持久化辅助**
   - `_save_conversation`:写一条 user / assistant 消息,支持 `citations_json` / `mermaid_code` / `tool_invocations_json`。
   - `_load_history(limit=20)`:倒序拉再反转得正序 history。
3. **工具注册表 `GET /tools`**:返回 `TOOLS_REGISTRY` 全部元数据 + 默认启用状态,前端 `ToolSelector` 用它渲染开关。
4. **同步对话 `POST /chat`**:走完整 LangGraph(`rag_retriever → generator → END`),把 `mermaid_code` / `tool_invocations` 一次性落库。
5. **流式对话 `POST /chat/stream`**(`EventSourceResponse`):
   - 八阶段流水线(见架构图)。
   - 用独立 `async with async_session_factory()` 维持 session,显式 commit 两次:user 消息进库一次、done 前 assistant 消息 + 工件一起 commit。
   - 第一轮 `llm.bind_tools(available).astream()`:边流式推 token,边把 `AIMessageChunk` 累加成完整 `AIMessage` 以抓取 `tool_calls`。
   - 顺序执行每个 tool_call,推 `tool_result` 事件,统一 schema `{name, args, status, result|error}`。
   - 若有 tool_call → 第二轮 `_get_llm().astream(followup_msgs)`(带 `ToolMessage` 回灌)继续推 token。
6. **会话管理**
   - `GET /sessions`:单条 SQL 用 `DISTINCT ON` 取每 session 最后一条 + 窗口函数算总数 + `LEFT JOIN sessions` 拼标题,避免 N+1。
   - `GET /sessions/{id}/messages`:正序拉所有消息,反序列化 `citations_json` / `tool_invocations_json`,把 `mermaid_code` 一并塞回响应。
   - `DELETE /sessions/{id}`:先查 conversation_id,然后串联删 `feedbacks` / `conversations` / `sessions`。

### 5.18 `app/api/knowledge.py`

- `POST /knowledge/upload`:仅接受 `.md`,先 `DELETE WHERE source_file=?` 再调 `load_file` 实现"覆盖式更新"。
- `GET /knowledge/list`:按文件名 group by 数 chunk。
- `DELETE /knowledge/{doc_name}`:数 chunk → delete → 移除磁盘文件。

### 5.19 `app/api/feedback.py`

- 校验 `rating ∈ {1, -1}`。
- 校验 `conversation_id` 是 assistant 消息;再回溯找同 session、`created_at <=` 它的最近一条 user 消息当作 prompt。
- 把 `prompt+response` 一起入 `feedbacks` 表(冗余存,方便日后导出做 SFT 数据)。

### 5.20 `mcp_server/server.py`

- `FastMCP("eduagent-mcp-tools")`,挨个调 `register_mermaid` / `register_wechat` 注册工具,`stdio` 传输运行。

### 5.21 `mcp_server/tools/mermaid.py`

- `PROMPT_TEMPLATE` **极度详尽**:硬性语法规则(节点 ID 必须英文、危险字符自动包裹双引号)、丰富度要求(10-20 节点,3 层 mindmap),配 flowchart 与 mindmap 完整示例。
- `_strip_fences`:剥 ```` ```mermaid ```` 围栏(防 LLM 偷偷加 markdown)。
- `_sanitize_labels`:正则扫 `[...]` / `((...))` / `{{...}}` / `{...}`,标签内含 `( ) < > ; , & / \ :` 时**自动用双引号包**;LLM 经常忘的语法陷阱被强制兜底。
- 失败时返回 `flowchart TD\n    A[生成失败: ...]` 兜底图,不抛异常。

### 5.22 `mcp_server/tools/wechat.py`

- 从 `WECHAT_WEBHOOK_URL` 环境变量读 webhook。
- 支持 `text` / `markdown` 两种消息类型,httpx POST,errcode=0 视为成功。

---

## 六、关键功能处理流程

### 6.1 流式对话 `POST /api/chat/stream`

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as chat_stream
    participant DB as Postgres
    participant RAG as retriever
    participant LLM as ChatOpenAI
    participant MCP as MCP tools

    C->>API: POST {query, session_id, enabled_tools}
    par 并行
        API->>RAG: retrieve(query, top_k=5)  [可选]
    and
        API->>DB: insert user message + commit
    end
    RAG-->>API: top_k docs
    API->>C: event: citation
    Note over API: build_system_prompt(retrieved_docs, enabled_tools)
    API->>LLM: bind_tools(available).astream() — 第一轮
    loop 每个 token
        LLM-->>API: chunk
        API->>C: event: token
    end
    Note over API: 累加 AIMessageChunk 抓 tool_calls
    opt 有 tool_calls
        loop 每个 tool_call
            API->>MCP: tool.ainvoke(args)
            API->>C: event: tool_result {name, args, status, result|error}
        end
        API->>LLM: astream(followup_msgs + ToolMessages) — 第二轮
        loop 每个 token
            LLM-->>API: chunk
            API->>C: event: token
        end
    end
    opt generate_mermaid 命中
        API->>C: event: mermaid
    end
    API->>DB: insert assistant message + commit
    API->>C: event: done (conversation_id)
    API->>C: schedule_title_generation (fire-and-forget)
```

**关键设计点**:

- **统一工具决策**:所有工具(内置 + 外部)都通过 `bind_tools(get_available_tools(loaded, enabled))` 暴露给 LLM,由 LLM 看 prompt 自主决策。没有 dispatcher 直通路径,也没有"need_visualization 关键词自动开"启发式。
- **流式 + tool_calls**:第一轮 astream 同时把 token 推给前端 + 累加 chunk 抓 tool_calls。第二轮在 tool 执行结果回灌后继续流式推 token。前端收到的 token 视觉上是连续的。
- **token 真实流式**:`llm.astream()` 而非 `ainvoke()`,首字延迟 = TTFT 而非完整 latency。
- **风险隔离**:高风险工具(`send_wechat` / `filesystem`)在 prompt 里被显式标注"高风险,仅当用户明确要求时调用",同时前端 `ToolSelector` 用红色徽标提醒用户取消勾选。

### 6.2 混合检索 `retriever.retrieve()`

```
query
  ├─ embed_text → 1024-dim 向量          (TTL 缓存命中则秒返)
  └─ _extract_keywords → ["牛顿", "第二定律", ...]
        │
        ▼
  pgvector SQL: ORDER BY embedding <=> :q LIMIT top_k * 4
        │
        ▼ (over-fetch 候选, e.g. 20)
  ┌─────────────────────────────────────────────┐
  │ 向量序 vector_ranked  (按余弦距离)          │
  │ 关键词序 keyword_ranked (按命中次数, 0分过滤)│
  └─────────────────────────────────────────────┘
        │
        ▼  _rrf_fuse(k=60)   score_i = Σ 1 / (k + rank)
  fused 列表 (按 RRF 分降序)
        │
        ▼  截断到 top_k * 2
  pre_rerank
        │
        ▼  reranker?
  ┌──── 有 ──── SiliconFlow bge-reranker-v2-m3
  └──── 无 ──── 原样
        │
        ▼  取前 top_k
  最终结果(带 score)
```

### 6.3 知识库导入 `POST /api/knowledge/upload`

```
multipart .md 文件
  ↓
保存到 uploads/{filename}
  ↓
DELETE conversations WHERE source_file = filename   (覆盖式更新)
  ↓
load_file:
  read file → _split_by_headings (H1/H2)
            → _secondary_split (>512 字符再切,64 字符 overlap)
            → embed_batch (batch_size=32)
            → db.add_all + flush
  ↓
返回 {message, source_file, chunks_count}
```

### 6.4 LLM 自主工具调用(统一注册表)

```
build_system_prompt(retrieved_docs, enabled_tools)
  └─ 拼入"本轮可用工具清单"(含 LOW/MEDIUM/HIGH 风险标签 + 描述)
       + 调用原则(高风险须明确意图 / 不要复述结果 / 不要在正文嵌 mermaid 代码块)
  ↓
available = get_available_tools(loaded_mcp_tools, enabled_tools)
  └─ registry × enabled × loaded 三方交集
  ↓
llm.bind_tools(available).astream(messages) — 第一轮
  ├─ 边推 token 到 SSE,边累加 AIMessageChunk 抓 response.tool_calls
  ↓
若 tool_calls 非空:
    for each call ∈ tool_calls:
        if name not in available: yield {status="rejected", error="tool not enabled"}
        else:
            try:    result = await tool.ainvoke(args)
                    yield {status="success", result=text[:2000]}
            except: yield {status="error", error=str(e)}
        每个 yield 都对应一条 SSE tool_result 事件
    _get_llm().astream(followup_msgs + ToolMessages) — 第二轮
      └─ 把工具结果回灌给 LLM,继续推 token
```

### 6.5 工件持久化与会话回放

```
首次访问会话                          切走再切回
─────────────────                    ───────────────────
SSE stream                            GET /sessions/{id}/messages
  ↓                                     ↓
ChatView 增量累计:                    后端反序列化 row.citations_json /
  content / citations /                tool_invocations_json
  mermaid_code /                       传回 MessageItem 列表
  tool_invocations[]                     ↓
  ↓                                    ChatView 加载历史:
后端 Stage 8 UPDATE                    map → UIMessage (含 mermaid_code /
  conversations.mermaid_code           tool_invocations)
  conversations.tool_invocations_json    ↓
                                       ToolInvocationsList +
                                       MermaidRenderer 自动重渲染
```

### 6.6 会话标题异步生成

- 触发点:`/chat` 与 `/chat/stream` 在 `_load_history` 返回空 list 时设 `is_first_turn=True`,assistant 消息存盘后调 `schedule_title_generation(session_id, first_query)`。
- 调用链:`asyncio.create_task → ensure_session_title → _generate_title_text → _clean_title → INSERT/UPDATE sessions`。
- 失败完全静默,日志一行 warning。前端通过 `GET /sessions` 拉到 `title` 字段后会自动覆盖侧栏。

---

## 七、数据模型

```
knowledge_chunks                conversations
─────────────────               ───────────────────────────────────
id           PK                  id            PK
source_file  IDX  ──┐            session_id    IDX ─┐
heading_path        │            role               │
chunk_index         │            content            │
original_text       │            citations_json     │
embedding   Vector(1024) HNSW    mermaid_code        │
created_at          │            tool_invocations_json│
                    │            created_at         │
sessions            │                                │
──────────────────  │                                │
session_id   PK ────┘  (松耦合,不建外键)             │
title       (LLM 自动生成)                            │
created_at                                            │
updated_at                                            │
                                                      ▼
feedbacks                              (松耦合,不建外键)
──────────────────────
id              PK
conversation_id IDX
rating          1 | -1
comment         nullable
prompt          冗余存,便于 SFT 采样
response        冗余存
created_at
```

设计说明:
- **不建外键**:写入路径全在应用层控,降低 DDL 演进负担(早期项目策略,接 Alembic 后再加)。
- **citations_json / tool_invocations_json 用 TEXT 而非 JSONB**:简化迁移,反正后端解析后扔给 Pydantic 校验,查询不依赖内部字段。
- **conversations 复合索引** `(session_id, created_at DESC)`:覆盖 `_load_history` 与 `list_sessions`,O(log N) 拉取。

---

## 八、API 参考

### 对话

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET`  | `/api/tools` | 返回全部工具元数据(id / display_name / description / category / risk_level / default_enabled / mcp_server),供前端渲染开关 |
| `POST` | `/api/chat` | 同步对话,返回 `conversation_id` 用于反馈 |
| `POST` | `/api/chat/stream` | SSE 流式,事件:`citation` / `token` / `mermaid` / `tool_result` / `done` / `error` |

请求体:
```json
{
  "query": "请总结牛顿第二定律",
  "session_id": "s-xxx",
  "enabled_tools": {
    "generate_mermaid": true,
    "send_wechat": false,
    "github": false,
    "fetch": false,
    "filesystem": false
  }
}
```

`enabled_tools` 中缺失的键回退到 `TOOLS_REGISTRY[id].default_enabled`,所以前端可以只传用户调整过的工具。

SSE `tool_result` 事件 data 载荷(每次工具调用一条):
```json
{
  "name": "generate_mermaid",
  "args": {"description": "...", "diagram_type": "flowchart"},
  "status": "success",
  "result": "flowchart TD ..."
}
```

`status` 可能为 `success` / `error` / `rejected`(LLM 调用了一个本轮未启用的工具)。

### 会话管理

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/sessions` | 所有会话,按 `updated_at` 倒序 |
| `GET` | `/api/sessions/{id}/messages` | 单会话全部历史(正序),含 mermaid_code / tool_invocations |
| `DELETE` | `/api/sessions/{id}` | 删会话 + 关联 feedbacks |

### 知识库

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/knowledge/upload` | multipart,`.md` 文件;同名覆盖 |
| `GET` | `/api/knowledge/list` | 文档列表 + chunk 数 |
| `DELETE` | `/api/knowledge/{doc_name}` | 删除指定文档全部 chunk |

### 反馈

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/feedback` | `conversation_id` + `rating(±1)` + 可选 `comment` |

### 健康检查

- `GET /` → `{"status": "ok", "service": "EduAgent"}`
- `GET /health` → `{"status": "healthy"}`
- 完整 OpenAPI 在 `/docs`(Swagger UI) / `/redoc`。

---

## 九、MCP 工具生态

### 9.1 内置工具(`mcp_server/`)

#### `generate_mermaid(description, diagram_type="flowchart")`

- 拼极度详尽的 prompt 调 LLM 产出纯 DSL。
- `_strip_fences` + `_sanitize_labels` 双重兜底,处理常见 LLM 输出脏数据。
- 失败返回兜底 flowchart,不抛异常,前端永远能渲染。

#### `send_wechat(content, msg_type="markdown")`

- 直接 POST 企业微信 webhook,errcode=0 视为成功。

### 9.2 外部 MCP server

编辑 `mcp_servers.json`,把 `_disabled_xxx` 改为 `xxx` 即可启用。`env` / `args` 支持 `${VAR}` 占位符,从宿主环境变量解析:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
    }
  }
}
```

**变量缺失时该 server 跳过 + 日志告警**,不影响其他工具与正常对话。

启动日志确认接入成功:
```
INFO:app.graph.nodes:MCP 服务器: ['eduagent', 'github']; 已加载工具: ['generate_mermaid', 'send_wechat', 'list_issues', 'create_issue', ...]
```

### 9.3 工具调度策略(统一注册表)

不再区分"内置直通"和"外部自主"两条路径,所有工具由 LLM 看 system prompt 自主决定调用与否:

| 风险等级 | system prompt 中的描述 | UI 提示 |
|---|---|---|
| `LOW`   | "风险:LOW",描述只读 / 无副作用 | 绿色徽标 |
| `MEDIUM`| "风险:MEDIUM",描述会发外部网络请求但不写状态 | 琥珀徽标 |
| `HIGH`  | "高风险(会改写远端/本地状态,仅当用户明确要求时调用)" | 红色徽标 + 警告图标 |

工具进入 LLM 决策的条件:
1. 在 `TOOLS_REGISTRY` 中已注册。
2. `enabled_tools[id]` 为 `True`(或缺失但 `default_enabled=True`)。
3. 对应的 MCP server 实际启动成功(外部 server 缺失环境变量则跳过)。

三者都满足才会进入 `bind_tools(...)` 列表。LLM 调用了"未启用"或"未注册"的工具时,后端会以 `status: rejected` 反馈给前端。

---

## 十、配置与环境变量

`backend/.env`(从 `.env.example` 复制后填):

```dotenv
# LLM
LLM_API_KEY=ms-xxxx
LLM_BASE_URL=https://api-inference.modelscope.cn/v1
LLM_MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507

# Embedding (1024 维须与 db_models.Vector(1024) 对齐)
EMBEDDING_API_KEY=sk-xxxx
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL_NAME=BAAI/bge-m3

# Reranker (可选)
RERANKER_API_KEY=
RERANKER_BASE_URL=https://api.siliconflow.cn/v1/rerank
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3

# DB (docker-compose 会覆盖为容器内地址)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/eduagent

# 工具
WECHAT_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
GITHUB_PERSONAL_ACCESS_TOKEN=
```

可调参数:
- `RAG_OVERFETCH_MULTIPLIER`(默认 4):向量层 over-fetch 倍数。
- `RAG_RRF_K`(默认 60):RRF 融合常数,值越大越偏向均匀融合。

---

## 十一、快速启动

### Docker Compose(推荐)

```bash
cp backend/.env.example backend/.env
# 编辑 backend/.env,至少填 LLM_API_KEY 和 EMBEDDING_API_KEY

docker compose up -d --build
docker compose logs -f backend
```

启动日志关注:
- `pgvector 扩展已启用` / `数据库表已创建` / `conversations 表列补齐`
- `MCP 服务器: ['eduagent', ...]; 已加载工具: [...]`
- `Application startup complete.`

### 本地开发(后端)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 确保本地有 PostgreSQL 16+ + pgvector 扩展
uvicorn app.main:app --reload --port 8000
```

### 本地开发(前端)

```bash
cd frontend
npm install
npm run dev    # 默认 http://localhost:5173,自动代理 /api → :8000
```

### 上传知识库 + 对话

```bash
curl -X POST http://localhost:8000/api/knowledge/upload -F "file=@第一章.md"
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"请总结第一章","session_id":"s1"}'
```

API 文档:http://localhost:8000/docs

---

## 十二、前端(简略)

技术栈:**React 18 + Vite 5 + TypeScript 5.6 + Tailwind 3.4 + react-markdown + mermaid 11**。

| 文件 | 职责 |
|---|---|
| `src/App.tsx` | 顶层布局,Sidebar + ChatView/KnowledgePanel 切换,生成会话 id |
| `src/components/ChatView.tsx` | SSE 解析,流式渲染,工具调用卡片,反馈按钮 |
| `src/components/ToolSelector.tsx` | 拉 `GET /api/tools` 渲染开关,按 internal/external 分组,LOW/MEDIUM/HIGH 风险徽标 |
| `src/components/Sidebar.tsx` | 会话列表,新建 / 选择 / 删除 |
| `src/components/MermaidRenderer.tsx` | 按需 init mermaid,渲染 DSL |
| `src/components/KnowledgePanel.tsx` | 知识库上传 / 列表 / 删除 |
| `src/lib/api.ts` | REST + 自实现 SSE 解析(`fetch` + `ReadableStream`,POST 走不了原生 EventSource);含 `getToolsConfig()` |
| `src/lib/types.ts` | Citation / MessageItem / ToolInvocation / UIMessage / ToolConfig / RiskLevel 等共享类型 |

SSE 解析关键点(`api.ts::streamChat`):
- 用 `fetch` + `ReadableStream` + `TextDecoder` 自实现 SSE,因为浏览器 `EventSource` 不支持 POST。
- 按 `\r?\n\r?\n` 分 event,`event:` / `data:` 行分别解析,兼容 `sse-starlette` 默认 CRLF。

工具调用卡片(`ChatView::ToolInvocationCard`):
- 折叠 `<details>`,头部展示工具名 + args 单行预览 + 状态图标。
- 四种状态各自一种样式:`success`(绿勾) / `error`(红 ✗) / `rejected`(琥珀 Ban,工具未启用) / `pending`(灰色旋转)。
- 展开后两块 `<pre>`:参数完整 JSON、返回摘要 or 错误堆栈。

---

## 十三、运维与排障

| 现象 | 排查 |
|---|---|
| 启动日志 `跳过 MCP server 'github'：环境变量未设置或为空` | 在 `backend/.env` 加 `GITHUB_PERSONAL_ACCESS_TOKEN=xxx` 后重启,并在前端 `ToolSelector` 中勾上 GitHub 工具 |
| `MCP 工具预热失败` | Node.js 缺失 / npx 网络不通 / npm 镜像源问题。Dockerfile 默认设了 `npmmirror`,自建镜像后请保留 |
| `pgvector 扩展已启用` 失败 | 用 `pgvector/pgvector:pg16` 镜像而非纯 Postgres |
| 流式接口 token 卡顿 | 检查 LLM 端点是否支持 stream;Qwen3 ModelScope 端点已验证支持 |
| `首请求很慢` | lifespan 预热失败导致冷启动,看启动日志确认 `MCP 工具已预热,共 N 个` |
| 切回旧会话看不到 mermaid / 工具卡片 | 该会话的消息是新字段加进 DB 之前写的,后续新消息会有 |
| 同名 `.md` 重传 chunk 翻倍 | 不会;upload 端点会先 DELETE 同名再插入 |
| 容器内挂载源码却不重载 | uvicorn `--reload` 监听 `/app`,docker-compose 把 `./backend` 挂为 `/app`;Windows 上文件改动可能延迟,WSL2 下正常 |

---

## 十四、路线图

- [ ] **接入 Alembic** 替换 `ADD COLUMN IF NOT EXISTS` 兜底
- [ ] **API 鉴权 + 限流**:目前 `/api/*` 全裸,生产前必接
- [ ] **高风险工具二次确认**:`HIGH` 风险工具被 LLM 调用时,弹出确认框由用户人工 approve 再执行
- [ ] **`_get_mcp_tools` 加锁**:消除冷启动并发 fork 子进程的竞争
- [ ] **工具调用统计**:把 `tool_invocations` 聚合到反馈表,跑评估时可观测哪些工具被滥用 / 被忽略
- [ ] **测试套件**:`pytest` + 关键路径覆盖(混合检索 / registry 过滤 / SSE 双轮流式)
- [ ] **可选切换** `ghcr.io/github/github-mcp-server`(GitHub 官方 Go 实现,功能更全)

---

## 十五、License

MIT
