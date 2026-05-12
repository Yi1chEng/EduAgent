# EduAgent — 教育智能体

基于 RAG + LangGraph + MCP 的教育领域 AI 助手。  
后端 FastAPI，向量库 PostgreSQL+pgvector，LLM 默认 ModelScope Qwen3-235B，Embedding 默认 SiliconFlow bge-m3。

---

## 架构

```
用户 → FastAPI ─► LangGraph 编排
                  rag_retriever ──► generator ──► [dispatcher (LLM 驱动 MCP)]
                       │                              │
                       ▼                              ▼
                  pgvector                 内置 mcp_server (mermaid/wechat)
                                           ＋ 外部 MCP servers (mcp_servers.json)
```

- **RAG**：Markdown 教材按标题切块 → bge-m3 向量化 → pgvector 余弦检索 → 自动引用 `[1][2]`
- **生成**：单次 LLM 调用，禁止在正文嵌入 mermaid（图由工具生成）
- **MCP 工具**：
  - 内置：`generate_mermaid`（LLM 生成 DSL）、`send_wechat`（企业微信推送）
  - 外部：通过 `mcp_servers.json` 接入任意 npx/python MCP server（filesystem、fetch、github 等）
  - 调度：LLM `bind_tools` 自主决定调用哪些工具

---

## 项目结构

```
EduAgent/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI 入口
│   │   ├── config.py          # Pydantic Settings
│   │   ├── api/{chat, knowledge, feedback}.py
│   │   ├── graph/{state, nodes, edges, builder}.py
│   │   ├── rag/{embeddings, retriever, citation, knowledge_loader}.py
│   │   ├── models/{schemas, db_models}.py
│   │   └── db/{database, init_db}.py
│   ├── Dockerfile             # Python + Node.js (用于 npx MCP servers)
│   └── requirements.txt
├── mcp_server/                # 内置 MCP，stdio 子进程
│   ├── server.py              # FastMCP 入口
│   └── tools/{mermaid, wechat}.py
├── mcp_servers.json           # 外部 MCP server 配置
├── frontend/                  # 待开发
├── docker-compose.yml
└── README.md
```

---

## 快速启动

```bash
# 1. 配置环境变量
cp backend/.env.example backend/.env
# 编辑 backend/.env，至少填 LLM_API_KEY 和 EMBEDDING_API_KEY

# 2. 启动
docker compose up -d --build

# 3. 上传知识库（Markdown 教材）
curl -X POST http://localhost:8000/api/knowledge/upload -F "file=@第一章.md"

# 4. 对话
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"请总结第一章","session_id":"s1"}'
```

API 文档：http://localhost:8000/docs

---

## API 总览

### 对话
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 同步对话，返回 `conversation_id`（用于 feedback） |
| POST | `/api/chat/stream` | SSE 流式，事件：`citation`/`token`/`mermaid`/`tool_result`/`done`/`error` |

### 会话管理
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/sessions` | 列出所有会话（按最后活动倒序） |
| GET | `/api/sessions/{id}/messages` | 拉取会话全部历史消息 |
| DELETE | `/api/sessions/{id}` | 删除会话及关联反馈 |

### 知识库
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/knowledge/upload` | 上传 Markdown 文件 |
| GET | `/api/knowledge/list` | 列出已导入文档 |
| DELETE | `/api/knowledge/{source_file}` | 删除指定文档全部 chunks |

### 反馈
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/feedback` | 对回答打分（rating ±1）+ 评论 |

---

## 集成第三方 MCP server

编辑 `mcp_servers.json`，把示例里 `_disabled_xxx` 改为 `xxx` 即可启用。`env` / `args` 支持 `${VAR}` 占位符，从 `backend/.env` 读取（避免密钥进仓库）；变量缺失时启动会**跳过该 server 并告警**，不影响其他工具。

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/app/uploads"]
    }
  }
}
```

启用 GitHub MCP：

1. 在 `backend/.env` 写入 `GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxx`（[生成入口](https://github.com/settings/personal-access-tokens)）
2. `docker compose restart backend`
3. 启动日志看到 `MCP 服务器: [..., 'github']` 即接入成功

LLM 会在 `dispatcher` 阶段看到所有工具，按需自动调用(列 issue、查 PR、读文件、提交等)。

---

## 技术栈

- Python 3.11 / FastAPI / SQLAlchemy 2.0 async
- PostgreSQL 16 + pgvector
- LangGraph + langchain-openai + langchain-mcp-adapters
- MCP (FastMCP) / Node.js 22（用于 npx 启动外部 MCP）

---

## License

MIT
