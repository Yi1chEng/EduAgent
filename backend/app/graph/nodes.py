"""LangGraph 图节点：RAG 检索、生成、MCP 工具调用（LLM 驱动）。"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.graph.state import AgentState
from app.rag.citation import build_citation_list, format_citations, format_context
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)

# 外部 MCP 配置文件路径（容器内由 volume 挂载）
EXTERNAL_MCP_CONFIG_PATH = "/mcp_servers.json"

_llm: ChatOpenAI | None = None
_mcp_tools: dict[str, Any] | None = None


def _get_llm() -> ChatOpenAI:
    """延迟初始化 LLM。"""
    global _llm
    if _llm is None:
        s = get_settings()
        _llm = ChatOpenAI(
            api_key=s.LLM_API_KEY,
            base_url=s.LLM_BASE_URL,
            model=s.LLM_MODEL_NAME,
            temperature=0.7,
        )
    return _llm


# ${VAR} 占位符：用于在 mcp_servers.json 的 env/args 中引用环境变量，避免把密钥写入仓库
_ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_placeholders(value: str) -> tuple[str, list[str]]:
    """展开字符串中的 ${VAR} 占位符。

    返回 (展开后的字符串, 缺失的变量名列表)。变量未设置或为空时记入缺失列表。
    """
    missing: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        var = match.group(1)
        resolved = os.environ.get(var, "")
        if not resolved:
            missing.append(var)
        return resolved

    return _ENV_PLACEHOLDER_PATTERN.sub(_sub, value), missing


def _load_external_mcp_servers() -> dict[str, dict[str, Any]]:
    """从 mcp_servers.json 读取外部 MCP server 配置。

    - 跳过以 '_' 开头的 key（约定为禁用项）。
    - 支持 env 和 args 中的 ${VAR} 占位符，从宿主环境变量解析。
    - 若占位符引用的变量缺失，跳过该 server 并告警（避免子进程启动后才报错）。
    """
    path = Path(EXTERNAL_MCP_CONFIG_PATH)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"无法解析 {path}: {e}")
        return {}

    servers = data.get("mcpServers", {})
    result: dict[str, dict[str, Any]] = {}
    for name, cfg in servers.items():
        if name.startswith("_"):
            continue

        missing_vars: list[str] = []

        env_overrides: dict[str, str] = {}
        for key, raw in cfg.get("env", {}).items():
            if isinstance(raw, str):
                expanded, missing = _expand_env_placeholders(raw)
                missing_vars.extend(missing)
                env_overrides[key] = expanded
            else:
                env_overrides[key] = raw

        expanded_args: list[str] = []
        for raw in cfg.get("args", []):
            if isinstance(raw, str):
                expanded, missing = _expand_env_placeholders(raw)
                missing_vars.extend(missing)
                expanded_args.append(expanded)
            else:
                expanded_args.append(raw)

        if missing_vars:
            logger.warning(
                f"跳过 MCP server '{name}'：环境变量未设置或为空 {sorted(set(missing_vars))}"
            )
            continue

        result[name] = {
            "command": cfg["command"],
            "args": expanded_args,
            "transport": "stdio",
            "env": {**os.environ, **env_overrides},
        }
    return result


async def _get_mcp_tools() -> dict[str, Any]:
    """加载内置 + 外部 MCP server 的所有工具，缓存为 {name: tool} 字典。"""
    global _mcp_tools
    if _mcp_tools is None:
        s = get_settings()
        env = os.environ.copy()
        # 内置 mcp_server 需要的环境变量（wechat 推送、mermaid LLM 生成）
        env.update({
            "WECHAT_WEBHOOK_URL": s.WECHAT_WEBHOOK_URL,
            "LLM_API_KEY": s.LLM_API_KEY,
            "LLM_BASE_URL": s.LLM_BASE_URL,
            "LLM_MODEL_NAME": s.LLM_MODEL_NAME,
        })

        connections: dict[str, dict[str, Any]] = {
            "eduagent": {
                "command": s.MCP_SERVER_COMMAND,
                "args": [s.MCP_SERVER_ARGS],
                "transport": "stdio",
                "env": env,
            }
        }
        connections.update(_load_external_mcp_servers())

        client = MultiServerMCPClient(connections)
        try:
            tools = await client.get_tools()
        except Exception as e:
            logger.error(f"MCP 工具加载失败: {e}")
            _mcp_tools = {}
            return _mcp_tools

        _mcp_tools = {t.name: t for t in tools}
        logger.info(
            f"MCP 服务器: {list(connections.keys())}; "
            f"已加载工具: {list(_mcp_tools.keys())}"
        )
    return _mcp_tools


GENERATOR_SYSTEM_PROMPT = """你是 EduAgent，专业的教育AI助手。基于参考资料生成准确、有教育价值的回答。

规则：
1. 必须基于参考资料，使用 [1][2] 等标记引用来源
2. 资料不足时坦诚说明
3. 使用 Markdown 格式
4. 复杂概念用类比或例子解释
5. **关于图表的硬性规定**：
   - **绝对不要**说"我无法生成图片/图表"、"由于版权限制"、"请使用 Draw.io / Mermaid"等推脱话术
   - **绝对不要**在正文中嵌入 mermaid / flowchart / mindmap 代码块
   - 系统会**自动**通过独立工具调用生成 Mermaid 图表并渲染给用户，你不需要也不应该自己画
   - 如果用户要求图表，正文专心解释知识内容即可；图表会作为附加产物自动出现在你的回答下方

参考资料：
{context}

引用来源：
{citations}
"""


TOOL_AGENT_SYSTEM_PROMPT = """你是工具调度助手。基于下方信息决定调用哪些工具。

强约束：
- 如果 need_visualization=True，**必须**调用 generate_mermaid：
    * description 参数：**必须把"用户原始问题 + 已生成的回答全文"完整拼接传入**，不要自己提炼缩写，否则图会很简单。
    * diagram_type：**默认 flowchart**；当问题在询问"知识体系/分类/构成/有哪些"时用 mindmap；流程性强用 flowchart；时间顺序用 timeline；对比用 classDiagram。
- 如果 need_dispatch=True，**必须**调用 send_wechat，content 用已生成的回答主体（截取前 1500 字符）。
- 如果两个标志都为 False，根据用户问题语义自行判断；若无需工具，直接回复"无需工具"。
- 工具调用结果会自动返回给用户，你无需复述。

用户原始问题：{query}
need_visualization={need_visualization}, need_dispatch={need_dispatch}

已生成的回答（**调用 generate_mermaid 时必须把这段完整文本作为 description 的主体**）：
{generated_content}
"""


async def rag_retriever_node(state: AgentState) -> dict[str, Any]:
    """RAG 检索节点。"""
    query = state.get("query", "")
    logger.info(f"RAG 检索: query='{query}'")
    try:
        docs = await retrieve(query, top_k=5)
        citations = build_citation_list(docs)
        logger.info(f"检索到 {len(docs)} 个相关文档")
    except Exception as e:
        logger.error(f"RAG 检索失败: {e}")
        docs, citations = [], []
    return {"retrieved_docs": docs, "citations": citations}


async def generator_node(state: AgentState) -> dict[str, Any]:
    """内容生成节点：基于 RAG 结果生成回答。"""
    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    messages = state.get("messages", [])

    system_prompt = GENERATOR_SYSTEM_PROMPT.format(
        context=format_context(retrieved_docs),
        citations=format_citations(retrieved_docs),
    )
    llm_messages = [SystemMessage(content=system_prompt)]
    llm_messages.extend(messages[-10:])
    llm_messages.append(HumanMessage(content=query))

    response = await _get_llm().ainvoke(llm_messages)
    return {"generated_content": response.content}


def _infer_diagram_type(query: str) -> str:
    """根据用户问题特征猜测合适的 mermaid 图类型。"""
    q = query.lower()
    if any(k in query for k in ["知识体系", "分类", "构成", "有哪些", "组成", "脑图", "思维导图", "mindmap"]):
        return "mindmap"
    if any(k in query for k in ["时间", "历史", "演变", "发展史", "timeline"]):
        return "timeline"
    if any(k in query for k in ["对比", "区别", "差异", "vs", "比较"]):
        return "classDiagram"
    if "时序" in query or "sequence" in q:
        return "sequenceDiagram"
    return "flowchart"


async def dispatcher_node(state: AgentState) -> dict[str, Any]:
    """MCP 工具调度节点。

    策略：
    - 显式标志 (need_visualization / need_dispatch) → **直接调工具**，跳过 LLM 决策，
      保证完整内容传入，且省一次 LLM 调用。
    - 无显式标志但有外部 MCP server → 让 LLM 通过 tool-calling 自主决定。
    """
    tool_results: dict[str, Any] = {}
    mermaid_code: str = ""

    tools_dict = await _get_mcp_tools()
    if not tools_dict:
        return {"tool_results": tool_results, "mermaid_code": mermaid_code}

    need_viz = state.get("need_visualization", False)
    need_dispatch = state.get("need_dispatch", False)
    query = state.get("query", "")
    generated_content = state.get("generated_content", "")

    # ===== 直通路径 1：生成图表 =====
    if need_viz and "generate_mermaid" in tools_dict:
        full_desc = (
            f"用户原始问题：{query}\n\n"
            f"基于以下完整知识内容生成丰富、详细的图表（覆盖所有关键概念、子步骤、分支与关系）：\n\n"
            f"{generated_content}"
        )
        diagram_type = _infer_diagram_type(query)
        logger.info(f"直接调用 generate_mermaid (diagram_type={diagram_type}, desc_len={len(full_desc)})")
        try:
            result = await tools_dict["generate_mermaid"].ainvoke(
                {"description": full_desc, "diagram_type": diagram_type}
            )
            mermaid_code = _extract_text(result)
            tool_results["generate_mermaid"] = {"status": "success", "result": mermaid_code}
        except Exception as e:
            logger.error(f"generate_mermaid 失败: {e}")
            tool_results["generate_mermaid"] = {"status": "error", "message": str(e)}

    # ===== 直通路径 2：企业微信推送 =====
    if need_dispatch and "send_wechat" in tools_dict:
        content = generated_content[:1500] if generated_content else query
        logger.info(f"直接调用 send_wechat (content_len={len(content)})")
        try:
            result = await tools_dict["send_wechat"].ainvoke({"content": content})
            tool_results["send_wechat"] = {"status": "success", "result": _extract_text(result)}
        except Exception as e:
            logger.error(f"send_wechat 失败: {e}")
            tool_results["send_wechat"] = {"status": "error", "message": str(e)}

    # ===== LLM 自主路径：仅在无显式标志且有 MCP 工具时 =====
    if not need_viz and not need_dispatch:
        prompt = TOOL_AGENT_SYSTEM_PROMPT.format(
            query=query,
            need_visualization=False,
            need_dispatch=False,
            generated_content=generated_content[:1500],
        )
        llm_with_tools = _get_llm().bind_tools(list(tools_dict.values()))
        response = await llm_with_tools.ainvoke([HumanMessage(content=prompt)])
        for call in getattr(response, "tool_calls", []) or []:
            name = call["name"]
            args = call.get("args", {})
            if name not in tools_dict:
                tool_results[name] = {"status": "error", "message": "tool not found"}
                continue
            try:
                result = await tools_dict[name].ainvoke(args)
                text_result = _extract_text(result)
                tool_results[name] = {"status": "success", "result": text_result}
                if name == "generate_mermaid":
                    mermaid_code = text_result
                logger.info(f"LLM 调用工具 {name} 成功")
            except Exception as e:
                logger.error(f"LLM 调用工具 {name} 失败: {e}")
                tool_results[name] = {"status": "error", "message": str(e)}

    return {"tool_results": tool_results, "mermaid_code": mermaid_code}


def _extract_text(result: Any) -> str:
    """从 MCP 工具返回中提取纯文本（处理 TextContent 列表/dict/字符串多种形态）。"""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if hasattr(result, "text"):
        return result.text
    return str(result)
