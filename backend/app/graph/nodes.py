"""LangGraph 图节点：RAG 检索 + 生成（含 LLM 自主工具调用）。

设计说明：
- 不再区分 internal / external 工具，全部由 TOOLS_REGISTRY 统一描述。
- 不再有独立的 dispatcher_node：generator 节点直接 bind_tools(enabled) 让 LLM 自主决策。
- chat.py 的流式接口会重用本模块的 prompt + tool 加载逻辑，但自己跑 astream。
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.graph.state import AgentState
from app.rag.citation import build_citation_list, format_citations, format_context
from app.rag.retriever import retrieve
from app.tools.registry import TOOLS_REGISTRY, get_available_tools

logger = logging.getLogger(__name__)

# 外部 MCP 配置文件路径（容器内由 volume 挂载）
EXTERNAL_MCP_CONFIG_PATH = "/mcp_servers.json"

_llm: ChatOpenAI | None = None
_mcp_tools: dict[str, Any] | None = None
_mcp_tools_by_server: dict[str, dict[str, Any]] | None = None


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
    """展开字符串中的 ${VAR} 占位符。返回 (展开后字符串, 缺失变量名列表)。"""
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

    跳过以 '_' 开头的禁用项；env/args 中的 ${VAR} 缺失时跳过该 server。
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


async def _get_mcp_tools_by_server() -> dict[str, dict[str, Any]]:
    """加载所有 MCP server 的工具，按 server 分组缓存：{server: {tool_name: tool}}。

    每个 server 单独跑一次 MCP client.get_tools()，是为了拿到稳定的"工具属于哪个 server"信息
    （langchain-mcp-adapters 的扁平 get_tools 不会回填 server 来源）。
    多 fork 的代价仅在冷启动时一次，且很多 server（如 npx 的 node 进程）本就是独立子进程。
    """
    global _mcp_tools_by_server
    if _mcp_tools_by_server is not None:
        return _mcp_tools_by_server

    s = get_settings()
    env = os.environ.copy()
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

    by_server: dict[str, dict[str, Any]] = {}
    for server_name, server_conn in connections.items():
        try:
            single_client = MultiServerMCPClient({server_name: server_conn})
            tools = await single_client.get_tools()
            by_server[server_name] = {t.name: t for t in tools}
            logger.info(
                f"MCP server '{server_name}' 已加载工具: {list(by_server[server_name].keys())}"
            )
        except Exception as e:
            logger.error(f"MCP server '{server_name}' 加载失败: {e}")
            by_server[server_name] = {}

    _mcp_tools_by_server = by_server
    return _mcp_tools_by_server


async def _get_mcp_tools() -> dict[str, Any]:
    """扁平视图：{tool_name: tool}。保留作为兼容入口，内部由 _get_mcp_tools_by_server 派生。"""
    global _mcp_tools
    if _mcp_tools is None:
        by_server = await _get_mcp_tools_by_server()
        _mcp_tools = {
            name: tool
            for server_tools in by_server.values()
            for name, tool in server_tools.items()
        }
    return _mcp_tools


def _format_tool_line(meta: Any) -> str:
    """把一条 ToolMetadata 渲染成提示词里的列表项。"""
    risk_tag = (
        "高风险（会改写远端/本地状态，仅当用户明确要求时调用）"
        if meta.risk_level.value == "HIGH"
        else f"风险:{meta.risk_level.value}"
    )
    return f"- `{meta.id}`（{meta.display_name}，{risk_tag}）：{meta.description}"


def build_system_prompt(
    retrieved_docs: list[dict[str, Any]],
    enabled_tools: dict[str, bool] | None,
) -> str:
    """统一系统提示词。

    Agent 始终能看到**全量工具菜单**（registry），但只能调用本轮 enabled_tools=True 的工具。
    被用户禁用的工具会以"未启用"标签列出，LLM 可以**告知用户"如需此功能请在工具开关中启用"**，
    但绝不主动调用。
    """
    enabled_tools = enabled_tools or {}

    enabled_lines: list[str] = []
    disabled_lines: list[str] = []
    for tool_id, meta in TOOLS_REGISTRY.items():
        line = _format_tool_line(meta)
        if enabled_tools.get(tool_id, False):
            enabled_lines.append(line)
        else:
            disabled_lines.append(line)

    sections: list[str] = []
    if enabled_lines:
        sections.append("本轮**可直接调用**的工具：\n" + "\n".join(enabled_lines))
    else:
        sections.append("本轮没有任何工具被启用（无法调用任何工具）。")

    if disabled_lines:
        sections.append(
            "用户**未启用**的工具（菜单里有，但本轮不可调用）：\n"
            + "\n".join(disabled_lines)
            + "\n注意：上述工具不在本轮 bind_tools 列表里，**不要尝试调用**；"
            "当用户需要相关功能时，请告诉他在输入框上方的【工具】面板里勾选对应项后再发送。"
        )

    tools_section = "\n\n".join(sections) + (
        "\n\n工具调用原则：\n"
        "1. 仅当任务确实需要时才调用；闲聊或简单回答时不要强行调用工具。\n"
        "2. 高风险工具必须有用户的明确意图（如 推送一下、保存到文件），不要主动调用。\n"
        "3. 工具调用结果会自动展示给用户，正文中不必复述。\n"
        "4. 如需图表/可视化，调用 generate_mermaid，把【用户问题 + 当前回答全文】"
        "作为 description 传入；正文专注解释知识，不要嵌入 mermaid 代码块。\n"
        "5. 当用户问【你有哪些工具 / 你能做什么】时，把上面【可直接调用】和【未启用】两类完整告诉他。\n"
    )

    return (
        "你是 EduAgent，专业的教育 AI 助手。基于参考资料生成准确、有教育价值的回答。\n\n"
        "输出规则：\n"
        "1. 必须基于参考资料，使用 [1][2] 等标记引用来源。\n"
        "2. 资料不足时坦诚说明，不要编造。\n"
        "3. 使用 Markdown 格式；复杂概念用类比或例子解释。\n"
        "4. 不要在正文里嵌入 mermaid / flowchart 代码块——需要图表时调用工具。\n\n"
        f"{tools_section}\n"
        f"参考资料：\n{format_context(retrieved_docs)}\n\n"
        f"引用来源：\n{format_citations(retrieved_docs)}\n"
    )


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
    """生成节点：基于 RAG 结果生成回答；按 enabled_tools 给 LLM 暴露可调工具。

    流程：
    1. 拼系统提示词（含本轮启用的工具说明）
    2. LLM 一次性生成；如有 tool_calls，依次执行并把结果回灌给 LLM 再生成最终正文
    3. 收集 tool_invocations + mermaid_code 落到 state
    """
    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    messages = state.get("messages", [])
    enabled_tools = state.get("enabled_tools", {})

    system_prompt = build_system_prompt(retrieved_docs, enabled_tools)

    loaded_by_server = await _get_mcp_tools_by_server()
    available = get_available_tools(loaded_by_server, enabled_tools)
    available_by_name = {t.name: t for t in available}

    llm = _get_llm()
    if available:
        llm = llm.bind_tools(available)

    history_msgs = [
        SystemMessage(content=system_prompt),
        *messages[-10:],
        HumanMessage(content=query),
    ]
    response = await llm.ainvoke(history_msgs)

    tool_invocations: list[dict[str, Any]] = []
    mermaid_code = ""
    tool_calls = getattr(response, "tool_calls", []) or []

    if tool_calls:
        followup_msgs = [*history_msgs, response]
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            call_id = call.get("id", "")
            tool = available_by_name.get(name)
            if tool is None:
                tool_invocations.append(
                    {"name": name, "args": args, "status": "error", "error": "tool not enabled"}
                )
                followup_msgs.append(
                    ToolMessage(content="tool not enabled", tool_call_id=call_id)
                )
                continue
            try:
                raw = await tool.ainvoke(args)
                text_result = _extract_text(raw)
                tool_invocations.append(
                    {"name": name, "args": args, "status": "success", "result": text_result}
                )
                if name == "generate_mermaid":
                    mermaid_code = text_result
                followup_msgs.append(
                    ToolMessage(content=text_result, tool_call_id=call_id)
                )
            except Exception as e:
                logger.error(f"工具 {name} 调用失败: {e}", exc_info=True)
                tool_invocations.append(
                    {"name": name, "args": args, "status": "error", "error": str(e)}
                )
                followup_msgs.append(
                    ToolMessage(content=f"error: {e}", tool_call_id=call_id)
                )

        # 让 LLM 基于工具结果生成最终正文（不再绑工具，避免无限循环）
        final = await _get_llm().ainvoke(followup_msgs)
        content = _content_to_text(final.content)
    else:
        content = _content_to_text(response.content)

    return {
        "generated_content": content,
        "tool_invocations": tool_invocations,
        "mermaid_code": mermaid_code,
    }


def _content_to_text(content: Any) -> str:
    """把 LangChain 消息 content（可能是 list[dict]）拍平成字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


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
