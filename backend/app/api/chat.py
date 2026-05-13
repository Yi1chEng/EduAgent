"""对话与会话管理接口：同步对话 / SSE 流式 / 会话列表 / 历史 / 删除。"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import delete, select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.db.database import async_session_factory, get_db
from app.graph.builder import app_graph
from app.graph.edges import _has_external_mcp_servers
from app.graph.nodes import (
    GENERATOR_SYSTEM_PROMPT,
    TOOL_AGENT_SYSTEM_PROMPT,
    _extract_text,
    _get_llm,
    _get_mcp_tools,
    _infer_diagram_type,
)
from app.graph.titles import schedule_title_generation
from app.models.db_models import Conversation, Feedback, Session as SessionModel
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    MessageItem,
    SessionDeleteResponse,
    SessionListItem,
    SessionListResponse,
    SessionMessagesResponse,
)
from app.rag.citation import (
    build_citation_list,
    format_citations,
    format_context,
)
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)

router = APIRouter()


# 触发自动可视化的关键词（用户问题命中即视为 need_visualization=True）
_VIZ_KEYWORDS = (
    "流程图", "流程", "图表", "可视化", "示意图", "结构图",
    "脑图", "思维导图", "mindmap", "mermaid",
    "画一个", "画一张", "画图", "画出", "图示", "图解",
    "知识图谱", "关系图", "时序图", "类图", "架构图",
)


def _detect_visualization_intent(query: str) -> bool:
    """根据用户问题文本判断是否需要生成图表。"""
    q = query.lower()
    return any(k.lower() in q for k in _VIZ_KEYWORDS)


# 明确不需要查知识库的"短语精确匹配"集合
_NO_RAG_EXACT = {
    "你好", "您好", "hi", "hello", "hey", "嗨",
    "早上好", "上午好", "下午好", "晚上好", "晚安",
    "谢谢", "多谢", "感谢", "thanks", "thank you", "ok", "okay", "好的", "收到",
    "再见", "拜拜", "bye", "goodbye",
    "测试", "test", "ping",
}

# 含这些子串可视为闲聊/元问题（不需要 RAG）
_NO_RAG_SUBSTRINGS = (
    "你是谁", "你叫什么", "你能做什么", "你有什么功能",
    "what can you do", "who are you", "your name",
)


def _needs_rag(query: str) -> bool:
    """启发式：判断当前 query 是否需要走 RAG 检索。

    保守策略：明显的闲聊/打招呼/致谢/元问题返回 False，其它一律返回 True。
    零额外延迟、零 LLM 调用。
    """
    q = query.strip().lower().rstrip("。.!！?？~～")
    if not q:
        return False
    if q in _NO_RAG_EXACT:
        return False
    if any(s in q for s in _NO_RAG_SUBSTRINGS):
        return False
    # 极短查询（去标点后 ≤2 个字符）多为闲聊
    stripped = "".join(c for c in q if c.isalnum())
    if len(stripped) <= 2:
        return False
    return True


# ===== Helpers =====

async def _save_conversation(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    citations_json: str | None = None,
    mermaid_code: str | None = None,
    tool_invocations_json: str | None = None,
) -> Conversation:
    """保存对话记录到数据库。"""
    conv = Conversation(
        session_id=session_id,
        role=role,
        content=content,
        citations_json=citations_json,
        mermaid_code=mermaid_code,
        tool_invocations_json=tool_invocations_json,
    )
    db.add(conv)
    await db.flush()
    return conv


async def _load_history(db: AsyncSession, session_id: str, limit: int = 20) -> list[Any]:
    """加载指定会话的历史消息（按时间正序）。"""
    stmt = (
        select(Conversation)
        .where(Conversation.session_id == session_id)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = list(reversed(result.scalars().all()))

    messages = []
    for row in rows:
        if row.role == "user":
            messages.append(HumanMessage(content=row.content))
        elif row.role == "assistant":
            messages.append(AIMessage(content=row.content))
    return messages


def _build_initial_state(request_or_kwargs: dict[str, Any], history: list[Any]) -> dict[str, Any]:
    """构建 LangGraph 初始 state。"""
    return {
        "messages": history,
        "query": request_or_kwargs["query"],
        "session_id": request_or_kwargs["session_id"],
        "need_visualization": request_or_kwargs.get("need_visualization", False),
        "need_dispatch": request_or_kwargs.get("need_dispatch", False),
        "retrieved_docs": [],
        "citations": [],
        "generated_content": "",
        "tool_results": {},
        "mermaid_code": "",
    }


# ===== 对话接口 =====

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)) -> ChatResponse:
    """同步对话接口：执行完整 LangGraph 流程，一次性返回结果。"""
    try:
        history = await _load_history(db, request.session_id)
        is_first_turn = len(history) == 0
        await _save_conversation(db, request.session_id, "user", request.query)

        payload = request.model_dump()
        # 自动可视化意图检测：开关未开但问题里有关键词时也触发
        if not payload.get("need_visualization") and _detect_visualization_intent(request.query):
            payload["need_visualization"] = True
            logger.info(f"自动检测到可视化意图，启用 need_visualization")

        result = await app_graph.ainvoke(_build_initial_state(payload, history))

        content = result.get("generated_content", "")
        citations = result.get("citations", [])
        mermaid_code = result.get("mermaid_code") or None
        tool_results = result.get("tool_results") or None

        # 老 dispatcher_node 的 tool_results 是 {name: {status, result}}，
        # 转换为统一的 tool_invocations 列表存盘（args 在该路径下缺失，留空 dict）
        tool_invocations_json = (
            json.dumps(
                _legacy_tool_results_to_invocations(tool_results),
                ensure_ascii=False,
            )
            if tool_results
            else None
        )

        assistant_conv = await _save_conversation(
            db,
            request.session_id,
            "assistant",
            content,
            json.dumps(citations, ensure_ascii=False) if citations else None,
            mermaid_code=mermaid_code,
            tool_invocations_json=tool_invocations_json,
        )

        if is_first_turn:
            schedule_title_generation(request.session_id, request.query)

        return ChatResponse(
            conversation_id=assistant_conv.id,
            content=content,
            citations=citations,
            mermaid_code=mermaid_code,
            tool_results=tool_results,
        )

    except Exception as e:
        logger.error(f"对话处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"对话处理失败: {e}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> EventSourceResponse:
    """SSE 真·token 流式对话：手动编排 RAG → 流式 LLM → dispatcher。

    注意：不能用 `Depends(get_db)` —— FastAPI 在 endpoint return EventSourceResponse 时
    会立即关闭依赖注入的 db session，但 SSE generator 才刚开始迭代，所有 _save_conversation
    都会写入一个已关闭的连接（事务回滚 + 消息丢失）。这里在 generator 内部用独立 session
    并显式 commit。

    优化点：
    1. 用户消息保存与 RAG 检索并行（asyncio.gather）
    2. RAG 完成立即推 citation
    3. LLM 用 astream() 逐 token 推送（首字延迟 ≈ TTFT）
    4. dispatcher 仅在需要时调用，结果实时推送

    事件：citation / token / mermaid / tool_result / done(=conversation_id) / error
    """

    # 自动可视化意图检测：开关未开但问题里有关键词时也触发
    need_visualization = request.need_visualization or _detect_visualization_intent(request.query)
    if need_visualization and not request.need_visualization:
        logger.info("自动检测到可视化意图，启用 need_visualization")

    use_rag = _needs_rag(request.query)
    if not use_rag:
        logger.info(f"启发式判定无需 RAG: {request.query!r}")

    async def event_generator():
        # 自管 session：声明周期与 generator 自身绑定，避免被 FastAPI 提前关闭
        async with async_session_factory() as db:
            try:
                # ===== Stage 1: RAG（按需）+ DB 写入 =====
                retrieve_task = (
                    asyncio.create_task(retrieve(request.query, top_k=5)) if use_rag else None
                )
                history = await _load_history(db, request.session_id)
                is_first_turn = len(history) == 0
                await _save_conversation(db, request.session_id, "user", request.query)
                await db.commit()  # 用户消息单独提交，确保即使后续异常也已落库
                retrieved_docs = await retrieve_task if retrieve_task is not None else []

                citations = build_citation_list(retrieved_docs) if retrieved_docs else []
                yield {
                    "event": "citation",
                    "data": json.dumps(citations, ensure_ascii=False),
                }

                # ===== Stage 2: 并行启动 mermaid 生成（与正文流式同时进行） =====
                # 关键性能点：不等正文写完，用 query + RAG 上下文当 description 直接开跑
                mermaid_task: asyncio.Task[str] | None = None
                if need_visualization:
                    mermaid_task = asyncio.create_task(
                        _call_mermaid_with_context(request.query, retrieved_docs)
                    )

                # ===== Stage 3: 流式生成正文 =====
                if use_rag:
                    system_prompt = GENERATOR_SYSTEM_PROMPT.format(
                        context=format_context(retrieved_docs),
                        citations=format_citations(retrieved_docs),
                    )
                else:
                    # 闲聊/元问题直接用简短系统提示，避免"资料不足"的尴尬回复
                    system_prompt = (
                        "你是 EduAgent，专业的教育AI助手。直接、自然地回应用户消息。"
                        "如果用户在打招呼或闲聊，简短热情地回应；如果在问你的功能，"
                        "说明你能基于上传的教材回答学习问题、生成图表、推送笔记。"
                    )
                llm_messages = [SystemMessage(content=system_prompt)]
                llm_messages.extend(history[-10:])
                llm_messages.append(HumanMessage(content=request.query))

                final_content_parts: list[str] = []
                async for chunk in _get_llm().astream(llm_messages):
                    token = getattr(chunk, "content", "")
                    if not token:
                        continue
                    if isinstance(token, list):
                        token = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in token
                        )
                    final_content_parts.append(token)
                    yield {"event": "token", "data": token}

                final_content = "".join(final_content_parts)

                # ===== Stage 4: 持久化 + 立即发 done（用户立刻能看到完整正文） =====
                assistant_conv = await _save_conversation(
                    db,
                    request.session_id,
                    "assistant",
                    final_content,
                    json.dumps(citations, ensure_ascii=False) if citations else None,
                )
                await db.commit()
                yield {"event": "done", "data": str(assistant_conv.id)}

                if is_first_turn:
                    schedule_title_generation(request.session_id, request.query)

                # ===== Stage 5: 等并行的 mermaid 任务结束，单独推 mermaid 事件 =====
                final_mermaid_code: str = ""
                if mermaid_task is not None:
                    try:
                        final_mermaid_code = (await mermaid_task) or ""
                        if final_mermaid_code:
                            yield {"event": "mermaid", "data": final_mermaid_code}
                    except Exception as e:
                        logger.error(f"并行 mermaid 失败: {e}")

                # ===== Stage 6: 工具调用（互斥：显式微信推送 vs LLM 自主调用外部 MCP） =====
                collected_invocations: list[dict[str, Any]] = []
                if request.need_dispatch:
                    async for invocation in _stream_wechat_dispatch(
                        request.query, final_content
                    ):
                        collected_invocations.append(invocation)
                        yield {
                            "event": "tool_result",
                            "data": json.dumps(invocation, ensure_ascii=False),
                        }
                elif request.allow_external_tools and _has_external_mcp_servers():
                    async for invocation in _stream_autonomous_tools(
                        request.query, final_content
                    ):
                        collected_invocations.append(invocation)
                        yield {
                            "event": "tool_result",
                            "data": json.dumps(invocation, ensure_ascii=False),
                        }

                # ===== Stage 7: 工件落盘（mermaid / 工具调用），刷新历史可重放 =====
                # 仅当有内容才写 UPDATE，避免无谓 IO
                if final_mermaid_code or collected_invocations:
                    assistant_conv.mermaid_code = final_mermaid_code or None
                    assistant_conv.tool_invocations_json = (
                        json.dumps(collected_invocations, ensure_ascii=False)
                        if collected_invocations
                        else None
                    )
                    await db.commit()

            except Exception as e:
                logger.error(f"流式对话失败: {e}", exc_info=True)
                await db.rollback()
                yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())


def _legacy_tool_results_to_invocations(
    tool_results: dict[str, Any],
) -> list[dict[str, Any]]:
    """把老 dispatcher_node 的 {name: {status, result|message}} 转成统一 invocations 列表。

    args 字段在该路径下缺失，留空 dict；用于把同步 /chat 也补齐持久化形态。
    """
    invocations: list[dict[str, Any]] = []
    for name, payload in tool_results.items():
        if not isinstance(payload, dict):
            continue
        status = payload.get("status", "success")
        item: dict[str, Any] = {"name": name, "args": {}, "status": status}
        if status == "error":
            item["error"] = payload.get("message") or payload.get("error", "")
        else:
            item["result"] = payload.get("result", "")
        invocations.append(item)
    return invocations


# SSE tool_result 事件载荷上限：避免大返回阻塞流 / 撑爆前端
_TOOL_RESULT_PREVIEW_LIMIT = 2000
# 内置工具集合：已被其他直通路径处理（mermaid 并行、wechat 显式），不进入 LLM 自主决策
_BUILTIN_TOOL_NAMES: frozenset[str] = frozenset({"generate_mermaid", "send_wechat"})


def _truncate_preview(text: str, limit: int = _TOOL_RESULT_PREVIEW_LIMIT) -> str:
    """截断超长文本并附省略号，避免单个工具结果撑爆 SSE 单帧。"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…(已截断，原文 {len(text)} 字符)"


async def _stream_wechat_dispatch(
    query: str, generated_content: str
) -> AsyncGenerator[dict[str, Any], None]:
    """显式微信推送路径：直接调用 send_wechat，按 per-tool 形态 yield 调用记录。"""
    tools = await _get_mcp_tools()
    tool = tools.get("send_wechat")
    if tool is None:
        return
    args = {"content": (generated_content[:1500] if generated_content else query)}
    try:
        result = await tool.ainvoke(args)
        yield {
            "name": "send_wechat",
            "args": args,
            "status": "success",
            "result": _truncate_preview(_extract_text(result), 500),
        }
    except Exception as e:
        logger.error(f"send_wechat 失败: {e}", exc_info=True)
        yield {
            "name": "send_wechat",
            "args": args,
            "status": "error",
            "error": str(e),
        }


async def _stream_autonomous_tools(
    query: str, generated_content: str
) -> AsyncGenerator[dict[str, Any], None]:
    """LLM 自主调用外部 MCP 工具（如 github），按调用顺序 yield 结果。

    - 只暴露非内置工具给 LLM，避免与并行 mermaid / 显式 wechat 路径打架。
    - 每个 tool_call 独立 yield 一次：success 携带截断后的 result 摘要，error 携带 message。
    """
    tools_dict = await _get_mcp_tools()
    external_tools = {
        n: t for n, t in tools_dict.items() if n not in _BUILTIN_TOOL_NAMES
    }
    if not external_tools:
        return

    prompt = TOOL_AGENT_SYSTEM_PROMPT.format(
        query=query,
        need_visualization=False,
        need_dispatch=False,
        generated_content=generated_content[:1500],
    )
    try:
        llm_with_tools = _get_llm().bind_tools(list(external_tools.values()))
        response = await llm_with_tools.ainvoke([HumanMessage(content=prompt)])
    except Exception as e:
        logger.error(f"LLM 工具决策失败: {e}", exc_info=True)
        return

    tool_calls = getattr(response, "tool_calls", []) or []
    if not tool_calls:
        return

    logger.info(
        f"LLM 决定调用 {len(tool_calls)} 个外部工具: "
        f"{[c.get('name') for c in tool_calls]}"
    )
    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("args", {}) or {}
        tool = external_tools.get(name)
        if tool is None:
            yield {
                "name": name,
                "args": args,
                "status": "error",
                "error": "tool not found",
            }
            continue
        try:
            result = await tool.ainvoke(args)
            text_result = _extract_text(result)
            yield {
                "name": name,
                "args": args,
                "status": "success",
                "result": _truncate_preview(text_result),
            }
            logger.info(f"自主调用 {name} 成功 (result_len={len(text_result)})")
        except Exception as e:
            logger.error(f"自主调用 {name} 失败: {e}", exc_info=True)
            yield {
                "name": name,
                "args": args,
                "status": "error",
                "error": str(e),
            }


async def _call_mermaid_with_context(query: str, retrieved_docs: list) -> str:
    """直接调用 generate_mermaid 工具，用 query + RAG 上下文作为 description。

    比等正文生成完再调省一整个 LLM 流的延迟，让图表能跟正文几乎同时到达。
    """
    tools = await _get_mcp_tools()
    tool = tools.get("generate_mermaid")
    if tool is None:
        return ""
    description = (
        f"用户原始问题：{query}\n\n"
        f"基于以下知识内容生成丰富、详细的图表（覆盖所有关键概念、子步骤、分支与关系）：\n\n"
        f"{format_context(retrieved_docs)}"
    )
    diagram_type = _infer_diagram_type(query)
    logger.info(
        f"并行调用 generate_mermaid (diagram_type={diagram_type}, desc_len={len(description)})"
    )
    result = await tool.ainvoke({"description": description, "diagram_type": diagram_type})
    return _extract_text(result)


# ===== 会话管理 =====

@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(db: AsyncSession = Depends(get_db)) -> SessionListResponse:
    """列出所有会话，按最后活动时间倒序。

    单条 SQL 同时获取：每个 session 的最后一条消息（DISTINCT ON）+ 消息总数（窗口函数），
    避免按 session 数量循环发 SQL。
    """
    stmt = sql_text(
        """
        SELECT s.session_id, s.role, s.content, s.updated_at, s.message_count,
               sess.title AS title
        FROM (
            SELECT DISTINCT ON (session_id)
                session_id,
                role,
                content,
                created_at AS updated_at,
                COUNT(*) OVER (PARTITION BY session_id) AS message_count
            FROM conversations
            ORDER BY session_id, created_at DESC
        ) s
        LEFT JOIN sessions sess ON sess.session_id = s.session_id
        ORDER BY s.updated_at DESC
        """
    )
    result = await db.execute(stmt)
    rows = result.all()

    sessions: list[SessionListItem] = [
        SessionListItem(
            session_id=row.session_id,
            title=row.title,
            last_message=row.content[:80] + ("…" if len(row.content) > 80 else ""),
            last_role=row.role,
            message_count=row.message_count,
            updated_at=row.updated_at.isoformat(),
        )
        for row in rows
    ]
    return SessionListResponse(sessions=sessions)


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str, db: AsyncSession = Depends(get_db)
) -> SessionMessagesResponse:
    """获取指定会话的全部消息，按时间正序。"""
    stmt = (
        select(Conversation)
        .where(Conversation.session_id == session_id)
        .order_by(Conversation.created_at.asc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    if not rows:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages: list[MessageItem] = []
    for row in rows:
        try:
            cites = json.loads(row.citations_json) if row.citations_json else []
        except json.JSONDecodeError:
            cites = []
        try:
            invocations = (
                json.loads(row.tool_invocations_json)
                if row.tool_invocations_json
                else []
            )
        except json.JSONDecodeError:
            invocations = []
        messages.append(
            MessageItem(
                id=row.id,
                role=row.role,
                content=row.content,
                citations=cites,
                mermaid_code=row.mermaid_code,
                tool_invocations=invocations,
                created_at=row.created_at.isoformat(),
            )
        )
    return SessionMessagesResponse(session_id=session_id, messages=messages)


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(
    session_id: str, db: AsyncSession = Depends(get_db)
) -> SessionDeleteResponse:
    """删除指定会话及其关联的反馈记录。"""
    # 先查会话所有 conversation_id
    conv_stmt = select(Conversation.id).where(Conversation.session_id == session_id)
    conv_ids = [row[0] for row in (await db.execute(conv_stmt)).all()]

    if not conv_ids:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 删除关联反馈
    await db.execute(delete(Feedback).where(Feedback.conversation_id.in_(conv_ids)))
    # 删除会话消息
    deleted = await db.execute(
        delete(Conversation).where(Conversation.session_id == session_id)
    )
    # 删除会话元数据（若有）
    await db.execute(delete(SessionModel).where(SessionModel.session_id == session_id))
    return SessionDeleteResponse(
        message=f"会话 {session_id} 已删除",
        deleted_count=deleted.rowcount or len(conv_ids),
    )
