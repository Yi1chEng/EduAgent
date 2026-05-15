"""对话与会话管理接口：同步对话 / SSE 流式 / 工具列表 / 会话列表 / 历史 / 删除。"""

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy import delete, select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.db.database import async_session_factory, get_db
from app.graph.builder import app_graph
from app.graph.nodes import (
    _extract_text,
    _get_llm,
    _get_mcp_tools,
    build_system_prompt,
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
    ToolConfig,
    ToolsListResponse,
)
from app.rag.citation import build_citation_list
from app.rag.retriever import retrieve
from app.tools.registry import TOOLS_REGISTRY, get_available_tools

logger = logging.getLogger(__name__)

router = APIRouter()


# 明确不需要查知识库的"短语精确匹配"集合
_NO_RAG_EXACT = {
    "你好", "您好", "hi", "hello", "hey", "嗨",
    "早上好", "上午好", "下午好", "晚上好", "晚安",
    "谢谢", "多谢", "感谢", "thanks", "thank you", "ok", "okay", "好的", "收到",
    "再见", "拜拜", "bye", "goodbye",
    "测试", "test", "ping",
}

_NO_RAG_SUBSTRINGS = (
    "你是谁", "你叫什么", "你能做什么", "你有什么功能",
    "what can you do", "who are you", "your name",
)


def _needs_rag(query: str) -> bool:
    """启发式：明显的闲聊/打招呼/致谢/元问题返回 False，其它一律返回 True。"""
    q = query.strip().lower().rstrip("。.!！?？~～")
    if not q:
        return False
    if q in _NO_RAG_EXACT:
        return False
    if any(s in q for s in _NO_RAG_SUBSTRINGS):
        return False
    stripped = "".join(c for c in q if c.isalnum())
    if len(stripped) <= 2:
        return False
    return True


# SSE tool_result 事件载荷上限：避免大返回阻塞流 / 撑爆前端
_TOOL_RESULT_PREVIEW_LIMIT = 2000


def _truncate_preview(text: str, limit: int = _TOOL_RESULT_PREVIEW_LIMIT) -> str:
    """截断超长文本并附省略号，避免单个工具结果撑爆 SSE 单帧。"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…(已截断，原文 {len(text)} 字符)"


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


def _build_initial_state(request: ChatRequest, history: list[Any]) -> dict[str, Any]:
    """构建 LangGraph 初始 state。"""
    return {
        "messages": history,
        "query": request.query,
        "session_id": request.session_id,
        "enabled_tools": request.enabled_tools,
        "retrieved_docs": [],
        "citations": [],
        "generated_content": "",
        "tool_invocations": [],
        "mermaid_code": "",
    }


# ===== 工具注册表 =====

@router.get("/tools", response_model=ToolsListResponse)
async def list_tools() -> ToolsListResponse:
    """返回所有已注册工具的元数据 + 默认启用状态。前端用此渲染 ToolSelector。"""
    return ToolsListResponse(
        tools=[ToolConfig(**meta.to_dict()) for meta in TOOLS_REGISTRY.values()]
    )


# ===== 对话接口 =====

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)) -> ChatResponse:
    """同步对话接口：执行完整 LangGraph 流程，一次性返回结果。"""
    try:
        history = await _load_history(db, request.session_id)
        is_first_turn = len(history) == 0
        await _save_conversation(db, request.session_id, "user", request.query)

        result = await app_graph.ainvoke(_build_initial_state(request, history))

        content = result.get("generated_content", "")
        citations = result.get("citations", [])
        mermaid_code = result.get("mermaid_code") or None
        tool_invocations: list[dict[str, Any]] = result.get("tool_invocations") or []

        assistant_conv = await _save_conversation(
            db,
            request.session_id,
            "assistant",
            content,
            json.dumps(citations, ensure_ascii=False) if citations else None,
            mermaid_code=mermaid_code,
            tool_invocations_json=(
                json.dumps(tool_invocations, ensure_ascii=False) if tool_invocations else None
            ),
        )

        if is_first_turn:
            schedule_title_generation(request.session_id, request.query)

        return ChatResponse(
            conversation_id=assistant_conv.id,
            content=content,
            citations=citations,
            mermaid_code=mermaid_code,
            tool_results=(
                {inv["name"]: inv for inv in tool_invocations} if tool_invocations else None
            ),
        )

    except Exception as e:
        logger.error(f"对话处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"对话处理失败: {e}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> EventSourceResponse:
    """SSE 流式对话。

    流程：
    1. RAG 检索 + 用户消息落库（并行）
    2. push citation
    3. LLM bind_tools(enabled) astream 第一轮：同时累计 token 与 tool_calls
    4. 顺序执行 tool_calls，每个推一条 tool_result 事件
    5. 若有 tool_calls，再跑一次 LLM（带 ToolMessage 回灌）流式推最终正文 token
    6. 落库 → done

    事件：citation / token / tool_result / mermaid / done / error
    """
    use_rag = _needs_rag(request.query)
    if not use_rag:
        logger.info(f"启发式判定无需 RAG: {request.query!r}")

    async def event_generator():
        async with async_session_factory() as db:
            try:
                # ===== Stage 1: RAG（按需）+ DB 写入 =====
                retrieve_task = (
                    asyncio.create_task(retrieve(request.query, top_k=5)) if use_rag else None
                )
                history = await _load_history(db, request.session_id)
                is_first_turn = len(history) == 0
                await _save_conversation(db, request.session_id, "user", request.query)
                await db.commit()
                retrieved_docs = await retrieve_task if retrieve_task is not None else []
                citations = build_citation_list(retrieved_docs) if retrieved_docs else []
                yield {
                    "event": "citation",
                    "data": json.dumps(citations, ensure_ascii=False),
                }

                # ===== Stage 2: 准备 LLM + 工具 =====
                # 不论是否走 RAG，系统提示都用同一个 build_system_prompt 渲染，
                # 这样 LLM 在闲聊场景下也能完整看到工具菜单，正确回答"你有哪些工具"。
                system_prompt = build_system_prompt(retrieved_docs, request.enabled_tools)

                loaded_tools = await _get_mcp_tools()
                available = get_available_tools(loaded_tools, request.enabled_tools)
                available_by_name = {t.name: t for t in available}

                base_msgs: list[Any] = [
                    SystemMessage(content=system_prompt),
                    *history[-10:],
                    HumanMessage(content=request.query),
                ]

                # ===== Stage 3: 第一轮 LLM 流式 =====
                llm = _get_llm()
                llm_with_tools = llm.bind_tools(available) if available else llm

                first_round_text_parts: list[str] = []
                first_response_msg: Any = None  # AIMessageChunk 累加

                async for chunk in llm_with_tools.astream(base_msgs):
                    first_response_msg = (
                        chunk if first_response_msg is None else first_response_msg + chunk
                    )
                    token = getattr(chunk, "content", "")
                    if isinstance(token, list):
                        token = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in token
                        )
                    if token:
                        first_round_text_parts.append(token)
                        yield {"event": "token", "data": token}

                accumulated_tool_calls: list[dict[str, Any]] = (
                    getattr(first_response_msg, "tool_calls", []) or []
                    if first_response_msg is not None
                    else []
                )

                # ===== Stage 4: 顺序执行工具调用 =====
                collected_invocations: list[dict[str, Any]] = []
                final_mermaid_code = ""
                followup_msgs: list[Any] = (
                    [*base_msgs, first_response_msg] if accumulated_tool_calls else []
                )

                for call in accumulated_tool_calls:
                    name = call.get("name", "")
                    args = call.get("args", {}) or {}
                    call_id = call.get("id", "")
                    tool = available_by_name.get(name)
                    if tool is None:
                        invocation = {
                            "name": name,
                            "args": args,
                            "status": "error",
                            "error": "tool not enabled",
                        }
                        collected_invocations.append(invocation)
                        followup_msgs.append(
                            ToolMessage(content="tool not enabled", tool_call_id=call_id)
                        )
                        yield {
                            "event": "tool_result",
                            "data": json.dumps(invocation, ensure_ascii=False),
                        }
                        continue
                    try:
                        raw = await tool.ainvoke(args)
                        text_result = _extract_text(raw)
                        invocation = {
                            "name": name,
                            "args": args,
                            "status": "success",
                            "result": _truncate_preview(text_result),
                        }
                        if name == "generate_mermaid":
                            final_mermaid_code = text_result
                        collected_invocations.append(invocation)
                        followup_msgs.append(
                            ToolMessage(content=text_result, tool_call_id=call_id)
                        )
                        yield {
                            "event": "tool_result",
                            "data": json.dumps(invocation, ensure_ascii=False),
                        }
                    except Exception as e:
                        logger.error(f"工具 {name} 调用失败: {e}", exc_info=True)
                        invocation = {
                            "name": name,
                            "args": args,
                            "status": "error",
                            "error": str(e),
                        }
                        collected_invocations.append(invocation)
                        followup_msgs.append(
                            ToolMessage(content=f"error: {e}", tool_call_id=call_id)
                        )
                        yield {
                            "event": "tool_result",
                            "data": json.dumps(invocation, ensure_ascii=False),
                        }

                # ===== Stage 5: 若有工具调用，跑第二轮（带 ToolMessage）流式推最终正文 =====
                final_content_parts: list[str] = list(first_round_text_parts)
                if accumulated_tool_calls:
                    async for chunk in _get_llm().astream(followup_msgs):
                        token = getattr(chunk, "content", "")
                        if isinstance(token, list):
                            token = "".join(
                                part.get("text", "") if isinstance(part, dict) else str(part)
                                for part in token
                            )
                        if token:
                            final_content_parts.append(token)
                            yield {"event": "token", "data": token}

                final_content = "".join(final_content_parts)

                # ===== Stage 6: 单独 mermaid 事件（前端单字段渲染） =====
                if final_mermaid_code:
                    yield {"event": "mermaid", "data": final_mermaid_code}

                # ===== Stage 7: 持久化 + done =====
                assistant_conv = await _save_conversation(
                    db,
                    request.session_id,
                    "assistant",
                    final_content,
                    json.dumps(citations, ensure_ascii=False) if citations else None,
                    mermaid_code=final_mermaid_code or None,
                    tool_invocations_json=(
                        json.dumps(collected_invocations, ensure_ascii=False)
                        if collected_invocations
                        else None
                    ),
                )
                await db.commit()
                yield {"event": "done", "data": str(assistant_conv.id)}

                if is_first_turn:
                    schedule_title_generation(request.session_id, request.query)

            except Exception as e:
                logger.error(f"流式对话失败: {e}", exc_info=True)
                await db.rollback()
                yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())


# ===== 会话管理 =====

@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(db: AsyncSession = Depends(get_db)) -> SessionListResponse:
    """列出所有会话，按最后活动时间倒序。"""
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
    conv_stmt = select(Conversation.id).where(Conversation.session_id == session_id)
    conv_ids = [row[0] for row in (await db.execute(conv_stmt)).all()]

    if not conv_ids:
        raise HTTPException(status_code=404, detail="会话不存在")

    await db.execute(delete(Feedback).where(Feedback.conversation_id.in_(conv_ids)))
    deleted = await db.execute(
        delete(Conversation).where(Conversation.session_id == session_id)
    )
    await db.execute(delete(SessionModel).where(SessionModel.session_id == session_id))
    return SessionDeleteResponse(
        message=f"会话 {session_id} 已删除",
        deleted_count=deleted.rowcount or len(conv_ids),
    )
