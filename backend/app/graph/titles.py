"""会话标题自动生成：在每个 session 第一轮对话完成后异步触发。"""

import asyncio
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select

from app.db.database import async_session_factory
from app.graph.nodes import _get_llm
from app.models.db_models import Session as SessionModel

logger = logging.getLogger(__name__)

_TITLE_SYSTEM_PROMPT = (
    "你是会话标题生成助手。根据用户的第一条问题，用不超过 12 个汉字（或对应英文长度）"
    "概括该对话主题。直接输出标题文本，不要任何引号、标点、解释或前缀。"
)

# 标题清洗：去除引号、句号、换行
_TITLE_CLEAN_PATTERN = re.compile(r"[\s\"'“”‘’《》<>「」【】.。!！?？:：;；]+")


def _clean_title(raw: str) -> str:
    """裁掉标题里的引号、标点、空白，并截断到合理长度。"""
    s = raw.strip().splitlines()[0] if raw else ""
    s = _TITLE_CLEAN_PATTERN.sub("", s)
    return s[:24]


async def _generate_title_text(first_query: str) -> str:
    """调用 LLM 生成标题文本（不入库）。"""
    messages = [
        SystemMessage(content=_TITLE_SYSTEM_PROMPT),
        HumanMessage(content=first_query),
    ]
    response = await _get_llm().ainvoke(messages)
    content = getattr(response, "content", "") or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return _clean_title(content)


async def ensure_session_title(session_id: str, first_query: str) -> None:
    """若该 session 还没有标题，则生成并写入。失败仅记录日志。"""
    try:
        async with async_session_factory() as db:
            existing = await db.execute(
                select(SessionModel).where(SessionModel.session_id == session_id)
            )
            row = existing.scalar_one_or_none()
            if row is not None and row.title:
                return

            title = await _generate_title_text(first_query)
            if not title:
                logger.info(f"标题生成结果为空，跳过: session={session_id}")
                return

            if row is None:
                db.add(SessionModel(session_id=session_id, title=title))
            else:
                row.title = title
            await db.commit()
            logger.info(f"会话标题已生成: session={session_id} title={title!r}")
    except Exception as e:
        logger.warning(f"生成会话标题失败 session={session_id}: {e}")


def schedule_title_generation(session_id: str, first_query: str) -> None:
    """fire-and-forget：在当前 event loop 起一个后台任务生成标题。"""
    try:
        asyncio.create_task(ensure_session_title(session_id, first_query))
    except RuntimeError:
        # 没有运行中的 event loop（极少见，例如同步上下文）→ 直接放弃
        logger.debug("无事件循环，跳过标题生成调度")
