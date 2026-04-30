"""用户反馈接口：收集反馈数据用于后续微调。"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.db_models import Conversation, Feedback
from app.models.schemas import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    """提交用户反馈，关联对应的 prompt-response 对后存入 Feedback 表。

    Args:
        request: 反馈请求，包含 conversation_id, rating, comment。
        db: 异步数据库会话。

    Returns:
        反馈提交结果。
    """
    # 验证 rating 值
    if request.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating 必须为 1（好）或 -1（差）")

    # 查找 assistant 消息
    assistant_stmt = select(Conversation).where(
        Conversation.id == request.conversation_id,
        Conversation.role == "assistant",
    )
    result = await db.execute(assistant_stmt)
    assistant_msg = result.scalar_one_or_none()

    if not assistant_msg:
        raise HTTPException(
            status_code=404,
            detail=f"未找到 ID 为 {request.conversation_id} 的 assistant 消息",
        )

    # 查找对应的 user 消息（同 session，时间上最近的一条在 assistant 之前的）
    user_stmt = (
        select(Conversation)
        .where(
            Conversation.session_id == assistant_msg.session_id,
            Conversation.role == "user",
            Conversation.created_at <= assistant_msg.created_at,
        )
        .order_by(Conversation.created_at.desc())
        .limit(1)
    )
    user_result = await db.execute(user_stmt)
    user_msg = user_result.scalar_one_or_none()

    prompt = user_msg.content if user_msg else ""

    # 创建反馈记录
    feedback = Feedback(
        conversation_id=request.conversation_id,
        rating=request.rating,
        comment=request.comment,
        prompt=prompt,
        response=assistant_msg.content,
    )
    db.add(feedback)
    await db.flush()

    logger.info(
        f"反馈已提交: feedback_id={feedback.id}, "
        f"conversation_id={request.conversation_id}, rating={request.rating}"
    )

    return FeedbackResponse(
        message="反馈提交成功",
        feedback_id=feedback.id,
    )
