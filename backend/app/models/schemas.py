"""Pydantic 请求/响应模型定义。"""

from typing import Any, Optional

from pydantic import BaseModel, Field


# ========== 对话相关 ==========

class ChatRequest(BaseModel):
    """对话请求模型。"""
    query: str = Field(..., description="用户当前问题")
    session_id: str = Field(..., description="会话ID，用于关联上下文")
    need_visualization: bool = Field(default=False, description="是否需要生成图表")
    need_dispatch: bool = Field(default=False, description="是否需要推送消息")


class ChatResponse(BaseModel):
    """对话响应模型。"""
    conversation_id: int = Field(..., description="本次 assistant 消息的对话记录 ID（用于反馈）")
    content: str = Field(..., description="AI 生成的回答内容")
    citations: list[dict[str, Any]] = Field(default_factory=list, description="引用来源列表")
    mermaid_code: Optional[str] = Field(default=None, description="Mermaid 图表 DSL 代码")
    tool_results: Optional[dict[str, Any]] = Field(default=None, description="工具调用结果")


# ========== 会话管理 ==========

class SessionListItem(BaseModel):
    """会话列表项。"""
    session_id: str
    last_message: str = Field(..., description="最后一条消息预览")
    last_role: str = Field(..., description="最后一条消息角色")
    message_count: int
    updated_at: str


class SessionListResponse(BaseModel):
    sessions: list[SessionListItem]


class MessageItem(BaseModel):
    """单条历史消息。"""
    id: int
    role: str
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: list[MessageItem]


class SessionDeleteResponse(BaseModel):
    message: str
    deleted_count: int


class StreamEvent(BaseModel):
    """SSE 流式事件。"""
    event: str = Field(..., description="事件类型：token / citation / mermaid / done / error")
    data: str = Field(..., description="事件数据")


# ========== 知识库相关 ==========

class KnowledgeUploadResponse(BaseModel):
    """知识库上传响应。"""
    message: str
    source_file: str
    chunks_count: int


class KnowledgeListItem(BaseModel):
    """知识库文档列表项。"""
    source_file: str
    chunks_count: int
    created_at: str


class KnowledgeListResponse(BaseModel):
    """知识库文档列表响应。"""
    documents: list[KnowledgeListItem]


class KnowledgeDeleteResponse(BaseModel):
    """知识库删除响应。"""
    message: str
    deleted_count: int


# ========== 反馈相关 ==========

class FeedbackRequest(BaseModel):
    """用户反馈请求模型。"""
    conversation_id: int = Field(..., description="关联的对话ID（assistant 消息的 ID）")
    rating: int = Field(..., description="评分：1（好）或 -1（差）")
    comment: Optional[str] = Field(default=None, description="用户评论")


class FeedbackResponse(BaseModel):
    """反馈提交响应。"""
    message: str
    feedback_id: int
