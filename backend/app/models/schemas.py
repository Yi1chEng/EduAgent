"""Pydantic 请求/响应模型定义。"""

from typing import Any, Optional

from pydantic import BaseModel, Field


# ========== 对话相关 ==========

class ChatRequest(BaseModel):
    """对话请求模型。

    enabled_tools 由前端传入，键为工具 id（与 TOOLS_REGISTRY 对齐），值为是否启用。
    缺失的键回退到该工具在 registry 中的 default_enabled。
    """
    query: str = Field(..., description="用户当前问题")
    session_id: str = Field(..., description="会话ID，用于关联上下文")
    enabled_tools: dict[str, bool] = Field(
        default_factory=dict,
        description="本轮对话中启用的工具集合，键 = 工具 id，值 = 是否启用",
    )


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
    title: Optional[str] = Field(default=None, description="LLM 自动生成的会话标题")
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
    mermaid_code: Optional[str] = Field(default=None, description="Mermaid DSL 代码")
    tool_invocations: list[dict[str, Any]] = Field(
        default_factory=list, description="工具调用记录列表"
    )
    created_at: str


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: list[MessageItem]


class SessionDeleteResponse(BaseModel):
    message: str
    deleted_count: int


class StreamEvent(BaseModel):
    """SSE 流式事件。"""
    event: str = Field(..., description="事件类型：token / citation / mermaid / tool_result / done / error")
    data: str = Field(..., description="事件数据")


# ========== 工具注册表 ==========

class ToolConfig(BaseModel):
    """单个工具的配置项，用于 GET /api/tools 响应。"""
    id: str = Field(..., description="工具 id（与 LLM tool name 一致）")
    display_name: str = Field(..., description="UI 展示名称")
    description: str = Field(..., description="工具用途说明")
    category: str = Field(..., description="internal / external")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH")
    default_enabled: bool = Field(..., description="是否默认启用")
    mcp_server: str = Field(..., description="提供该工具的 MCP server 名称")


class ToolsListResponse(BaseModel):
    """GET /api/tools 响应。"""
    tools: list[ToolConfig]


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
