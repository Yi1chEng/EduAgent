"""SQLAlchemy 2.0 ORM 模型定义：知识库表、对话表、反馈表。"""

from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """ORM 基类。"""
    pass


class KnowledgeChunk(Base):
    """知识库文本块表，存储教材切分后的文本段落及其向量。"""

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_file: Mapped[str] = mapped_column(String(512), nullable=False, index=True, comment="来源文件名")
    heading_path: Mapped[str] = mapped_column(String(1024), nullable=False, comment="标题路径，如 '第三章>3.2 牛顿第二定律'")
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="该文件内的段落序号")
    original_text: Mapped[str] = mapped_column(Text, nullable=False, comment="原始文本内容")
    embedding = mapped_column(Vector(1024), nullable=True, comment="文本向量（1024维，匹配BAAI/bge-m3模型）")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间"
    )

    def __repr__(self) -> str:
        return f"<KnowledgeChunk(id={self.id}, source='{self.source_file}', heading='{self.heading_path}')>"


class Session(Base):
    """会话元数据表：保存自动生成的标题等不随消息变化的字段。"""

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True, comment="会话ID")
    title: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment="LLM 自动生成的会话标题")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="最近更新时间",
    )

    def __repr__(self) -> str:
        return f"<Session(session_id='{self.session_id}', title='{self.title}')>"


class Conversation(Base):
    """对话记录表，保存用户与AI的对话历史。"""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True, comment="会话ID")
    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="角色：user 或 assistant")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="消息内容")
    citations_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="引用来源JSON")
    mermaid_code: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="本条回答附带的 Mermaid DSL（仅 assistant 消息可能有）"
    )
    tool_invocations_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="本条回答触发的工具调用列表 JSON：[{name, args, status, result?, error?}]",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间"
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, session='{self.session_id}', role='{self.role}')>"


class Feedback(Base):
    """用户反馈表，用于收集反馈数据以支持后续微调。"""

    __tablename__ = "feedbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True, comment="关联的对话ID")
    rating: Mapped[int] = mapped_column(Integer, nullable=False, comment="评分：1（好）或 -1（差）")
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="用户评论")
    prompt: Mapped[str] = mapped_column(Text, nullable=False, comment="对应的用户提问")
    response: Mapped[str] = mapped_column(Text, nullable=False, comment="对应的AI回答")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间"
    )

    def __repr__(self) -> str:
        return f"<Feedback(id={self.id}, conversation_id={self.conversation_id}, rating={self.rating})>"
