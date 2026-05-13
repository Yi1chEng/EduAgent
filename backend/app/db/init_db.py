"""数据库初始化：启用 pgvector 扩展并创建所有表。"""

import asyncio
import logging

from sqlalchemy import text

from app.db.database import engine
from app.models.db_models import Base

logger = logging.getLogger(__name__)


async def init_database() -> None:
    """执行数据库初始化：启用 pgvector 扩展，创建所有 ORM 表与必要索引。"""
    async with engine.begin() as conn:
        # 启用 pgvector 扩展
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector 扩展已启用")

        # 创建所有表
        await conn.run_sync(Base.metadata.create_all)
        logger.info("数据库表已创建")

        # 轻量"迁移"：增量列补齐（暂未接入 Alembic，用 IF NOT EXISTS 幂等加列）
        # 见 backend/app/models/db_models.py::Conversation
        await conn.execute(text(
            "ALTER TABLE conversations "
            "ADD COLUMN IF NOT EXISTS mermaid_code TEXT, "
            "ADD COLUMN IF NOT EXISTS tool_invocations_json TEXT"
        ))
        logger.info("conversations 表列补齐 (mermaid_code, tool_invocations_json)")

        # HNSW 向量索引：cosine 距离，O(log N) 检索，替代原全表扫描
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding_hnsw "
            "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"
        ))
        # 会话历史复合索引：覆盖 _load_history / list_sessions 的常见排序
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_conversations_session_created "
            "ON conversations (session_id, created_at DESC)"
        ))
        logger.info("向量索引与会话复合索引已就绪")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(init_database())
