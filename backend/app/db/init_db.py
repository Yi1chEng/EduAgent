"""数据库初始化：启用 pgvector 扩展并创建所有表。"""

import asyncio
import logging

from sqlalchemy import text

from app.db.database import engine
from app.models.db_models import Base

logger = logging.getLogger(__name__)


async def init_database() -> None:
    """执行数据库初始化：启用 pgvector 扩展，创建所有 ORM 表。"""
    async with engine.begin() as conn:
        # 启用 pgvector 扩展
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector 扩展已启用")

        # 创建所有表
        await conn.run_sync(Base.metadata.create_all)
        logger.info("数据库表已创建")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(init_database())
