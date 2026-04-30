"""向量检索模块：基于 pgvector 的 cosine 距离检索。"""

import logging
import re
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import async_session_factory
from app.rag.embeddings import embed_text

logger = logging.getLogger(__name__)

# 中文章节匹配（第一章/第1章/第十章 等）
_CHAPTER_PATTERN = re.compile(r"第\s*([一二三四五六七八九十百零\d]+)\s*章")


def _detect_chapter_filter(query: str) -> Optional[str]:
    """从查询中识别章节关键词，返回 SQL LIKE 模式。"""
    match = _CHAPTER_PATTERN.search(query)
    if match:
        chapter_str = match.group(0).replace(" ", "")
        return f"%{chapter_str}%"
    return None


async def retrieve(
    query: str,
    top_k: int = 5,
    db: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """根据用户查询进行向量检索，返回最相似的 top_k 个文本块。

    若查询中包含明确章节信息（如"第一章"），则优先在该章节内检索。

    Args:
        query: 用户查询文本。
        top_k: 返回的最大结果数。
        db: 可选的数据库会话；未传入时自动创建。

    Returns:
        检索结果列表，每个元素包含 text, source_file, heading_path, chunk_index, score。
    """
    query_embedding = await embed_text(query)
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    chapter_filter = _detect_chapter_filter(query)
    if chapter_filter:
        logger.info(f"检测到章节过滤条件: {chapter_filter}")
        sql = text("""
            SELECT
                id,
                source_file,
                heading_path,
                chunk_index,
                original_text,
                embedding <=> :query_embedding AS distance
            FROM knowledge_chunks
            WHERE source_file LIKE :chapter_filter
            ORDER BY embedding <=> :query_embedding
            LIMIT :top_k
        """)
        params = {
            "query_embedding": embedding_str,
            "top_k": top_k,
            "chapter_filter": chapter_filter,
        }
    else:
        sql = text("""
            SELECT
                id,
                source_file,
                heading_path,
                chunk_index,
                original_text,
                embedding <=> :query_embedding AS distance
            FROM knowledge_chunks
            ORDER BY embedding <=> :query_embedding
            LIMIT :top_k
        """)
        params = {"query_embedding": embedding_str, "top_k": top_k}

    async def _execute(session: AsyncSession) -> list[dict[str, Any]]:
        result = await session.execute(sql, params)
        rows = result.fetchall()
        docs: list[dict[str, Any]] = []
        for row in rows:
            docs.append({
                "text": row.original_text,
                "source_file": row.source_file,
                "heading_path": row.heading_path,
                "chunk_index": row.chunk_index,
                "score": 1.0 - float(row.distance),  # cosine similarity
            })
        return docs

    if db is not None:
        return await _execute(db)
    else:
        async with async_session_factory() as session:
            return await _execute(session)
