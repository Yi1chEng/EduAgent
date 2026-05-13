"""向量检索 + 关键词召回 + RRF 融合 + 可选 reranker。

- 向量层：pgvector cosine 距离，over-fetch top_k * RAG_OVERFETCH_MULTIPLIER
- 关键词层：在 over-fetched 候选内，用查询中的关键词命中数排序
- 融合：Reciprocal Rank Fusion（RRF, k=RAG_RRF_K）
- 可选 reranker：若 RERANKER_API_KEY 已配置，调用 SiliconFlow bge-reranker 二次排序

评估接入：retrieve_detailed() 返回各阶段中间结果，配合 RetrievalConfig 可做 ablation 实验。
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.database import async_session_factory
from app.rag.embeddings import embed_text


@dataclass(frozen=True)
class RetrievalConfig:
    """检索 ablation 配置：关闭某层即模拟单一组件的检索质量。"""

    use_keyword: bool = True
    use_rrf: bool = True
    use_reranker: bool = True
    use_chapter_filter: bool = True
    overfetch_multiplier: Optional[int] = None  # None 则用 settings.RAG_OVERFETCH_MULTIPLIER
    rrf_k: Optional[int] = None  # None 则用 settings.RAG_RRF_K


DEFAULT_CONFIG = RetrievalConfig()

logger = logging.getLogger(__name__)

# 中文章节匹配（第一章/第1章/第十章 等）
_CHAPTER_PATTERN = re.compile(r"第\s*([一二三四五六七八九十百零\d]+)\s*章")

# 关键词抽取：连续 CJK 字符段（≥2）或 ASCII 单词（≥2）
_KEYWORD_PATTERN = re.compile(r"[一-鿿]{2,}|[A-Za-z0-9_]{2,}")

# 不参与关键词召回的停用片段
_KEYWORD_STOPWORDS = frozenset({
    "什么", "怎么", "如何", "为什么", "请", "请问", "你能", "帮我", "告诉", "解释",
    "what", "how", "why", "the", "and", "for", "with", "please", "tell",
})


def _detect_chapter_filter(query: str) -> Optional[str]:
    """从查询中识别章节关键词，返回 SQL LIKE 模式。"""
    match = _CHAPTER_PATTERN.search(query)
    if match:
        chapter_str = match.group(0).replace(" ", "")
        return f"%{chapter_str}%"
    return None


def _extract_keywords(query: str) -> list[str]:
    """提取查询中可用于关键词召回的实义词片段。"""
    tokens = _KEYWORD_PATTERN.findall(query)
    seen: set[str] = set()
    result: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in _KEYWORD_STOPWORDS:
            continue
        if low in seen:
            continue
        seen.add(low)
        result.append(tok)
    return result


def _keyword_score(text_body: str, keywords: list[str]) -> int:
    """文档对查询关键词的命中分（按出现次数累加）。"""
    if not keywords:
        return 0
    lower = text_body.lower()
    score = 0
    for kw in keywords:
        score += lower.count(kw.lower())
    return score


def _rrf_fuse(
    vector_ranked: list[dict[str, Any]],
    keyword_ranked: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion：score = sum(1 / (k + rank))。

    依靠 id 作为去重键。
    """
    scores: dict[Any, float] = {}
    by_id: dict[Any, dict[str, Any]] = {}

    for rank, doc in enumerate(vector_ranked, start=1):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        by_id[doc_id] = doc

    for rank, doc in enumerate(keyword_ranked, start=1):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        by_id.setdefault(doc_id, doc)

    fused = [
        {**by_id[doc_id], "rrf_score": score}
        for doc_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return fused


async def _rerank(query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """调用外部 reranker API（如未配置则原样返回）。"""
    settings = get_settings()
    if not settings.RERANKER_API_KEY or not docs:
        return docs

    payload = {
        "model": settings.RERANKER_MODEL_NAME,
        "query": query,
        "documents": [d["text"] for d in docs],
        "return_documents": False,
    }
    headers = {"Authorization": f"Bearer {settings.RERANKER_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                settings.RERANKER_BASE_URL, json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"Reranker 调用失败，回退到 RRF 顺序: {e}")
        return docs

    results = data.get("results") or []
    if not results:
        return docs

    reranked: list[dict[str, Any]] = []
    for item in results:
        idx = item.get("index")
        score = item.get("relevance_score") or item.get("score") or 0.0
        if idx is None or idx >= len(docs):
            continue
        reranked.append({**docs[idx], "rerank_score": float(score)})
    return reranked


async def retrieve_detailed(
    query: str,
    top_k: int = 5,
    config: RetrievalConfig = DEFAULT_CONFIG,
    db: AsyncSession | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """与 retrieve() 等价，但返回各阶段中间结果用于评估/调试。

    返回 dict 的 key:
        vector    : 向量层召回结果（按 cosine 距离升序，等价于"vector-only top-k"）
        keyword   : 关键词层重排结果（仅在候选集内，0 分文档已过滤）
        fused     : RRF 融合后的列表
        reranked  : 交叉 reranker 二次排序后的列表（未启用 reranker 时 == fused）
        final     : 截断到 top_k 后的最终结果（等价于 retrieve() 的返回）
    """
    settings = get_settings()
    multiplier = config.overfetch_multiplier or settings.RAG_OVERFETCH_MULTIPLIER
    rrf_k = config.rrf_k or settings.RAG_RRF_K
    overfetch = max(top_k, top_k * multiplier)

    query_embedding = await embed_text(query)
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    chapter_filter = _detect_chapter_filter(query) if config.use_chapter_filter else None
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
            LIMIT :overfetch
        """)
        params = {
            "query_embedding": embedding_str,
            "overfetch": overfetch,
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
            LIMIT :overfetch
        """)
        params = {"query_embedding": embedding_str, "overfetch": overfetch}

    async def _execute(session: AsyncSession) -> list[dict[str, Any]]:
        result = await session.execute(sql, params)
        rows = result.fetchall()
        return [
            {
                "id": row.id,
                "text": row.original_text,
                "source_file": row.source_file,
                "heading_path": row.heading_path,
                "chunk_index": row.chunk_index,
                "score": 1.0 - float(row.distance),  # cosine similarity
            }
            for row in rows
        ]

    if db is not None:
        candidates = await _execute(db)
    else:
        async with async_session_factory() as session:
            candidates = await _execute(session)

    if not candidates:
        empty: list[dict[str, Any]] = []
        return {"vector": empty, "keyword": empty, "fused": empty, "reranked": empty, "final": empty}

    # 关键词层
    if config.use_keyword:
        keywords = _extract_keywords(query)
        if keywords:
            scored = [
                (_keyword_score(c["text"], keywords), c) for c in candidates
            ]
            keyword_ranked = [
                c for s, c in sorted(scored, key=lambda x: x[0], reverse=True) if s > 0
            ]
        else:
            keyword_ranked = []
    else:
        keyword_ranked = []

    # RRF 融合（关闭时直接退化为向量序）
    if config.use_rrf and keyword_ranked:
        fused = _rrf_fuse(candidates, keyword_ranked, k=rrf_k)
    else:
        fused = list(candidates)

    # Reranker（关闭或未配置时透传 fused）
    pre_rerank = fused[: max(top_k * 2, top_k)]
    if config.use_reranker:
        reranked = await _rerank(query, pre_rerank)
    else:
        reranked = pre_rerank

    final = reranked[:top_k]
    logger.info(
        f"混合检索({config}): 候选 {len(candidates)} → 融合 {len(fused)} → 重排 {len(reranked)} → 返回 {len(final)}"
    )
    return {
        "vector": candidates,
        "keyword": keyword_ranked,
        "fused": fused,
        "reranked": reranked,
        "final": final,
    }


async def retrieve(
    query: str,
    top_k: int = 5,
    db: AsyncSession | None = None,
    config: RetrievalConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    """混合检索：向量召回 + 关键词召回 → RRF 融合 → 可选 reranker。

    Args:
        query: 用户查询文本。
        top_k: 最终返回数量。
        db: 可选的数据库会话；未传入时自动创建。
        config: 检索 ablation 配置（默认启用所有层）。

    Returns:
        检索结果列表，每个元素包含 text, source_file, heading_path, chunk_index, score。
    """
    stages = await retrieve_detailed(query, top_k=top_k, config=config, db=db)
    return stages["final"]
