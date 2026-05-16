"""关键词优先 + 向量重排 + 可选 reranker。

- 关键词层：jieba 分词 → SQL ILIKE AND 前置过滤，缩小搜索空间
- 向量层：在关键词过滤后的候选集内，pgvector cosine 距离排序
- 可选 reranker：若 RERANKER_API_KEY 已配置，调用交叉 reranker 二次排序
- 回退：若 N 个关键词 AND 过滤后候选不足，逐级减关键词；最终回退到全量向量搜索

评估接入：retrieve_detailed() 返回各阶段中间结果。
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import jieba
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.database import async_session_factory
from app.rag.embeddings import embed_text

# 领域词汇，防止 jieba 误切
_DOMAIN_TERMS = [
    "智能体", "反应式", "规划式", "混合式", "学习型",
    "上下文工程", "迁移学习", "模型压缩", "强化学习",
    "卷积神经网络", "循环神经网络", "目标检测", "图像分割",
    "知识图谱", "向量检索", "混合检索", "大语言模型",
    "超参数调整", "反向传播", "损失函数", "激活函数",
    "过拟合", "欠拟合", "正则化", "归一化", "批标准化",
]
for _t in _DOMAIN_TERMS:
    jieba.add_word(_t)


@dataclass(frozen=True)
class RetrievalConfig:
    """检索配置。"""

    keyword_filter_count: int = 5
    use_reranker: bool = True
    use_chapter_filter: bool = True


DEFAULT_CONFIG = RetrievalConfig()

logger = logging.getLogger(__name__)

# 中文章节匹配（第一章/第1章/第十章 等）
_CHAPTER_PATTERN = re.compile(r"第\s*([一二三四五六七八九十百零\d]+)\s*章")

# 不参与关键词召回的停用片段
_KEYWORD_STOPWORDS = frozenset({
    "什么", "怎么", "如何", "为什么", "请", "请问", "你能", "帮我", "告诉", "解释",
    "what", "how", "why", "the", "and", "for", "with", "please", "tell",
    "可以", "是否", "哪些", "哪个", "这是", "这个", "那个", "一下", "有没有",
    "两个", "四个", "几个", "不是", "还是", "常用", "方法", "概念",
})


def _detect_chapter_filter(query: str) -> Optional[str]:
    """从查询中识别章节关键词，返回 SQL LIKE 模式。"""
    match = _CHAPTER_PATTERN.search(query)
    if match:
        chapter_str = match.group(0).replace(" ", "")
        return f"%{chapter_str}%"
    return None


def _extract_keywords(query: str) -> list[str]:
    """用 jieba 分词提取查询中的实义词（过滤停用词和单字）。"""
    words = jieba.cut(query)
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        w = w.strip()
        if len(w) < 2:
            continue
        low = w.lower()
        if low in _KEYWORD_STOPWORDS:
            continue
        if low in seen:
            continue
        seen.add(low)
        result.append(w)
    return result


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
        logger.warning(f"Reranker 调用失败，回退: {e}")
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


def _build_row(doc: Any) -> dict[str, Any]:
    return {
        "id": doc.id,
        "text": doc.original_text,
        "source_file": doc.source_file,
        "heading_path": doc.heading_path,
        "chunk_index": doc.chunk_index,
        "score": 1.0 - float(doc.distance),
    }


async def _vector_search(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int,
    chapter_filter: str | None = None,
    keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    """向量搜索，可选章节过滤和关键词 OR + 命中数排序。"""
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
    params: dict[str, Any] = {
        "query_embedding": embedding_str,
        "top_k": top_k,
    }

    clauses: list[str] = []
    score_parts: list[str] = []

    if chapter_filter:
        clauses.append("source_file LIKE :chapter_filter")
        params["chapter_filter"] = chapter_filter

    if keywords:
        for i, kw in enumerate(keywords):
            pname = f"kw{i}"
            params[pname] = f"%{kw}%"
            clauses.append(f"original_text ILIKE :{pname}")
            score_parts.append(
                f"CASE WHEN original_text ILIKE :{pname} THEN 1 ELSE 0 END"
            )

    if score_parts:
        kw_score = " + ".join(score_parts)
        order_by = f"({kw_score}) DESC, embedding <=> :query_embedding"
    else:
        order_by = "embedding <=> :query_embedding"

    where = " OR ".join(clauses) if clauses else "TRUE"

    sql = text(f"""
        SELECT
            id, source_file, heading_path, chunk_index, original_text,
            embedding <=> :query_embedding AS distance
        FROM knowledge_chunks
        WHERE {where}
        ORDER BY {order_by}
        LIMIT :top_k
    """)

    result = await session.execute(sql, params)
    return [_build_row(row) for row in result.fetchall()]


async def retrieve_detailed(
    query: str,
    top_k: int = 5,
    config: RetrievalConfig = DEFAULT_CONFIG,
    db: AsyncSession | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """关键词优先 + 向量重排检索。

    流程：jieba 分词 → SQL ILIKE AND 前置过滤 → 向量排序 → 可选 reranker

    返回 dict 的 key:
        keyword_filtered : 关键词过滤 + 向量排序的结果（主要结果）
        fallback          : 回退到全量向量搜索的结果（仅当关键词过滤不足时非空）
        reranked          : reranker 二次排序后
        final             : 最终 top_k
    """
    keywords = _extract_keywords(query)[:config.keyword_filter_count]
    chapter_filter = _detect_chapter_filter(query) if config.use_chapter_filter else None
    query_embedding = await embed_text(query)

    async def _search(session: AsyncSession) -> list[dict[str, Any]]:
        # 逐级尝试：全部关键词 → 减 1 → ... → 1 → 回退全量向量
        min_results = max(3, top_k // 3)
        for n in range(len(keywords), 0, -1):
            subset = keywords[:n]
            result = await _vector_search(
                session, query_embedding, top_k, chapter_filter, subset
            )
            if len(result) >= min_results:
                if n < len(keywords):
                    logger.info(
                        f"关键词从 {len(keywords)} 降为 {n} 后命中 {len(result)} 条"
                    )
                return result

        logger.info(f"关键词过滤候选不足，回退到全量向量搜索")
        return await _vector_search(
            session, query_embedding, top_k, chapter_filter, keywords=None
        )

    if db is not None:
        candidates = await _search(db)
    else:
        async with async_session_factory() as session:
            candidates = await _search(session)

    empty: list[dict[str, Any]] = []
    if not candidates:
        return {"keyword_filtered": empty, "fallback": empty, "reranked": empty, "final": empty}

    # Reranker
    pre_rerank = candidates[: min(len(candidates), top_k * 2)]
    if config.use_reranker:
        reranked = await _rerank(query, pre_rerank)
    else:
        reranked = pre_rerank

    final = reranked[:top_k]
    logger.info(
        f"检索(query={query[:30]}...): keywords={keywords} "
        f"候选={len(candidates)} → rerank={len(reranked)} → final={len(final)}"
    )
    return {
        "keyword_filtered": candidates,
        "fallback": empty,
        "reranked": reranked,
        "final": final,
    }


async def retrieve(
    query: str,
    top_k: int = 5,
    db: AsyncSession | None = None,
    config: RetrievalConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    """关键词优先检索。

    Args:
        query: 用户查询文本。
        top_k: 最终返回数量。
        db: 可选的数据库会话。
        config: 检索配置。

    Returns:
        检索结果列表，每个元素包含 text, source_file, heading_path, chunk_index, score。
    """
    stages = await retrieve_detailed(query, top_k=top_k, config=config, db=db)
    return stages["final"]
