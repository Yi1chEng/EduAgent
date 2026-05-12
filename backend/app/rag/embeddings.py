"""文本向量化模块，基于 LangChain OpenAIEmbeddings。"""

import time
from collections import OrderedDict
from typing import List, Optional, Tuple

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings

_embeddings_model: Optional[OpenAIEmbeddings] = None

# 简易 TTL+LRU 缓存：仅用于 embed_text（单 query 向量化），避免重复外部 API 调用。
# 不引入额外依赖，单进程使用足够；多 worker 部署时建议改用 Redis。
_EMBED_CACHE_MAXSIZE: int = 512
_EMBED_CACHE_TTL_SECONDS: float = 600.0
_embed_cache: "OrderedDict[str, Tuple[float, List[float]]]" = OrderedDict()


def _cache_get(key: str) -> Optional[List[float]]:
    item = _embed_cache.get(key)
    if item is None:
        return None
    ts, vec = item
    if time.monotonic() - ts > _EMBED_CACHE_TTL_SECONDS:
        _embed_cache.pop(key, None)
        return None
    _embed_cache.move_to_end(key)
    return vec


def _cache_put(key: str, value: List[float]) -> None:
    _embed_cache[key] = (time.monotonic(), value)
    _embed_cache.move_to_end(key)
    while len(_embed_cache) > _EMBED_CACHE_MAXSIZE:
        _embed_cache.popitem(last=False)


def _get_embeddings_model() -> OpenAIEmbeddings:
    """延迟初始化 Embeddings 模型，确保配置已完整加载。"""
    global _embeddings_model
    if _embeddings_model is None:
        settings = get_settings()
        _embeddings_model = OpenAIEmbeddings(
            openai_api_key=settings.EMBEDDING_API_KEY,
            openai_api_base=settings.EMBEDDING_BASE_URL,
            model=settings.EMBEDDING_MODEL_NAME,
        )
    return _embeddings_model


async def embed_text(text: str) -> List[float]:
    """将单条文本向量化（带 TTL 缓存，重复 query 直接命中）。

    Args:
        text: 待向量化的文本字符串。

    Returns:
        1024 维浮点向量列表。
    """
    cached = _cache_get(text)
    if cached is not None:
        return cached
    result = await _get_embeddings_model().aembed_query(text)
    _cache_put(text, result)
    return result


async def embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """批量文本向量化，自动分批以避免超过 API 限制。

    Args:
        texts: 待向量化的文本列表。
        batch_size: 每批最大文本数量，默认 32。

    Returns:
        向量列表，每个元素为 1024 维浮点向量。
    """
    model = _get_embeddings_model()
    all_results: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = await model.aembed_documents(batch)
        all_results.extend(results)
    return all_results
