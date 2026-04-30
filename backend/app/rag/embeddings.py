"""文本向量化模块，基于 LangChain OpenAIEmbeddings。"""

from typing import List, Optional

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings

_embeddings_model: Optional[OpenAIEmbeddings] = None


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
    """将单条文本向量化。

    Args:
        text: 待向量化的文本字符串。

    Returns:
        1536 维浮点向量列表。
    """
    result = await _get_embeddings_model().aembed_query(text)
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
