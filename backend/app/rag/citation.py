"""引用链追踪与格式化模块。"""

from typing import Any


def format_context(retrieved_docs: list[dict[str, Any]]) -> str:
    """将检索结果组织为带编号的 context 字符串，供 LLM 使用。

    格式:
        [1] {chunk_text}
        [2] {chunk_text}
        ...

    Args:
        retrieved_docs: 向量检索返回的文档列表。

    Returns:
        带编号的上下文字符串。
    """
    if not retrieved_docs:
        return "未找到相关参考资料。"

    parts: list[str] = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        parts.append(f"[{idx}] {doc['text']}")
    return "\n\n".join(parts)


def format_citations(retrieved_docs: list[dict[str, Any]]) -> str:
    """生成引用来源列表，用于展示给用户。

    格式:
        [1] {source_file} > {heading_path}
        [2] {source_file} > {heading_path}
        ...

    Args:
        retrieved_docs: 向量检索返回的文档列表。

    Returns:
        引用来源列表字符串。
    """
    if not retrieved_docs:
        return ""

    parts: list[str] = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        parts.append(f"[{idx}] {doc['source_file']} > {doc['heading_path']}")
    return "\n".join(parts)


def build_citation_list(retrieved_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构建引用来源的结构化列表，用于 API 响应。

    Args:
        retrieved_docs: 向量检索返回的文档列表。

    Returns:
        引用来源字典列表。
    """
    citations: list[dict[str, Any]] = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        citations.append({
            "index": idx,
            "source_file": doc["source_file"],
            "heading_path": doc["heading_path"],
            "chunk_index": doc["chunk_index"],
            "score": doc.get("score", 0.0),
        })
    return citations
