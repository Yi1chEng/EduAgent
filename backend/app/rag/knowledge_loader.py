"""教材导入模块：Markdown 文件按标题切分并入库。"""

import logging
import os
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import KnowledgeChunk
from app.rag.embeddings import embed_batch

logger = logging.getLogger(__name__)

# 最大段落长度与 overlap 参数
MAX_CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 64


def _split_by_headings(content: str) -> list[dict[str, Any]]:
    """按一级/二级标题（# / ##）切分 Markdown 内容为语义段落。

    Args:
        content: Markdown 文件全文内容。

    Returns:
        段落列表，每个元素包含 heading_path 和 text。
    """
    lines = content.split("\n")
    chunks: list[dict[str, Any]] = []
    current_h1 = ""
    current_h2 = ""
    current_text_lines: list[str] = []

    def flush() -> None:
        text = "\n".join(current_text_lines).strip()
        if text:
            heading_parts = [p for p in [current_h1, current_h2] if p]
            heading_path = " > ".join(heading_parts) if heading_parts else "未分类"
            chunks.append({"heading_path": heading_path, "text": text})

    for line in lines:
        # 检查二级标题（必须在一级标题之前检查，因为 ## 也以 # 开头）
        if re.match(r"^##\s+", line):
            flush()
            current_h2 = re.sub(r"^##\s+", "", line).strip()
            current_text_lines = []
        elif re.match(r"^#\s+", line):
            flush()
            current_h1 = re.sub(r"^#\s+", "", line).strip()
            current_h2 = ""
            current_text_lines = []
        else:
            current_text_lines.append(line)

    flush()
    return chunks


def _secondary_split(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """对超过 max_size 字符的段落按段落换行二次切分。

    Args:
        text: 需要切分的文本。
        max_size: 单个 chunk 最大字符数。
        overlap: 相邻 chunk 之间的重叠字符数。

    Returns:
        切分后的文本列表。
    """
    if len(text) <= max_size:
        return [text]

    paragraphs = text.split("\n\n")
    sub_chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_size and current_chunk:
            sub_chunks.append(current_chunk.strip())
            # 保留 overlap
            current_chunk = current_chunk[-overlap:] + "\n\n" + para if overlap > 0 else para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        sub_chunks.append(current_chunk.strip())

    # 如果段落切分仍然产生超长块，按字符硬切
    final_chunks: list[str] = []
    for chunk in sub_chunks:
        if len(chunk) <= max_size:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), max_size - overlap):
                final_chunks.append(chunk[i:i + max_size])

    return final_chunks


async def load_file(file_path: str, db: AsyncSession) -> int:
    """读取 Markdown 文件，按标题切分并入库。

    Args:
        file_path: Markdown 文件路径。
        db: 异步数据库会话。

    Returns:
        导入的 chunk 数量。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    source_file = os.path.basename(file_path)
    heading_chunks = _split_by_headings(content)

    # 展开所有 chunk（含二次切分）
    all_chunks: list[dict[str, Any]] = []
    for hc in heading_chunks:
        sub_texts = _secondary_split(hc["text"])
        for sub_text in sub_texts:
            all_chunks.append({
                "heading_path": hc["heading_path"],
                "text": sub_text,
            })

    if not all_chunks:
        logger.warning(f"文件 {source_file} 切分后无有效内容")
        return 0

    # 批量生成 embedding
    texts = [c["heading_path"] + "\n" + c["text"] for c in all_chunks]
    embeddings = await embed_batch(texts)

    # 批量入库
    db_chunks: list[KnowledgeChunk] = []
    for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        db_chunk = KnowledgeChunk(
            source_file=source_file,
            heading_path=chunk["heading_path"],
            chunk_index=idx,
            original_text=chunk["text"],
            embedding=embedding,
        )
        db_chunks.append(db_chunk)

    db.add_all(db_chunks)
    await db.flush()

    logger.info(f"文件 {source_file} 已导入 {len(db_chunks)} 个 chunk")
    return len(db_chunks)
