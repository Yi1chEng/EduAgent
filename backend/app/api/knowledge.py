"""知识库管理接口：上传、列表、删除教材文档。"""

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.database import get_db
from app.models.db_models import KnowledgeChunk
from app.models.schemas import (
    KnowledgeDeleteResponse,
    KnowledgeListItem,
    KnowledgeListResponse,
    KnowledgeUploadResponse,
)
from app.rag.knowledge_loader import load_file

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.post("/knowledge/upload", response_model=KnowledgeUploadResponse)
async def upload_knowledge(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
) -> KnowledgeUploadResponse:
    """上传 Markdown 文件并导入知识库。

    接收 .md 文件，调用 knowledge_loader 按标题切分并入库。

    Args:
        file: 上传的 Markdown 文件。
        db: 异步数据库会话。

    Returns:
        导入结果，包含文件名和 chunk 数量。
    """
    if not file.filename or not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="仅支持 .md 格式的 Markdown 文件")

    # 保存文件到上传目录
    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # 检查是否已存在同名文件的 chunk，如果有则先删除
        existing_stmt = delete(KnowledgeChunk).where(
            KnowledgeChunk.source_file == file.filename
        )
        await db.execute(existing_stmt)

        # 导入知识库
        chunks_count = await load_file(file_path, db)

        logger.info(f"知识库上传成功: {file.filename}, {chunks_count} 个 chunk")

        return KnowledgeUploadResponse(
            message="知识库导入成功",
            source_file=file.filename,
            chunks_count=chunks_count,
        )

    except Exception as e:
        logger.error(f"知识库上传失败: {e}", exc_info=True)
        # 清理已保存的文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"知识库导入失败: {str(e)}")


@router.get("/knowledge/list", response_model=KnowledgeListResponse)
async def list_knowledge(
    db: AsyncSession = Depends(get_db),
) -> KnowledgeListResponse:
    """列出已导入的所有文档及其 chunk 数量。

    Returns:
        文档列表，每个文档包含名称、chunk 数量和创建时间。
    """
    stmt = (
        select(
            KnowledgeChunk.source_file,
            func.count(KnowledgeChunk.id).label("chunks_count"),
            func.min(KnowledgeChunk.created_at).label("created_at"),
        )
        .group_by(KnowledgeChunk.source_file)
        .order_by(func.min(KnowledgeChunk.created_at).desc())
    )

    result = await db.execute(stmt)
    rows = result.all()

    documents = [
        KnowledgeListItem(
            source_file=row.source_file,
            chunks_count=row.chunks_count,
            created_at=row.created_at.isoformat() if row.created_at else "",
        )
        for row in rows
    ]

    return KnowledgeListResponse(documents=documents)


@router.delete("/knowledge/{doc_name}", response_model=KnowledgeDeleteResponse)
async def delete_knowledge(
    doc_name: str,
    db: AsyncSession = Depends(get_db),
) -> KnowledgeDeleteResponse:
    """删除指定文档的所有 chunk。

    Args:
        doc_name: 文档文件名。
        db: 异步数据库会话。

    Returns:
        删除结果，包含删除的 chunk 数量。
    """
    # 先查询数量
    count_stmt = (
        select(func.count(KnowledgeChunk.id))
        .where(KnowledgeChunk.source_file == doc_name)
    )
    result = await db.execute(count_stmt)
    count = result.scalar() or 0

    if count == 0:
        raise HTTPException(status_code=404, detail=f"文档 '{doc_name}' 不存在")

    # 删除
    delete_stmt = delete(KnowledgeChunk).where(
        KnowledgeChunk.source_file == doc_name
    )
    await db.execute(delete_stmt)

    # 删除上传的文件
    file_path = os.path.join(settings.UPLOAD_DIR, doc_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    logger.info(f"删除文档 '{doc_name}'，共 {count} 个 chunk")

    return KnowledgeDeleteResponse(
        message=f"文档 '{doc_name}' 已删除",
        deleted_count=count,
    )
