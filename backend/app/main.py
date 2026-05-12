"""FastAPI 应用入口，配置 CORS、注册路由、启动时初始化数据库。"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, feedback, knowledge
from app.config import get_settings
from app.db.init_db import init_database
from app.graph.nodes import _get_mcp_tools
from app.rag.embeddings import _get_embeddings_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理：启动时初始化数据库、创建上传目录、预热外部依赖。"""
    # 启动阶段
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    try:
        await init_database()
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.warning(f"数据库初始化跳过（可能未连接）: {e}")

    # 预热：Embedding 客户端（同步构造，避免首请求时阻塞 event loop）
    try:
        _get_embeddings_model()
        logger.info("Embedding 模型已预热")
    except Exception as e:
        logger.warning(f"Embedding 预热失败: {e}")

    # 预热：MCP 工具（fork stdio 子进程 + 工具列表，首请求省 1~3s）
    try:
        tools = await _get_mcp_tools()
        logger.info(f"MCP 工具已预热，共 {len(tools)} 个")
    except Exception as e:
        logger.warning(f"MCP 工具预热失败: {e}")

    yield
    # 关闭阶段
    logger.info("应用关闭")


app = FastAPI(
    title="EduAgent - 教育多Agent大模型系统",
    description="面向教育场景的多Agent大模型系统，支持RAG检索增强、知识图谱可视化、代码执行等功能",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置：开发期允许所有源，关闭 credentials（与通配符共存合规）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(knowledge.router, prefix="/api", tags=["知识库"])
app.include_router(feedback.router, prefix="/api", tags=["反馈"])


@app.get("/", tags=["健康检查"])
async def root() -> dict[str, str]:
    """根路径健康检查。"""
    return {"status": "ok", "service": "EduAgent"}


@app.get("/health", tags=["健康检查"])
async def health_check() -> dict[str, str]:
    """健康检查接口。"""
    return {"status": "healthy"}
