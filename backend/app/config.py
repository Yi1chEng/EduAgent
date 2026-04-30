"""统一配置模块，使用 Pydantic Settings 从 .env 文件读取配置。"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置，所有 API 密钥均通过环境变量注入，禁止硬编码。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM 配置
    LLM_API_KEY: str = "your_api_key"
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL_NAME: str = "gpt-4o-mini"

    # Embedding 配置
    EMBEDDING_API_KEY: str = "your_api_key"
    EMBEDDING_BASE_URL: str = "https://api.openai.com/v1"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"

    # 数据库配置
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/eduagent"

    # 企业微信 Webhook
    WECHAT_WEBHOOK_URL: str = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"

    # 内置 MCP Server 路径（stdio，容器内 /mcp_server 由 volume 挂载）
    MCP_SERVER_COMMAND: str = "python"
    MCP_SERVER_ARGS: str = "/mcp_server/server.py"

    # 上传文件目录
    UPLOAD_DIR: str = "uploads"


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置单例。"""
    return Settings()
