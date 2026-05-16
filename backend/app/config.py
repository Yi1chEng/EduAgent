"""统一配置模块，使用 Pydantic Settings 从 .env 文件读取配置。"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置，所有 API 密钥均通过环境变量注入，禁止硬编码。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM 配置
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.siliconflow.cn/v1"
    LLM_MODEL_NAME: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    # Embedding 配置
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://api.siliconflow.cn/v1"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"

    # Reranker 配置
    RERANKER_API_KEY: str = ""
    RERANKER_BASE_URL: str = "https://api.siliconflow.cn/v1/rerank"
    RERANKER_MODEL_NAME: str = "Qwen/Qwen3-Reranker-4B"

    # 数据库配置
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/eduagent"

    # 企业微信 Webhook
    WECHAT_WEBHOOK_URL: str = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"

    # 上传文件目录
    UPLOAD_DIR: str = "uploads"


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置单例。"""
    return Settings()
