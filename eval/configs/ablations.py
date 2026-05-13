"""Ablation 配置清单(纯 Python,易于在 runner 中 import)。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.rag.retriever import RetrievalConfig


@dataclass(frozen=True)
class RetrievalAblation:
    """RQ1 单条 ablation。"""

    name: str
    description: str
    config: RetrievalConfig
    # 该配置最终从哪一阶段取 top_k:fused / reranked / final
    # vector-only 时取 vector;启用 RRF 取 fused;启用 reranker 取 reranked。
    stage: str = "final"


RQ1_ABLATIONS: list[RetrievalAblation] = [
    RetrievalAblation(
        name="R1_vector_only",
        description="仅向量召回(over-fetch=1),无关键词层 / RRF / reranker",
        config=RetrievalConfig(
            use_keyword=False, use_rrf=False, use_reranker=False, overfetch_multiplier=1
        ),
        stage="vector",
    ),
    RetrievalAblation(
        name="R2_vector_overfetch",
        description="向量召回 + 默认 over-fetch(无关键词层 / RRF / reranker)",
        config=RetrievalConfig(use_keyword=False, use_rrf=False, use_reranker=False),
        stage="vector",
    ),
    RetrievalAblation(
        name="R3_plus_keyword",
        description="向量 + 关键词层,但不用 RRF(仅关键词重排候选)",
        config=RetrievalConfig(use_keyword=True, use_rrf=False, use_reranker=False),
        stage="keyword",
    ),
    RetrievalAblation(
        name="R4_plus_rrf",
        description="向量 + 关键词 + RRF 融合",
        config=RetrievalConfig(use_keyword=True, use_rrf=True, use_reranker=False),
        stage="fused",
    ),
    RetrievalAblation(
        name="R5_full",
        description="完整混合检索:向量 + 关键词 + RRF + Reranker",
        config=RetrievalConfig(use_keyword=True, use_rrf=True, use_reranker=True),
        stage="final",
    ),
]


@dataclass(frozen=True)
class AnswerAblation:
    """RQ2 系统级 ablation:不同 LLM 或不同 RAG 路径。"""

    name: str
    description: str
    use_rag: bool = True
    use_tools: bool = True
    # LLM 来源:None=用 settings 默认,否则 (api_key_env, base_url_env, model_env)
    llm_override_env_prefix: str | None = None


RQ2_ABLATIONS: list[AnswerAblation] = [
    AnswerAblation(name="A0_plain_llm", description="纯 LLM(无 RAG / 无工具)", use_rag=False, use_tools=False),
    AnswerAblation(name="A1_rag_only", description="RAG + LLM,无工具", use_rag=True, use_tools=False),
    AnswerAblation(name="A2_eduagent_full", description="完整 EduAgent(RAG + 工具)", use_rag=True, use_tools=True),
    # 切换 LLM 的 ablation 需要预设 BASELINE_* 环境变量,留作扩展位
    AnswerAblation(
        name="A3_baseline_llm",
        description="替换 LLM 作为基线(读 BASELINE_* 环境变量)",
        use_rag=True,
        use_tools=True,
        llm_override_env_prefix="BASELINE",
    ),
]
