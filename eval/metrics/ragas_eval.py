"""LLM-as-judge 实现 RAGAS 三件套:Faithfulness / Answer Relevance / Context Relevance。

不依赖 ragas pip 包,直接调 ChatOpenAI(可指向比被测模型更强的 judge,避免自评偏置)。

每个指标返回 [0,1] 浮点。
- Faithfulness: 回答中的事实声明,有多少能从 context 推出
- Answer Relevance: 回答与问题的相关度(用 LLM 反向生成 N 个问题,与原问题 embedding 相似度)
- Context Relevance: 检索到的 context 与问题的相关度

为减少外部 API 依赖,本模块用纯 prompt 让 judge 直接给 0/1 标签,然后取均值。
Answer Relevance 在没有 embedding 模型时退化为 judge 评估的相关度评分。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


@dataclass
class JudgeConfig:
    """评估专用 LLM 配置。建议用比被测模型更强的模型当 judge。"""

    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0

    @classmethod
    def from_env(cls, prefix: str = "JUDGE") -> "JudgeConfig":
        """从环境变量读取 JUDGE_API_KEY / JUDGE_BASE_URL / JUDGE_MODEL_NAME。

        未配置 JUDGE_* 时回落到 LLM_*(用被测模型当 judge,仅用于 smoke test)。
        """
        api_key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get("LLM_API_KEY", "")
        base_url = os.environ.get(f"{prefix}_BASE_URL") or os.environ.get(
            "LLM_BASE_URL", "https://api.openai.com/v1"
        )
        model = os.environ.get(f"{prefix}_MODEL_NAME") or os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
        return cls(api_key=api_key, base_url=base_url, model=model)


def _make_judge(cfg: JudgeConfig) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=cfg.temperature,
    )


FAITHFULNESS_SYS = """你是答案忠实度评判员。

任务:
1. 从下方"AI 回答"中抽出所有事实性陈述(独立判断为真/假的命题,跳过纯叙述与提问)
2. 对每条陈述,判断是否能由"参考资料"完整推出 → faithful=true/false
3. 输出严格的 JSON: {"claims": [{"text": "...", "faithful": true|false}, ...]}

只输出 JSON,不要任何解释。"""


ANSWER_REL_SYS = """你是答案相关度评判员。

针对"用户问题"与"AI 回答",输出 0-1 之间的相关度分数:
- 1.0 完全直接回答了问题
- 0.5 部分相关,有偏离或冗余
- 0.0 完全无关

只输出 JSON: {"score": 0.85}"""


CONTEXT_REL_SYS = """你是检索相关度评判员。

针对"用户问题"与下方一段"检索文本",输出 0-1 分:
- 1.0 包含直接回答该问题的关键信息
- 0.5 部分相关或仅边缘提到
- 0.0 无关

只输出 JSON: {"score": 0.7}"""


def _parse_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def faithfulness(
    judge: ChatOpenAI,
    question: str,
    answer: str,
    context: str,
) -> tuple[float, list[dict]]:
    """返回 (faithfulness 分数, 抽取出的 claim 明细)。"""
    if not answer.strip():
        return 0.0, []

    prompt = f"【用户问题】{question}\n\n【AI 回答】\n{answer}\n\n【参考资料】\n{context[:4000]}"
    response = await judge.ainvoke([SystemMessage(content=FAITHFULNESS_SYS), HumanMessage(content=prompt)])
    raw = response.content if isinstance(response.content, str) else str(response.content)
    data = _parse_json(raw)
    if not data or "claims" not in data:
        logger.warning("faithfulness judge 返回不可解析: %r", raw[:200])
        return 0.0, []
    claims = data.get("claims", [])
    if not claims:
        return 1.0, []  # 无事实声明则视为忠实
    score = sum(1 for c in claims if c.get("faithful")) / len(claims)
    return score, claims


async def answer_relevance(judge: ChatOpenAI, question: str, answer: str) -> float:
    if not answer.strip():
        return 0.0
    prompt = f"【用户问题】{question}\n\n【AI 回答】\n{answer}"
    response = await judge.ainvoke([SystemMessage(content=ANSWER_REL_SYS), HumanMessage(content=prompt)])
    raw = response.content if isinstance(response.content, str) else str(response.content)
    data = _parse_json(raw)
    if not data:
        logger.warning("answer_relevance judge 返回不可解析: %r", raw[:200])
        return 0.0
    try:
        return float(max(0.0, min(1.0, data.get("score", 0.0))))
    except (TypeError, ValueError):
        return 0.0


async def context_relevance(
    judge: ChatOpenAI, question: str, retrieved_texts: list[str]
) -> tuple[float, list[float]]:
    """对每段 context 单独打分,返回 (均值, 每段分数列表)。"""
    if not retrieved_texts:
        return 0.0, []
    scores: list[float] = []
    for chunk in retrieved_texts:
        prompt = f"【用户问题】{question}\n\n【检索文本】\n{chunk[:2000]}"
        response = await judge.ainvoke(
            [SystemMessage(content=CONTEXT_REL_SYS), HumanMessage(content=prompt)]
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        data = _parse_json(raw)
        if data and "score" in data:
            try:
                scores.append(float(max(0.0, min(1.0, data["score"]))))
                continue
            except (TypeError, ValueError):
                pass
        scores.append(0.0)
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, scores


async def evaluate_sample(
    judge: ChatOpenAI,
    question: str,
    answer: str,
    retrieved_texts: list[str],
) -> dict:
    """对单个样本跑完三项,返回 {faithfulness, answer_relevance, context_relevance, ...detail}。"""
    context_str = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(retrieved_texts))

    faith_score, claims = await faithfulness(judge, question, answer, context_str)
    ans_score = await answer_relevance(judge, question, answer)
    ctx_score, ctx_scores = await context_relevance(judge, question, retrieved_texts)

    return {
        "faithfulness": faith_score,
        "answer_relevance": ans_score,
        "context_relevance": ctx_score,
        "claims_count": len(claims),
        "context_chunk_scores": ctx_scores,
    }
