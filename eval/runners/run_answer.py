"""RQ2 端到端答案质量 runner。

为每个 AnswerAblation × 每条 query:
  1. 视 ablation 决定是否走 RAG
  2. 调 LLM 生成 answer(直接在 Python 内拼调用,绕过 FastAPI 层加速 + 避免 SSE 解析)
  3. 调用 RAGAS judge 评 Faithfulness / Answer Relevance / Context Relevance
  4. 把每条 (ablation, sample) 落 per_sample.jsonl
最后按 ablation 汇总到 aggregate.csv / .md / .tex

环境变量:
    LLM_*           被测主模型(默认)
    BASELINE_*      可选,A3_baseline_llm 用
    JUDGE_*         可选,evaluator;未配置则回落到 LLM_*(仅 smoke,论文需独立 judge)

用法:
    python -m eval.runners.run_answer \
        --benchmark eval/benchmarks/qa_v1.jsonl \
        --output-dir eval/outputs/runs/$(date +%Y%m%d_%H%M%S)_rq2
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# import paths
_BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.rag.citation import format_citations, format_context  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402

from eval.benchmarks.schema import load_jsonl  # noqa: E402
from eval.configs.ablations import AnswerAblation, RQ2_ABLATIONS  # noqa: E402
from eval.metrics.ragas_eval import JudgeConfig, _make_judge, evaluate_sample  # noqa: E402
from eval.metrics.report import to_csv, to_latex, to_markdown  # noqa: E402

logger = logging.getLogger(__name__)


GENERATOR_PROMPT_RAG = """你是 EduAgent,基于参考资料生成准确、有教育价值的回答。
规则:
1. 必须基于参考资料,使用 [1][2] 标记引用
2. 资料不足时坦诚说明
3. 使用 Markdown 格式

参考资料:
{context}

引用来源:
{citations}
"""

GENERATOR_PROMPT_PLAIN = (
    "你是教育领域 AI 助手,基于自身知识直接回答用户问题。使用 Markdown 格式。"
)


def _make_llm(prefix: str | None) -> ChatOpenAI:
    if prefix is None:
        s = get_settings()
        return ChatOpenAI(
            api_key=s.LLM_API_KEY,
            base_url=s.LLM_BASE_URL,
            model=s.LLM_MODEL_NAME,
            temperature=0.0,
        )
    return ChatOpenAI(
        api_key=os.environ[f"{prefix}_API_KEY"],
        base_url=os.environ[f"{prefix}_BASE_URL"],
        model=os.environ[f"{prefix}_MODEL_NAME"],
        temperature=0.0,
    )


async def _generate_answer(
    llm: ChatOpenAI,
    query: str,
    docs: list[dict],
    use_rag: bool,
) -> str:
    if use_rag:
        system = GENERATOR_PROMPT_RAG.format(
            context=format_context(docs),
            citations=format_citations(docs),
        )
    else:
        system = GENERATOR_PROMPT_PLAIN
    response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=query)])
    content = response.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return content or ""


async def _with_retry(coro_factory, attempts: int = 3, base_delay: float = 2.0):
    """对 LLM/judge 调用做指数退避重试,缓解 API 间歇性 Connection error / 限流。"""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            delay = base_delay * (2 ** i)
            logger.warning(f"调用失败({type(e).__name__}: {e}),{delay}s 后重试 ({i+1}/{attempts})")
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def _run_one(
    ablation: AnswerAblation,
    llm: ChatOpenAI,
    judge: ChatOpenAI,
    query: str,
    top_k: int,
    inter_call_delay_sec: float = 0.0,
) -> dict[str, Any]:
    if ablation.use_rag:
        docs = await retrieve(query, top_k=top_k)
    else:
        docs = []
    answer = await _with_retry(lambda: _generate_answer(llm, query, docs, ablation.use_rag))
    if inter_call_delay_sec:
        await asyncio.sleep(inter_call_delay_sec)
    retrieved_texts = [d["text"] for d in docs] if docs else []
    scores = await _with_retry(lambda: evaluate_sample(judge, query, answer, retrieved_texts))
    return {
        "ablation": ablation.name,
        "answer": answer,
        "retrieved_chunk_ids": [int(d["id"]) for d in docs],
        **scores,
    }


async def main_async(args: argparse.Namespace) -> None:
    items = load_jsonl(args.benchmark)
    logger.info(f"加载 {len(items)} 条样本")

    judge_cfg = JudgeConfig.from_env()
    judge = _make_judge(judge_cfg)
    logger.info(f"judge 模型: {judge_cfg.model} @ {judge_cfg.base_url}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows: list[dict[str, Any]] = []
    for ablation in RQ2_ABLATIONS:
        # 该 ablation 用的 LLM
        try:
            llm = _make_llm(ablation.llm_override_env_prefix)
        except KeyError as e:
            logger.warning(f"跳过 {ablation.name}: 缺少环境变量 {e}")
            continue

        logger.info(f"== ablation {ablation.name}: {ablation.description}")
        for idx, item in enumerate(items):
            try:
                row = await _run_one(
                    ablation, llm, judge, item.query, args.top_k,
                    inter_call_delay_sec=args.inter_call_delay,
                )
                row.update({
                    "query_id": item.id,
                    "query": item.query,
                    "type": item.type,
                    "difficulty": item.difficulty,
                })
            except Exception as e:
                logger.error(f"{ablation.name}/{item.id} 失败: {e}")
                continue
            per_sample_rows.append(row)
            logger.info(f"  [{idx+1}/{len(items)}] {item.id} → AR={row['answer_relevance']:.2f} F={row['faithfulness']:.2f} CR={row['context_relevance']:.2f}")

    # 写 per-sample
    per_sample_path = output_dir / "per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for r in per_sample_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总
    summary_rows: list[dict[str, Any]] = []
    for ablation in RQ2_ABLATIONS:
        rows = [r for r in per_sample_rows if r["ablation"] == ablation.name]
        if not rows:
            continue
        n = len(rows)
        summary_rows.append({
            "ablation": ablation.name,
            "n": n,
            "faithfulness": sum(r["faithfulness"] for r in rows) / n,
            "answer_relevance": sum(r["answer_relevance"] for r in rows) / n,
            "context_relevance": sum(r["context_relevance"] for r in rows) / n,
        })

    cols = ["ablation", "n", "faithfulness", "answer_relevance", "context_relevance"]
    to_csv(summary_rows, output_dir / "aggregate.csv", columns=cols)
    (output_dir / "aggregate.md").write_text(to_markdown(summary_rows, columns=cols), encoding="utf-8")
    (output_dir / "aggregate.tex").write_text(
        to_latex(summary_rows, columns=cols, caption="RQ2: 答案质量(RAGAS)", label="tab:rq2"),
        encoding="utf-8",
    )

    manifest = {
        "task": "RQ2_answer",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(args.benchmark),
        "benchmark_sha256_16": hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest()[:16],
        "top_k": args.top_k,
        "judge_model": judge_cfg.model,
        "judge_base_url": judge_cfg.base_url,
        "ablations": [a.name for a in RQ2_ABLATIONS],
        "n_items": len(items),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"完成 → {output_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--inter-call-delay", type=float, default=0.0,
        help="生成与判定之间的固定 sleep 秒数,缓解 API 限流(默认 0)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
