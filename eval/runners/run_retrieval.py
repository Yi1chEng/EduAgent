"""RQ1 检索 ablation runner。

用法(在 backend 容器内 / 装好后端依赖的环境):
    python -m eval.runners.run_retrieval \
        --benchmark eval/benchmarks/qa_v1.jsonl \
        --output-dir eval/outputs/runs/$(date +%Y%m%d_%H%M%S)

输出:
    {output-dir}/per_sample.jsonl         每条样本 × 每个 ablation 的检索结果
    {output-dir}/aggregate.csv            ablation 级汇总指标
    {output-dir}/aggregate.md / .tex      论文表格
    {output-dir}/manifest.json            元数据(数据集 hash / 版本 / 时间)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 确保能从 backend/app 导入
_BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.db.database import async_session_factory  # noqa: E402
from app.rag.retriever import retrieve_detailed  # noqa: E402

from eval.benchmarks.schema import QAItem, load_jsonl  # noqa: E402
from eval.configs.ablations import RQ1_ABLATIONS, RetrievalAblation  # noqa: E402
from eval.metrics.retrieval import aggregate  # noqa: E402
from eval.metrics.report import to_csv, to_latex, to_markdown  # noqa: E402

logger = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


async def _run_one(
    item: QAItem, ablation: RetrievalAblation, top_k: int
) -> dict[str, Any]:
    """跑单条样本 × 单 ablation。"""
    async with async_session_factory() as db:
        stages = await retrieve_detailed(
            query=item.query, top_k=top_k, config=ablation.config, db=db
        )
    docs = stages.get(ablation.stage, stages["final"])
    retrieved_ids = [int(d["id"]) for d in docs[:top_k]]
    return {
        "query_id": item.id,
        "ablation": ablation.name,
        "stage": ablation.stage,
        "retrieved_ids": retrieved_ids,
        "gold_ids": list(item.gold_chunk_ids),
        "type": item.type,
        "difficulty": item.difficulty,
        "subject": item.subject,
    }


async def main_async(args: argparse.Namespace) -> None:
    items = load_jsonl(args.benchmark)
    logger.info(f"加载 {len(items)} 条样本(其中含 gold {sum(1 for i in items if i.gold_chunk_ids)} 条)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows: list[dict[str, Any]] = []
    for ablation in RQ1_ABLATIONS:
        logger.info(f"== ablation {ablation.name}: {ablation.description}")
        # 串行;若想加速可改成 gather 并控制并发
        for item in items:
            try:
                row = await _run_one(item, ablation, args.top_k)
            except Exception as e:
                logger.error(f"{ablation.name}/{item.id} 失败: {e}", exc_info=True)
                continue
            per_sample_rows.append(row)

    # 写 per-sample
    per_sample_path = output_dir / "per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for r in per_sample_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总 per ablation
    summary_rows: list[dict[str, Any]] = []
    for ablation in RQ1_ABLATIONS:
        rows = [r for r in per_sample_rows if r["ablation"] == ablation.name]
        agg = aggregate(rows, k_list=(1, 3, 5, 10))
        summary_rows.append({"ablation": ablation.name, **agg})

    cols = ["ablation", "n_samples", "recall@1", "recall@3", "recall@5", "recall@10", "hit@5", "mrr", "ndcg@10"]
    csv_path = to_csv(summary_rows, output_dir / "aggregate.csv", columns=cols)
    md_path = output_dir / "aggregate.md"
    md_path.write_text(to_markdown(summary_rows, columns=cols), encoding="utf-8")
    tex_path = output_dir / "aggregate.tex"
    tex_path.write_text(
        to_latex(summary_rows, columns=cols, caption="RQ1: 混合检索 ablation 结果", label="tab:rq1"),
        encoding="utf-8",
    )

    manifest = {
        "task": "RQ1_retrieval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(args.benchmark),
        "benchmark_sha256_16": _file_hash(Path(args.benchmark)),
        "top_k": args.top_k,
        "ablations": [a.name for a in RQ1_ABLATIONS],
        "n_items": len(items),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"完成 → {output_dir}")
    logger.info(f"  per_sample: {per_sample_path}")
    logger.info(f"  aggregate : {csv_path} / {md_path} / {tex_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="QA 基准 JSONL 路径")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--top-k", type=int, default=10, help="检索 top_k(默认 10,用于 nDCG@10)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
