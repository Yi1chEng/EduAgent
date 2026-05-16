"""知识库检索评估 — 单脚本跑完。

用法：
    docker compose exec backend python -m eval.eval

输出：
    - 每条问题的 Recall@5 / Recall@10 / MRR
    - 汇总均值
"""

import json
import sys
from pathlib import Path
from typing import Any

from app.rag.retriever import retrieve_detailed

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
TOP_K = 10


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            questions.append(json.loads(line))
    if not questions:
        print("问题集为空！")
        sys.exit(1)
    return questions


def calc_recall(gold_ids: list[int], retrieved_ids: list[int]) -> float:
    if not gold_ids:
        return -1
    hit = len(set(gold_ids) & set(retrieved_ids))
    return hit / len(gold_ids)


def calc_mrr(gold_ids: list[int], retrieved_ids: list[int]) -> float:
    if not gold_ids:
        return -1
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0


async def main() -> None:
    questions = load_questions(QUESTIONS_FILE)
    print(f"加载 {len(questions)} 条问题\n")

    results = []
    for q in questions:
        gold = q.get("gold_chunk_ids") or []
        stages = await retrieve_detailed(q["query"], top_k=TOP_K)
        retrieved_ids = [doc["id"] for doc in stages["final"]]

        recall5 = calc_recall(gold, retrieved_ids[:5])
        recall10 = calc_recall(gold, retrieved_ids[:10])
        mrr = calc_mrr(gold, retrieved_ids[:10])

        results.append({
            "id": q["id"],
            "query": q["query"],
            "gold_ids": gold,
            "retrieved_ids": retrieved_ids,
            "recall5": recall5,
            "recall10": recall10,
            "mrr": mrr,
        })

    # 表格输出
    labeled = [r for r in results if r["gold_ids"]]
    unlabeled = [r for r in results if not r["gold_ids"]]

    print(f"{'ID':<6} {'问题':<40} {'R@5':<8} {'R@10':<8} {'MRR':<8}")
    print("-" * 72)
    for r in results:
        r5 = f'{r["recall5"]:.2f}' if r["recall5"] >= 0 else "  -"
        r10 = f'{r["recall10"]:.2f}' if r["recall10"] >= 0 else "  -"
        m = f'{r["mrr"]:.2f}' if r["mrr"] >= 0 else "  -"
        title = r["query"][:38] + ("…" if len(r["query"]) > 38 else "")
        print(f"{r['id']:<6} {title:<40} {r5:<8} {r10:<8} {m:<8}")

    if labeled:
        avg_r5 = sum(r["recall5"] for r in labeled) / len(labeled)
        avg_r10 = sum(r["recall10"] for r in labeled) / len(labeled)
        avg_mrr = sum(r["mrr"] for r in labeled) / len(labeled)
        print("-" * 72)
        print(f"{'':6} {'平均 (n=' + str(len(labeled)) + ')':<40} {avg_r5:<8.2f} {avg_r10:<8.2f} {avg_mrr:<8.2f}")

    if unlabeled:
        print(f"\n提示: {len(unlabeled)} 条问题未标注 gold_chunk_ids，跳过了指标计算。"
              " 先跑一次看检索结果，再把正确的 chunk ID 填入 questions.jsonl。")

    # 存明细
    out_file = EVAL_DIR / "outputs" / "last_result.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n明细已保存: {out_file}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
