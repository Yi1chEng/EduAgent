"""检索质量指标:Recall@k、MRR、nDCG@k、Hit@k。

所有函数都对"单条样本"工作;批量统计交给 runner。
gold_ids 为空时一律返回 0.0,避免 0/0;runner 应在汇总时排除无标注样本。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(retrieved_ids: Sequence[int], gold_ids: Iterable[int], k: int) -> float:
    """Recall@k = |gold ∩ retrieved[:k]| / |gold|"""
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(gold_set & top_k) / len(gold_set)


def hit_at_k(retrieved_ids: Sequence[int], gold_ids: Iterable[int], k: int) -> float:
    """Hit@k: 至少命中一个 gold 时返回 1,否则 0。"""
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    for rid in retrieved_ids[:k]:
        if rid in gold_set:
            return 1.0
    return 0.0


def mrr(retrieved_ids: Sequence[int], gold_ids: Iterable[int]) -> float:
    """Mean Reciprocal Rank: 1 / 第一个命中 gold 的位次。

    单样本下其实是 RR;runner 平均后才是 MRR。
    """
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[int], gold_ids: Iterable[int], k: int) -> float:
    """二元相关度的 nDCG@k:gain = 1 if id in gold else 0。

    DCG  = Σ_{i=1..k} gain_i / log2(i + 1)
    IDCG = Σ_{i=1..min(k, |gold|)} 1 / log2(i + 1)
    nDCG = DCG / IDCG    (IDCG 为 0 时返回 0)
    """
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0

    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in gold_set:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(k, len(gold_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def aggregate(rows: list[dict], k_list: Sequence[int] = (1, 3, 5, 10)) -> dict[str, float]:
    """对一批 per-sample 结果做汇总,返回总体指标字典。

    每条 row 形如:
        {"retrieved_ids": [...], "gold_ids": [...]}

    返回:
        recall@1 / recall@3 / ... / mrr / ndcg@10 / hit@k_list
    """
    valid = [r for r in rows if r.get("gold_ids")]
    n = len(valid)
    out: dict[str, float] = {"n_samples": float(n), "n_total": float(len(rows))}
    if n == 0:
        return out

    for k in k_list:
        out[f"recall@{k}"] = sum(
            recall_at_k(r["retrieved_ids"], r["gold_ids"], k) for r in valid
        ) / n
        out[f"hit@{k}"] = sum(
            hit_at_k(r["retrieved_ids"], r["gold_ids"], k) for r in valid
        ) / n

    out["mrr"] = sum(mrr(r["retrieved_ids"], r["gold_ids"]) for r in valid) / n
    out["ndcg@10"] = sum(
        ndcg_at_k(r["retrieved_ids"], r["gold_ids"], 10) for r in valid
    ) / n
    return out
