"""性能延迟分位与吞吐统计。"""

from __future__ import annotations

import math
from typing import Iterable


def percentile(values: list[float], p: float) -> float:
    """nearest-rank 百分位(NIST 简单实现,适合小样本)。

    p 在 (0, 100] 区间。空列表返回 0。
    """
    if not values:
        return 0.0
    s = sorted(values)
    rank = math.ceil(p / 100.0 * len(s))
    rank = max(1, min(rank, len(s)))
    return s[rank - 1]


def summarize_latencies(latencies_ms: Iterable[float]) -> dict[str, float]:
    """对一组延迟(毫秒)给出 n / mean / std / p50 / p90 / p95 / p99 / max。"""
    arr = list(latencies_ms)
    n = len(arr)
    if n == 0:
        return {"n": 0}
    mean = sum(arr) / n
    var = sum((x - mean) ** 2 for x in arr) / n
    std = math.sqrt(var)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "p50": percentile(arr, 50),
        "p90": percentile(arr, 90),
        "p95": percentile(arr, 95),
        "p99": percentile(arr, 99),
        "min": min(arr),
        "max": max(arr),
    }


def throughput(total_requests: int, wall_time_sec: float) -> float:
    """QPS = 完成数 / 墙钟时间。"""
    if wall_time_sec <= 0:
        return 0.0
    return total_requests / wall_time_sec
