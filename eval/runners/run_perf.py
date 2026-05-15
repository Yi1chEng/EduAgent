"""RQ4 性能压测 runner:测 SSE 真实 TTFT / E2E / QPS。

实现方式:对 /api/chat/stream 用 httpx + AsyncClient 流式读取,记录每个 SSE 事件到达时间戳。
- TTFT_ms = 首 `token` event 时间 - 请求发起时间
- E2E_ms  = `done` event 时间 - 请求发起时间
- Mermaid_ms = `mermaid` event 时间 - 请求发起时间(若有)

并发模式:asyncio.Semaphore 限并发,沿用 benchmark 中的 query。

用法:
    python -m eval.runners.run_perf \
        --base-url http://localhost:8000 \
        --benchmark eval/benchmarks/qa_v1.jsonl \
        --concurrencies 1,5,10,20 \
        --repeats-per-query 3 \
        --output-dir eval/outputs/runs/$(date +%Y%m%d_%H%M%S)_rq4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# eval 同级 import
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.benchmarks.schema import load_jsonl  # noqa: E402
from eval.metrics.perf_stats import summarize_latencies, throughput  # noqa: E402
from eval.metrics.report import to_csv, to_latex, to_markdown  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class RequestTiming:
    query_id: str
    concurrency: int
    start_ts: float
    ttft_ms: float | None = None
    done_ms: float | None = None
    mermaid_ms: float | None = None
    first_tool_ms: float | None = None
    error: str | None = None


# 解析 SSE event 块:event: xxx \n data: yyy
_EVENT_HEAD = re.compile(rb"event:\s*(\S+)")


async def _stream_one(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
    query_id: str,
    concurrency: int,
) -> RequestTiming:
    timing = RequestTiming(query_id=query_id, concurrency=concurrency, start_ts=time.perf_counter())
    payload = {
        "query": query,
        "session_id": f"perf-{uuid.uuid4().hex[:8]}",
        # 性能测试关闭所有工具，避免引入外部依赖噪音
        "enabled_tools": {
            "generate_mermaid": False,
            "send_wechat": False,
            "github": False,
            "fetch": False,
            "filesystem": False,
        },
    }
    try:
        async with client.stream(
            "POST",
            f"{base_url}/api/chat/stream",
            json=payload,
            headers={"Accept": "text/event-stream"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        ) as response:
            response.raise_for_status()
            buffer = b""
            async for chunk in response.aiter_bytes():
                buffer += chunk
                while b"\n\n" in buffer or b"\r\n\r\n" in buffer:
                    sep_pos = buffer.find(b"\r\n\r\n")
                    if sep_pos < 0:
                        sep_pos = buffer.find(b"\n\n")
                        sep_len = 2
                    else:
                        sep_len = 4
                    event_bytes = buffer[:sep_pos]
                    buffer = buffer[sep_pos + sep_len:]
                    m = _EVENT_HEAD.search(event_bytes)
                    if not m:
                        continue
                    name = m.group(1).decode("ascii", errors="ignore")
                    now_ms = (time.perf_counter() - timing.start_ts) * 1000
                    if name == "token" and timing.ttft_ms is None:
                        timing.ttft_ms = now_ms
                    elif name == "mermaid":
                        timing.mermaid_ms = now_ms
                    elif name == "tool_result" and timing.first_tool_ms is None:
                        timing.first_tool_ms = now_ms
                    elif name == "done":
                        timing.done_ms = now_ms
    except Exception as e:
        timing.error = str(e)
        logger.warning(f"请求 {query_id} 失败: {e}")
    return timing


async def _run_concurrency(
    base_url: str, queries: list[tuple[str, str]], concurrency: int, total_requests: int
) -> tuple[list[RequestTiming], float]:
    """以指定并发跑 total_requests 个请求,返回 (timings, wall_time_sec)。"""
    sem = asyncio.Semaphore(concurrency)
    timings: list[RequestTiming] = []

    async with httpx.AsyncClient(http2=False) as client:

        async def _one(i: int) -> None:
            qid, q = queries[i % len(queries)]
            async with sem:
                t = await _stream_one(client, base_url, q, qid, concurrency)
                timings.append(t)

        start = time.perf_counter()
        await asyncio.gather(*[_one(i) for i in range(total_requests)])
        wall = time.perf_counter() - start
    return timings, wall


async def main_async(args: argparse.Namespace) -> None:
    items = load_jsonl(args.benchmark)
    queries: list[tuple[str, str]] = [(it.id, it.query) for it in items]
    logger.info(f"加载 {len(queries)} 条查询")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    summary_rows: list[dict[str, Any]] = []
    all_per_request: list[dict[str, Any]] = []

    for c in concurrencies:
        total = len(queries) * args.repeats_per_query
        logger.info(f"== 并发 {c},共 {total} 请求")
        timings, wall = await _run_concurrency(args.base_url, queries, c, total)

        # 各延迟统计(成功请求)
        ok = [t for t in timings if t.error is None and t.done_ms is not None]
        ttft = [t.ttft_ms for t in ok if t.ttft_ms is not None]
        done = [t.done_ms for t in ok if t.done_ms is not None]

        ttft_stats = summarize_latencies(ttft)
        done_stats = summarize_latencies(done)
        qps = throughput(len(ok), wall)

        summary_rows.append({
            "concurrency": c,
            "n_requests": total,
            "n_success": len(ok),
            "n_error": len(timings) - len(ok),
            "wall_time_sec": round(wall, 2),
            "qps": round(qps, 2),
            "ttft_p50_ms": round(ttft_stats.get("p50", 0.0), 1),
            "ttft_p95_ms": round(ttft_stats.get("p95", 0.0), 1),
            "ttft_p99_ms": round(ttft_stats.get("p99", 0.0), 1),
            "e2e_p50_ms": round(done_stats.get("p50", 0.0), 1),
            "e2e_p95_ms": round(done_stats.get("p95", 0.0), 1),
            "e2e_p99_ms": round(done_stats.get("p99", 0.0), 1),
        })

        for t in timings:
            all_per_request.append({
                "concurrency": c,
                "query_id": t.query_id,
                "ttft_ms": t.ttft_ms,
                "done_ms": t.done_ms,
                "mermaid_ms": t.mermaid_ms,
                "first_tool_ms": t.first_tool_ms,
                "error": t.error,
            })

    # 落盘
    per_request_path = output_dir / "per_request.jsonl"
    with per_request_path.open("w", encoding="utf-8") as f:
        for r in all_per_request:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cols = [
        "concurrency", "n_requests", "n_success", "n_error", "wall_time_sec", "qps",
        "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
        "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms",
    ]
    to_csv(summary_rows, output_dir / "aggregate.csv", columns=cols)
    (output_dir / "aggregate.md").write_text(to_markdown(summary_rows, columns=cols), encoding="utf-8")
    (output_dir / "aggregate.tex").write_text(
        to_latex(summary_rows, columns=cols, caption="RQ4: 流式接口性能(并发与延迟)", label="tab:rq4"),
        encoding="utf-8",
    )

    manifest = {
        "task": "RQ4_performance",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "benchmark": str(args.benchmark),
        "concurrencies": concurrencies,
        "repeats_per_query": args.repeats_per_query,
        "n_queries": len(queries),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"完成 → {output_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrencies", default="1,5,10", help="逗号分隔的并发档,如 1,5,10,20,50")
    parser.add_argument("--repeats-per-query", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
