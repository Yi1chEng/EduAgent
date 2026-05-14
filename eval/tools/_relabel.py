"""为 qa_v1.jsonl 自动反查 gold_chunk_ids,基于文本子串匹配。

策略:每条 query 配一个或多个关键短语,在 DB 全表 LIKE 搜索得到候选 chunk id 列表。
不是论文级标注(论文需要人工),但比盲填准确得多,适合 pipeline 验证。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, "/app")
from sqlalchemy import text
from app.db.database import async_session_factory


# (query_id, [phrase1, phrase2, ...]) — chunks 命中任一短语即视作 gold
# 用宽语义短语取并集,模拟人工标注的 recall 上界。注意:每条会拉到最多 30 个候选,
# 论文使用前请人工 review 并剔除明显无关项。
RELABEL_SPEC: dict[str, list[str]] = {
    "Q001": ["传感器", "执行器", "Sensors", "Actuators"],
    "Q002": ["AlphaGo", "强化学习发现"],
    "Q003": ["规划式", "Deliberative", "搜索算法", "A*算法"],
    "Q004": ["混合式", "Hybrid Agents"],
    "Q005": ["神经符号", "Neuro-Symbolic"],
    "Q006": ["符号主义", "可解释", "规则库", "知识图谱"],
    "Q007": ["亚符号", "神经网络", "内隐"],
    "Q008": ["系统 1", "系统 2", "快思考", "慢思考"],
    "Q009": ["传统智能体", "反射", "演进", "学习型"],
    "Q010": [],  # OOS — 知识库内不应有命中
}


async def lookup(phrases: list[str]) -> list[int]:
    if not phrases:
        return []
    async with async_session_factory() as db:
        ids: set[int] = set()
        for phr in phrases:
            # 用 ILIKE 取消大小写敏感
            sql = text(
                "SELECT id FROM knowledge_chunks WHERE original_text ILIKE :p ORDER BY id LIMIT 30"
            )
            rows = (await db.execute(sql, {"p": f"%{phr}%"})).fetchall()
            ids.update(r.id for r in rows)
        return sorted(ids)


async def main() -> None:
    path = Path("/app/eval/benchmarks/qa_v1.jsonl")
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for it in items:
        spec = RELABEL_SPEC.get(it["id"], [])
        ids = await lookup(spec) if spec else []
        it["gold_chunk_ids"] = ids
        print(f"{it['id']} ({len(spec)} phrases): {len(ids)} gold chunks → {ids[:8]}{'...' if len(ids)>8 else ''}")
    path.write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {path}")


asyncio.run(main())
