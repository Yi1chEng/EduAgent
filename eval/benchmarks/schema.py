"""QA 基准集 schema 定义与校验。

每条样本最少包含 query / gold_answer / gold_chunk_ids 三项,其余字段用于分层统计。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional

QuestionType = Literal["factual", "conceptual", "reasoning", "multi_hop", "instruction", "out_of_scope"]
Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class QAItem:
    """一条 QA 基准样本。"""

    id: str
    query: str
    gold_answer: str
    gold_chunk_ids: list[int] = field(default_factory=list)  # 命中的 knowledge_chunks.id 列表
    subject: Optional[str] = None
    chapter: Optional[str] = None
    type: Optional[QuestionType] = None
    difficulty: Optional[Difficulty] = None
    needs_tool: Optional[str] = None  # None / "mermaid" / "github" / ...
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "QAItem":
        return cls(
            id=str(d["id"]),
            query=str(d["query"]),
            gold_answer=str(d.get("gold_answer", "")),
            gold_chunk_ids=list(d.get("gold_chunk_ids", [])),
            subject=d.get("subject"),
            chapter=d.get("chapter"),
            type=d.get("type"),
            difficulty=d.get("difficulty"),
            needs_tool=d.get("needs_tool"),
            notes=d.get("notes"),
        )

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "query": self.query,
            "gold_answer": self.gold_answer,
            "gold_chunk_ids": list(self.gold_chunk_ids),
        }
        for k in ("subject", "chapter", "type", "difficulty", "needs_tool", "notes"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


def load_jsonl(path: str | Path) -> list[QAItem]:
    """从 JSONL 加载 QA 基准集；非法行抛出含位置的异常。"""
    p = Path(path)
    items: list[QAItem] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(QAItem.from_dict(json.loads(line)))
            except Exception as e:
                raise ValueError(f"{p}:{lineno}: 解析失败 — {e}") from e
    if not items:
        raise ValueError(f"{p}: 空基准集")
    _validate_uniqueness(items)
    return items


def dump_jsonl(items: list[QAItem], path: str | Path) -> None:
    """写出 JSONL（一行一条）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def _validate_uniqueness(items: list[QAItem]) -> None:
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise ValueError(f"重复的 id: {it.id}")
        seen.add(it.id)


def iter_by_type(items: list[QAItem]) -> Iterator[tuple[str, list[QAItem]]]:
    """按 type 分组,便于分层报告。"""
    buckets: dict[str, list[QAItem]] = {}
    for it in items:
        buckets.setdefault(it.type or "unknown", []).append(it)
    yield from buckets.items()
