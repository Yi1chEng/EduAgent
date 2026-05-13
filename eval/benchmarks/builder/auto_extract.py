"""半自动出题脚本:从教材 .md 中抽取候选 Q-A 对,产出待人工核对的 JSONL。

工作流:
  1. 读教材 md → 按 ## 二级标题切段
  2. 对每段调 LLM 生成 1-3 个候选问题 + 参考答案
  3. 写入 candidates.jsonl,人工 review 后改名为正式基准

不入应用主流程,仅评估准备阶段使用。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是教育评估数据集出题助手。基于教材段落,产出 1-3 条高质量的问答对。

输出严格遵循 JSON 数组格式,每个元素含字段:
- query: 自然语言问题 (中文)
- gold_answer: 简洁准确的参考答案
- type: factual / conceptual / reasoning / multi_hop / instruction 之一
- difficulty: easy / medium / hard 之一

只输出 JSON 数组,不要任何解释或 Markdown 围栏。"""


def _split_by_h2(content: str) -> list[tuple[str, str]]:
    """按 ## 二级标题切段,返回 [(heading_path, text), ...]"""
    lines = content.split("\n")
    out: list[tuple[str, str]] = []
    h1 = ""
    h2 = ""
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text and (h1 or h2):
            path = " > ".join([p for p in (h1, h2) if p]) or "未分类"
            out.append((path, text))

    for line in lines:
        if re.match(r"^##\s+", line):
            flush()
            h2 = re.sub(r"^##\s+", "", line).strip()
            buf = []
        elif re.match(r"^#\s+", line):
            flush()
            h1 = re.sub(r"^#\s+", "", line).strip()
            h2 = ""
            buf = []
        else:
            buf.append(line)
    flush()
    return out


def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL_NAME"],
        temperature=0.2,
    )


def _parse_qa(raw: str) -> list[dict[str, Any]]:
    """剥围栏并解析 JSON 数组。"""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return data


async def _generate_for_section(
    llm: ChatOpenAI, heading_path: str, section_text: str
) -> list[dict[str, Any]]:
    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"【段落标题】{heading_path}\n\n【段落内容】\n{section_text[:1500]}"),
    ])
    raw = response.content if isinstance(response.content, str) else str(response.content)
    return _parse_qa(raw)


async def build_from_markdown(
    md_path: Path, output_path: Path, max_sections: int | None = None
) -> int:
    content = md_path.read_text(encoding="utf-8")
    sections = _split_by_h2(content)
    if max_sections:
        sections = sections[:max_sections]
    logger.info(f"切出 {len(sections)} 段")

    llm = _make_llm()
    rows: list[dict[str, Any]] = []
    for idx, (path, text) in enumerate(sections):
        try:
            qas = await _generate_for_section(llm, path, text)
        except Exception as e:
            logger.warning(f"段落 {idx} 生成失败: {e}")
            continue
        for j, qa in enumerate(qas):
            rows.append({
                "id": f"AUTO_{md_path.stem}_{idx:03d}_{j}",
                "subject": md_path.stem,
                "chapter": path,
                "type": qa.get("type", "factual"),
                "difficulty": qa.get("difficulty", "medium"),
                "query": qa.get("query", ""),
                "gold_answer": qa.get("gold_answer", ""),
                "gold_chunk_ids": [],
                "needs_tool": None,
                "notes": "auto_extracted; 需人工核对并补 gold_chunk_ids",
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"写入 {len(rows)} 条候选到 {output_path}")
    return len(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path", help="教材 .md 文件路径")
    parser.add_argument("--output", default="eval/benchmarks/candidates.jsonl")
    parser.add_argument("--max-sections", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(
        build_from_markdown(Path(args.md_path), Path(args.output), args.max_sections)
    )


if __name__ == "__main__":
    main()
