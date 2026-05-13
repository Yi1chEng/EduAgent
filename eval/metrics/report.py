"""把 metrics 字典转成 CSV / Markdown / LaTeX 表格,供论文使用。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


def to_csv(rows: list[Mapping[str, object]], path: str | Path, columns: list[str] | None = None) -> Path:
    """list[dict] → CSV。columns 不传则取第一行的 keys。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    cols = list(columns) if columns else list(rows[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return p


def to_markdown(rows: list[Mapping[str, object]], columns: list[str] | None = None) -> str:
    """list[dict] → GFM 表格字符串。"""
    if not rows:
        return ""
    cols = list(columns) if columns else list(rows[0].keys())
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join([head, sep, *body])


def to_latex(
    rows: list[Mapping[str, object]],
    columns: list[str] | None = None,
    caption: str = "",
    label: str = "",
) -> str:
    """简单 booktabs 风格 LaTeX 表(无依赖)。"""
    if not rows:
        return ""
    cols = list(columns) if columns else list(rows[0].keys())
    col_spec = "l" + "r" * (len(cols) - 1)
    head = " & ".join(_esc_latex(c) for c in cols) + r" \\"
    body = [
        " & ".join(_esc_latex(_fmt(r.get(c, ""))) for c in cols) + r" \\"
        for r in rows
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_esc_latex(caption)}}}" if caption else "",
        rf"\label{{{label}}}" if label else "",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        head,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(l for l in lines if l)


def _fmt(v: object) -> str:
    if isinstance(v, float):
        # 4 位小数对论文够用
        return f"{v:.4f}"
    return str(v)


def _esc_latex(s: str) -> str:
    repl = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "^": r"\^{}",
        "~": r"\~{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)
