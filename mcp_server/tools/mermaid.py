"""Mermaid 图表生成工具：调用 LLM 生成高质量 Mermaid DSL。"""

import logging
import os
import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

PROMPT_TEMPLATE = """你是 Mermaid 图表专家。根据以下完整知识描述生成一段**丰富、详细**的 Mermaid {diagram_type} DSL 代码。

【硬性语法规则】
1. 只返回纯 Mermaid 代码，不要任何 Markdown 围栏（不要 ```mermaid```），不要解释文字
2. **节点 ID 必须是纯英文字母 / 数字 / 下划线**，例如 A、B1、node_root；**绝对不能在 ID 位置直接写中文或空格**
3. 中文标签只能写在节点形状内：A[中文标签]、B((圆形标签))、C{{菱形标签}}
4. **标签内若含 `(`、`)`、`,`、`;`、`:`、`&`、`<`、`>`、`/`、`\` 等特殊字符，必须用双引号整体包裹**：
   - 错误：`Run_Method[run()方法]`、`Init[__init__方法]`
   - 正确：`Run_Method["run()方法"]`、`Init["__init__方法"]`
   - 错误：`Cond{{x > 0?}}`  正确：`Cond{{"x > 0?"}}`
5. 边的语法：A --> B 或 A -->|关系说明| B
6. flowchart 示例（含子图与关系标注）：
   flowchart TD
     subgraph PreparePhase[准备阶段]
       A[源任务数据] --> B[预训练大模型]
     end
     B -->|提取通用特征| C[特征表示]
     C --> D{{是否冻结}}
     D -->|是| E[冻结浅层]
     D -->|否| F[全网络微调]
     E --> G[添加任务头]
     F --> G
     G --> H[目标任务训练]
7. mindmap 示例（多层级）：
   mindmap
     root((迁移学习))
       Source[源域]
         data[源数据]
         model[预训练模型]
       Strategy[迁移策略]
         featTransfer[特征迁移]
         paramTransfer[参数迁移]
         relTransfer[关系迁移]
       Target[目标域]
         finetune[微调]
         frozen[冻结+头部]

【丰富度要求 — 极其重要】
- **必须从描述中尽可能多地提取概念、子步骤、分类、条件、关系**，不要简化成 4-5 个节点
- **节点数目标 10-20 个**（mindmap 可以更多，flowchart 控制在 8-15 个）
- **必须使用层级**：flowchart 用 subgraph 分组主流程；mindmap 至少 3 层（root → 主分支 → 子节点）
- **必须标注关系**：边上用 `-->|动作/条件|` 说明转移条件
- **覆盖完整逻辑**：包含输入、处理步骤、判断分支、输出、异常情况（如适用）
- 标签简洁但具体，避免"步骤1/步骤2"这种空话

【描述】
{description}
"""


def _get_llm() -> ChatOpenAI:
    """延迟初始化 LLM 客户端，从环境变量读取配置。"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
            model=os.environ["LLM_MODEL_NAME"],
            temperature=0.3,
        )
    return _llm


def _strip_fences(text: str) -> str:
    """剥离可能存在的 ```mermaid ... ``` 围栏。"""
    text = text.strip()
    match = re.search(r"```(?:mermaid)?\s*\n([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


# 触发引号包裹的危险字符（在 mermaid 标签内会引起解析错误）
_DANGER_CHARS = re.compile(r'[()<>;,&/\\:]')


def _quote_label(content: str) -> str:
    """将标签内容用双引号包裹（若已包裹则不变），同时转义内部双引号。"""
    s = content.strip()
    if s.startswith('"') and s.endswith('"'):
        return s
    return '"' + s.replace('"', "'") + '"'


def _sanitize_labels(code: str) -> str:
    """自动修复 LLM 生成的常见 Mermaid 语法问题：
    - `Node[run()方法]`           → `Node["run()方法"]`
    - `Node{x > 0?}`              → `Node{"x > 0?"}`
    - `Node((含 / 或 () 的文本))` → `Node(("..."))`
    - `Node([流程])` / `Node[[子程序]]` 等复合形状同样处理
    仅当内容含危险字符且未被引号包裹时才包裹。
    """

    def repl_bracket(m: re.Match) -> str:
        inner = m.group(1)
        if _DANGER_CHARS.search(inner):
            return "[" + _quote_label(inner) + "]"
        return m.group(0)

    def repl_double_brace(m: re.Match) -> str:
        inner = m.group(1)
        if _DANGER_CHARS.search(inner):
            return "{{" + _quote_label(inner) + "}}"
        return m.group(0)

    def repl_brace(m: re.Match) -> str:
        inner = m.group(1)
        if _DANGER_CHARS.search(inner):
            return "{" + _quote_label(inner) + "}"
        return m.group(0)

    def repl_double_paren(m: re.Match) -> str:
        inner = m.group(1)
        if _DANGER_CHARS.search(inner):
            return "((" + _quote_label(inner) + "))"
        return m.group(0)

    # ((...)) 圆形（先于单括号）
    code = re.sub(r"\(\(([^()\n]+)\)\)", repl_double_paren, code)
    # {{...}} 六边形（先于 {...}）
    code = re.sub(r"\{\{([^{}\n]+)\}\}", repl_double_brace, code)
    # [...] 方形 / 复合形状内层
    code = re.sub(r"\[([^\[\]\n]+)\]", repl_bracket, code)
    # {...} 菱形
    code = re.sub(r"\{([^{}\n]+)\}", repl_brace, code)
    return code


def register_mermaid(mcp: FastMCP) -> None:
    """注册 generate_mermaid 工具到 MCP Server。"""

    @mcp.tool()
    async def generate_mermaid(description: str, diagram_type: str = "flowchart") -> str:
        """调用 LLM 生成 Mermaid 图表 DSL 代码。

        Args:
            description: 自然语言描述，可包含主题、关键节点、关系等。
            diagram_type: 图表类型，flowchart / mindmap / timeline / sequence / classDiagram / erDiagram。

        Returns:
            纯 Mermaid DSL 代码（无 Markdown 围栏）。
        """
        prompt = PROMPT_TEMPLATE.format(
            diagram_type=diagram_type,
            description=description,
        )
        try:
            response = await _get_llm().ainvoke([HumanMessage(content=prompt)])
            raw = _strip_fences(response.content)
            return _sanitize_labels(raw)
        except Exception as e:
            logger.error(f"generate_mermaid 失败: {e}")
            return f"flowchart TD\n    A[生成失败: {e}]"
