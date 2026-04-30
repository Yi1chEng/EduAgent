"""AgentState 定义：LangGraph 多 Agent 编排的全局状态。"""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """LangGraph 状态定义，所有节点共享此状态进行数据传递。

    Attributes:
        messages: 对话历史列表，包含 HumanMessage/AIMessage 等。
        query: 用户当前问题。
        retrieved_docs: RAG 检索到的文档列表。
        citations: 引用来源列表。
        generated_content: LLM 生成的回答内容。
        tool_results: 工具调用结果字典。
        need_visualization: 是否需要生成图表。
        need_dispatch: 是否需要推送消息。
        mermaid_code: 生成的 Mermaid DSL 代码。
        session_id: 当前会话 ID。
    """

    messages: list[Any]
    query: str
    retrieved_docs: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    generated_content: str
    tool_results: dict[str, Any]
    need_visualization: bool
    need_dispatch: bool
    mermaid_code: str
    session_id: str
