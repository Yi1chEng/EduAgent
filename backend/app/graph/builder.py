"""构建 StateGraph 并编译为可执行图。

线性结构：rag_retriever → generator → END。
工具调用由 generator 节点内部 LLM 自主决策（bind_tools），不再有独立 dispatcher。
"""

import logging

from langgraph.graph import END, StateGraph

from app.graph.nodes import generator_node, rag_retriever_node
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """构建 EduAgent 的 LangGraph 状态图。"""
    graph = StateGraph(AgentState)

    graph.add_node("rag_retriever", rag_retriever_node)
    graph.add_node("generator", generator_node)

    graph.set_entry_point("rag_retriever")
    graph.add_edge("rag_retriever", "generator")
    graph.add_edge("generator", END)

    compiled = graph.compile()
    logger.info("LangGraph 状态图编译完成（rag_retriever → generator → END）")
    return compiled


app_graph = build_graph()
