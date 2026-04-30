"""构建 StateGraph 并编译为可执行图。"""

import logging

from langgraph.graph import END, StateGraph

from app.graph.edges import route_after_generation
from app.graph.nodes import dispatcher_node, generator_node, rag_retriever_node
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """构建 EduAgent 的 LangGraph 状态图。

    图结构：
        rag_retriever → generator → route_after_generation
                                     ├─ dispatcher → END
                                     └─ END
    """
    graph = StateGraph(AgentState)

    graph.add_node("rag_retriever", rag_retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("dispatcher", dispatcher_node)

    graph.set_entry_point("rag_retriever")
    graph.add_edge("rag_retriever", "generator")
    graph.add_conditional_edges(
        "generator",
        route_after_generation,
        {"dispatcher": "dispatcher", "end": END},
    )
    graph.add_edge("dispatcher", END)

    compiled = graph.compile()
    logger.info("LangGraph 状态图编译完成")
    return compiled


app_graph = build_graph()
