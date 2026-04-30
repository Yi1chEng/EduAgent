"""条件边与路由逻辑：控制 LangGraph 图的分支走向。"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from app.graph.state import AgentState

logger = logging.getLogger(__name__)

EXTERNAL_MCP_CONFIG_PATH = "/mcp_servers.json"


@lru_cache(maxsize=1)
def _has_external_mcp_servers() -> bool:
    """检查是否配置了外部 MCP server（缓存结果）。"""
    path = Path(EXTERNAL_MCP_CONFIG_PATH)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        servers = data.get("mcpServers", {})
        return any(not name.startswith("_") for name in servers)
    except Exception:
        return False


def route_after_generation(state: AgentState) -> Literal["dispatcher", "end"]:
    """决定 generator 后是否进入 dispatcher。

    进入 dispatcher 的条件：
    - 显式标志 need_visualization / need_dispatch
    - 或配置了外部 MCP server（让 LLM 自主决定是否调用）
    """
    need_viz = state.get("need_visualization", False)
    need_dispatch = state.get("need_dispatch", False)

    if need_viz or need_dispatch or _has_external_mcp_servers():
        return "dispatcher"
    return "end"
