"""MCP Server 入口：基于 FastMCP，stdio 传输模式。"""

import logging

from mcp.server.fastmcp import FastMCP

from mermaid import register_mermaid
from wechat import register_wechat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("eduagent-mcp-tools")

register_mermaid(mcp)
register_wechat(mcp)
logger.info("MCP 工具已注册：mermaid, wechat")


if __name__ == "__main__":
    mcp.run(transport="stdio")
