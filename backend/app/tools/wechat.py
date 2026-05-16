"""企业微信 Webhook 推送工具。"""

import logging
import os

import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_wechat(mcp: FastMCP) -> None:
    """注册 send_wechat 工具到 MCP Server。"""

    @mcp.tool()
    async def send_wechat(content: str, msg_type: str = "markdown") -> str:
        """通过企业微信 Webhook 推送消息。

        Args:
            content: 消息内容（支持 Markdown 格式）。
            msg_type: 消息类型，text 或 markdown。

        Returns:
            发送状态描述。
        """
        webhook_url = os.environ.get("WECHAT_WEBHOOK_URL", "")
        if not webhook_url:
            return "发送失败: 未配置 WECHAT_WEBHOOK_URL"

        if msg_type not in ("text", "markdown"):
            return f"不支持的消息类型: {msg_type}"

        payload = {
            "msgtype": msg_type,
            msg_type: {"content": content},
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(webhook_url, json=payload, timeout=10.0)
            if resp.status_code != 200:
                return f"HTTP 失败: {resp.status_code}"
            data = resp.json()
            if data.get("errcode") == 0:
                return "发送成功"
            return f"发送失败: {data.get('errmsg', '')}"
        except Exception as e:
            logger.error(f"企业微信推送失败: {e}")
            return f"发送失败: {e}"
