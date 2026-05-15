"""工具注册表：所有可调用工具的元数据中心。

设计原则：
- 所有工具统一保管，不区分"内置 / 外部"。
- 所有工具**默认禁用**，由用户在前端按需勾选启用。
- registry 只保存"工具有什么、风险多大"等纯元数据，不持有 MCP 连接。
- 实际的 MCP 工具对象由 graph.nodes._get_mcp_tools 加载；本模块负责把"用户启用集合"
  与"已加载工具"做交集，得到一次请求实际暴露给 LLM 的工具列表。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    """风险等级。前端据此提示用户、决定是否需要确认。

    - LOW：纯只读 / 本地无副作用（如生成图表 DSL）。
    - MEDIUM：会向外部网络/服务发请求，但不会修改远端持久状态（如 fetch、读 GitHub）。
    - HIGH：会改写远端或本地持久状态（如发送企业微信消息、写文件、创建 PR）。
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class ToolMetadata:
    """工具元数据。"""

    id: str
    display_name: str
    description: str
    risk_level: RiskLevel
    # 该工具实际由哪个 MCP server 提供。仅作展示和调试用，不参与调度。
    mcp_server: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "risk_level": self.risk_level.value,
            "mcp_server": self.mcp_server,
        }


# 所有工具的统一注册表。键为工具的 MCP 名称，必须与 MCP server 暴露的 tool name 一致。
# 全部默认禁用，由用户主动选择启用。
TOOLS_REGISTRY: dict[str, ToolMetadata] = {
    "generate_mermaid": ToolMetadata(
        id="generate_mermaid",
        display_name="生成图表",
        description="将一段知识内容转换成 Mermaid DSL 图表（流程图 / 思维导图 / 时序图等）。"
        "当用户希望可视化、画图、做思维导图、看流程图时调用。",
        risk_level=RiskLevel.LOW,
        mcp_server="eduagent",
    ),
    "send_wechat": ToolMetadata(
        id="send_wechat",
        display_name="推送企业微信",
        description="把当前回答内容推送到预先配置的企业微信群机器人。"
        "当用户明确要求推送/发送/通知到企业微信时调用。",
        risk_level=RiskLevel.HIGH,
        mcp_server="eduagent",
    ),
    "github": ToolMetadata(
        id="github",
        display_name="GitHub",
        description="查询 GitHub 仓库、Issues、PR、代码搜索。当用户问到代码仓库、开源项目、"
        "issue 或代码示例时调用。",
        risk_level=RiskLevel.MEDIUM,
        mcp_server="github",
    ),
    "fetch": ToolMetadata(
        id="fetch",
        display_name="网页抓取",
        description="抓取指定 URL 的网页内容。当用户给了链接、希望看在线资源、需要最新外部信息时调用。",
        risk_level=RiskLevel.MEDIUM,
        mcp_server="fetch",
    ),
    "filesystem": ToolMetadata(
        id="filesystem",
        display_name="文件系统",
        description="读写本地上传目录中的文件。当用户希望保存笔记、读取已上传文件原文时调用。",
        risk_level=RiskLevel.HIGH,
        mcp_server="filesystem",
    ),
}


def get_default_enabled_map() -> dict[str, bool]:
    """返回所有工具的默认启用状态字典（全部 False）。"""
    return {tid: False for tid in TOOLS_REGISTRY}


def get_available_tools(
    loaded_tools: dict[str, Any],
    enabled_tools: dict[str, bool] | None,
) -> list[Any]:
    """根据用户启用集合 + 实际加载到的 MCP 工具，返回最终暴露给 LLM 的工具列表。

    入参：
        loaded_tools: 通过 MCP client 加载到的全部工具对象，键为 tool name。
        enabled_tools: 用户在本次请求中显式传入的启用集合。
                       未在该字典中显式置 True 的工具一律视为未启用。

    返回：
        list[BaseTool]：可直接 bind_tools(tools) 给 LLM 的工具对象列表。
    """
    enabled_tools = enabled_tools or {}
    chosen: list[Any] = []
    for tool_id in TOOLS_REGISTRY:
        if not enabled_tools.get(tool_id, False):
            continue
        tool = loaded_tools.get(tool_id)
        if tool is None:
            # 注册了但 MCP server 没起来（如外部 server 未配置环境变量）→ 静默跳过
            continue
        chosen.append(tool)
    return chosen


def is_tool_enabled(tool_id: str, enabled_tools: dict[str, bool] | None) -> bool:
    """单个工具是否在本次请求中启用。未注册或未显式启用一律视为未启用。"""
    if tool_id not in TOOLS_REGISTRY:
        return False
    if enabled_tools is None:
        return False
    return enabled_tools.get(tool_id, False)
