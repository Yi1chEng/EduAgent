"""工具注册表：所有可调用工具（内置 + 外部 MCP）的元数据中心。

设计原则：
- registry 只保存"工具有什么、风险多大、默认是否启用"等纯元数据，不持有 MCP 连接。
- 实际的 MCP 工具对象由 graph.nodes._get_mcp_tools 加载；本模块负责把"用户启用集合"
  与"已加载工具"做交集，得到一次请求实际暴露给 LLM 的工具列表。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToolCategory(str, Enum):
    """工具分类。internal = 项目内置 MCP server；external = 第三方 MCP server。"""

    INTERNAL = "internal"
    EXTERNAL = "external"


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
    category: ToolCategory
    risk_level: RiskLevel
    default_enabled: bool
    # 该工具实际由哪个 MCP server 提供。用于 _get_mcp_tools 加载时做来源校验。
    mcp_server: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category.value,
            "risk_level": self.risk_level.value,
            "default_enabled": self.default_enabled,
            "mcp_server": self.mcp_server,
        }


# 所有工具的统一注册表。键为工具的 MCP 名称，必须与 MCP server 暴露的 tool name 一致。
TOOLS_REGISTRY: dict[str, ToolMetadata] = {
    # ===== 内置工具（默认启用） =====
    "generate_mermaid": ToolMetadata(
        id="generate_mermaid",
        display_name="生成图表",
        description="将一段知识内容转换成 Mermaid DSL 图表（流程图 / 思维导图 / 时序图等）。"
        "当用户希望可视化、画图、做思维导图、看流程图时调用。",
        category=ToolCategory.INTERNAL,
        risk_level=RiskLevel.LOW,
        default_enabled=True,
        mcp_server="eduagent",
    ),
    "send_wechat": ToolMetadata(
        id="send_wechat",
        display_name="推送企业微信",
        description="把当前回答内容推送到预先配置的企业微信群机器人。"
        "当用户明确要求推送/发送/通知到企业微信时调用。",
        category=ToolCategory.INTERNAL,
        risk_level=RiskLevel.HIGH,
        default_enabled=True,
        mcp_server="eduagent",
    ),
    # ===== 外部工具（默认禁用，用户按需打开） =====
    "github": ToolMetadata(
        id="github",
        display_name="GitHub",
        description="查询 GitHub 仓库、Issues、PR、代码搜索。当用户问到代码仓库、开源项目、"
        "issue 或代码示例时调用。",
        category=ToolCategory.EXTERNAL,
        risk_level=RiskLevel.MEDIUM,
        default_enabled=False,
        mcp_server="github",
    ),
    "fetch": ToolMetadata(
        id="fetch",
        display_name="网页抓取",
        description="抓取指定 URL 的网页内容。当用户给了链接、希望看在线资源、需要最新外部信息时调用。",
        category=ToolCategory.EXTERNAL,
        risk_level=RiskLevel.MEDIUM,
        default_enabled=False,
        mcp_server="fetch",
    ),
    "filesystem": ToolMetadata(
        id="filesystem",
        display_name="文件系统",
        description="读写本地上传目录中的文件。当用户希望保存笔记、读取已上传文件原文时调用。",
        category=ToolCategory.EXTERNAL,
        risk_level=RiskLevel.HIGH,
        default_enabled=False,
        mcp_server="filesystem",
    ),
}


def get_default_enabled_map() -> dict[str, bool]:
    """返回所有工具的默认启用状态字典，用于前端首次加载。"""
    return {tid: meta.default_enabled for tid, meta in TOOLS_REGISTRY.items()}


def get_available_tools(
    loaded_tools: dict[str, Any],
    enabled_tools: dict[str, bool] | None,
) -> list[Any]:
    """根据用户启用集合 + 实际加载到的 MCP 工具，返回最终暴露给 LLM 的工具列表。

    入参：
        loaded_tools: 通过 MCP client 加载到的全部工具对象，键为 tool name。
        enabled_tools: 用户在本次请求中显式传入的启用集合。None / 缺失键 → 用 registry 默认值。

    返回：
        list[BaseTool]：可直接 bind_tools(tools) 给 LLM 的工具对象列表。
    """
    enabled_tools = enabled_tools or {}
    chosen: list[Any] = []
    for tool_id, meta in TOOLS_REGISTRY.items():
        is_enabled = enabled_tools.get(tool_id, meta.default_enabled)
        if not is_enabled:
            continue
        tool = loaded_tools.get(tool_id)
        if tool is None:
            # 注册了但 MCP server 没起来（如外部 server 未配置环境变量）→ 静默跳过
            continue
        chosen.append(tool)
    return chosen


def is_tool_enabled(tool_id: str, enabled_tools: dict[str, bool] | None) -> bool:
    """单个工具是否在本次请求中启用。未注册的工具一律视为未启用。"""
    meta = TOOLS_REGISTRY.get(tool_id)
    if meta is None:
        return False
    if enabled_tools is None:
        return meta.default_enabled
    return enabled_tools.get(tool_id, meta.default_enabled)
