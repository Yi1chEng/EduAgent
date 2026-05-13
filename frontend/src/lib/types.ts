// 后端响应数据类型

export interface Citation {
  index: number;
  source_file: string;
  heading_path: string;
  preview: string;
}

export interface ChatResponse {
  conversation_id: number;
  content: string;
  citations: Citation[];
  mermaid_code?: string | null;
  tool_results?: Record<string, unknown> | null;
}

export interface SessionListItem {
  session_id: string;
  title?: string | null;
  last_message: string;
  last_role: string;
  message_count: number;
  updated_at: string;
}

export interface MessageItem {
  id: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  mermaid_code?: string | null;
  tool_invocations?: ToolInvocation[];
  created_at: string;
}

export interface KnowledgeListItem {
  source_file: string;
  chunks_count: number;
  created_at: string;
}

// MCP 工具单次调用记录（SSE tool_result 事件的载荷，每次调用一条）
export interface ToolInvocation {
  name: string;
  args: Record<string, unknown>;
  status: "success" | "error";
  result?: string;
  error?: string;
}

// 前端运行时消息状态（用于流式增量更新）
export interface UIMessage {
  id?: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  mermaid_code?: string;
  tool_invocations?: ToolInvocation[];
  streaming?: boolean;
}

export type StreamEventName =
  | "citation"
  | "token"
  | "mermaid"
  | "tool_result"
  | "done"
  | "error";
