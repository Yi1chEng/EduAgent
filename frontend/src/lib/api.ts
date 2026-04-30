import type {
  ChatResponse,
  KnowledgeListItem,
  MessageItem,
  SessionListItem,
  StreamEventName,
} from "./types";

const BASE = "/api";

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res
      .json()
      .then((j) => j.detail ?? res.statusText)
      .catch(() => res.statusText);
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

// ===== 对话 =====

export interface ChatPayload {
  query: string;
  session_id: string;
  need_visualization?: boolean;
  need_dispatch?: boolean;
}

export async function postChat(payload: ChatPayload): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow<ChatResponse>(res);
}

/** SSE 流式对话：通过 fetch + ReadableStream 读取（POST 不支持原生 EventSource）。 */
export async function streamChat(
  payload: ChatPayload,
  onEvent: (name: StreamEventName, data: string) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`流式连接失败: ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE 以空行分割 events，行尾可能是 \n 或 \r\n（sse-starlette 用 CRLF）
    const sep = /\r?\n\r?\n/;
    let match: RegExpExecArray | null;
    while ((match = sep.exec(buffer))) {
      const rawEvent = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      let eventName: StreamEventName = "token";
      const dataLines: string[] = [];
      for (const line of rawEvent.split(/\r?\n/)) {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim() as StreamEventName;
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).replace(/^ /, ""));
        }
      }
      if (dataLines.length > 0) {
        onEvent(eventName, dataLines.join("\n"));
      }
    }
  }
}

// ===== 会话 =====

export async function listSessions(): Promise<SessionListItem[]> {
  const res = await fetch(`${BASE}/sessions`);
  const data = await jsonOrThrow<{ sessions: SessionListItem[] }>(res);
  return data.sessions;
}

export async function getSessionMessages(sessionId: string): Promise<MessageItem[]> {
  const res = await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/messages`);
  const data = await jsonOrThrow<{ messages: MessageItem[] }>(res);
  return data.messages;
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  await jsonOrThrow(res);
}

// ===== 知识库 =====

export async function listKnowledge(): Promise<KnowledgeListItem[]> {
  const res = await fetch(`${BASE}/knowledge/list`);
  const data = await jsonOrThrow<{ documents: KnowledgeListItem[] }>(res);
  return data.documents;
}

export async function uploadKnowledge(file: File): Promise<{ chunks_count: number }> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/knowledge/upload`, { method: "POST", body: fd });
  return jsonOrThrow<{ chunks_count: number; message: string; source_file: string }>(res);
}

export async function deleteKnowledge(sourceFile: string): Promise<void> {
  const res = await fetch(`${BASE}/knowledge/${encodeURIComponent(sourceFile)}`, {
    method: "DELETE",
  });
  await jsonOrThrow(res);
}

// ===== 反馈 =====

export async function postFeedback(
  conversationId: number,
  rating: 1 | -1,
  comment?: string
): Promise<void> {
  const res = await fetch(`${BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, rating, comment }),
  });
  await jsonOrThrow(res);
}
