import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Send,
  ImageIcon,
  Bell,
  Plug,
  ThumbsUp,
  ThumbsDown,
  Sparkles,
  User,
  Loader2,
  Wrench,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { MermaidRenderer } from "./MermaidRenderer";
import {
  getSessionMessages,
  postFeedback,
  streamChat,
} from "../lib/api";
import type { Citation, ToolInvocation, UIMessage } from "../lib/types";

interface Props {
  sessionId: string;
  onConversationUpdate: () => void;
}

export function ChatView({ sessionId, onConversationUpdate }: Props) {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [needViz, setNeedViz] = useState(false);
  const [needDispatch, setNeedDispatch] = useState(false);
  const [allowExternalTools, setAllowExternalTools] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 切换会话时加载历史
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const list = await getSessionMessages(sessionId);
        if (cancelled) return;
        setMessages(
          list.map((m) => ({
            id: m.id,
            role: m.role,
            content: m.content,
            citations: m.citations,
          }))
        );
      } catch {
        // 新会话——404，直接清空
        if (!cancelled) setMessages([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  // 自动滚到底
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const handleSend = async () => {
    const query = input.trim();
    if (!query || streaming) return;
    setInput("");

    // 乐观更新
    const userMsg: UIMessage = { role: "user", content: query, citations: [] };
    const assistantMsg: UIMessage = {
      role: "assistant",
      content: "",
      citations: [],
      streaming: true,
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    try {
      await streamChat(
        {
          query,
          session_id: sessionId,
          need_visualization: needViz,
          need_dispatch: needDispatch,
          allow_external_tools: allowExternalTools,
        },
        (event, data) => {
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (!last || last.role !== "assistant") return prev;
            const updated = { ...last };
            if (event === "citation") {
              try {
                updated.citations = JSON.parse(data) as Citation[];
              } catch {
                /* noop */
              }
            } else if (event === "token") {
              updated.content += data;
            } else if (event === "mermaid") {
              updated.mermaid_code = data;
            } else if (event === "tool_result") {
              try {
                const inv = JSON.parse(data) as ToolInvocation;
                updated.tool_invocations = [
                  ...(updated.tool_invocations ?? []),
                  inv,
                ];
              } catch {
                /* noop */
              }
            } else if (event === "done") {
              updated.id = parseInt(data, 10) || undefined;
              updated.streaming = false;
            } else if (event === "error") {
              updated.content += `\n\n⚠️ 错误：${data}`;
              updated.streaming = false;
            }
            return [...prev.slice(0, -1), updated];
          });
        }
      );
    } catch (err) {
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (!last) return prev;
        return [
          ...prev.slice(0, -1),
          { ...last, content: last.content + `\n\n⚠️ 请求失败：${err}`, streaming: false },
        ];
      });
    } finally {
      setStreaming(false);
      onConversationUpdate();
    }
  };

  const handleFeedback = async (msg: UIMessage, rating: 1 | -1) => {
    if (!msg.id) return;
    try {
      await postFeedback(msg.id, rating);
    } catch (e) {
      alert(`反馈失败: ${e}`);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full">
      <header className="px-8 py-4 border-b border-cream-300 bg-cream-50/50 backdrop-blur">
        <h2 className="font-serif text-lg font-semibold text-ink-700">
          {messages.length > 0
            ? truncate(messages[0].content, 40)
            : "新对话"}
        </h2>
        <p className="text-xs text-ink-400 mt-0.5">
          会话 ID: <code className="font-mono">{sessionId}</code>
        </p>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-8">
        <div className="max-w-3xl mx-auto space-y-8">
          {messages.length === 0 && <EmptyState />}
          {messages.map((m, i) => (
            <MessageBubble
              key={i}
              message={m}
              onFeedback={(r) => handleFeedback(m, r)}
            />
          ))}
        </div>
      </div>

      <div className="px-6 pb-6 pt-3 bg-gradient-to-t from-cream-100 via-cream-100 to-transparent">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-center gap-2 mb-2 text-xs flex-wrap">
            <Toggle
              icon={<ImageIcon className="w-3.5 h-3.5" />}
              label="生成图表"
              active={needViz}
              onChange={setNeedViz}
            />
            <Toggle
              icon={<Bell className="w-3.5 h-3.5" />}
              label="推送企业微信"
              active={needDispatch}
              onChange={setNeedDispatch}
            />
            <Toggle
              icon={<Plug className="w-3.5 h-3.5" />}
              label="允许外部工具"
              active={allowExternalTools}
              onChange={setAllowExternalTools}
            />
          </div>
          <div className="card flex items-end gap-2 p-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              rows={2}
              placeholder="问点什么…（Enter 发送，Shift+Enter 换行）"
              className="flex-1 resize-none bg-transparent outline-none px-3 py-2 text-ink-700 placeholder:text-ink-400 text-[15px] leading-relaxed"
              disabled={streaming}
            />
            <button
              onClick={handleSend}
              disabled={streaming || !input.trim()}
              className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {streaming ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({
  message,
  onFeedback,
}: {
  message: UIMessage;
  onFeedback: (rating: 1 | -1) => void;
}) {
  const isUser = message.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`shrink-0 w-8 h-8 rounded-full grid place-items-center ${
          isUser
            ? "bg-cream-300 text-ink-600"
            : "bg-terracotta-500 text-cream-50 shadow-warm"
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4" />
        ) : (
          <Sparkles className="w-4 h-4" strokeWidth={2.5} />
        )}
      </div>
      <div className={`flex-1 min-w-0 ${isUser ? "text-right" : ""}`}>
        <div
          className={`inline-block max-w-full text-left ${
            isUser
              ? "bg-terracotta-500 text-cream-50 rounded-2xl rounded-tr-sm px-4 py-2.5 shadow-warm"
              : "text-ink-700"
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
          ) : (
            <>
              <div className="markdown-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content || (message.streaming ? "思考中…" : "")}
                </ReactMarkdown>
              </div>
              {message.mermaid_code && (
                <MermaidRenderer code={message.mermaid_code} />
              )}
              {message.tool_invocations && message.tool_invocations.length > 0 && (
                <ToolInvocationsList invocations={message.tool_invocations} />
              )}
              {message.citations.length > 0 && (
                <CitationsList citations={message.citations} />
              )}
              {!message.streaming && message.id && (
                <div className="flex gap-1 mt-3 -ml-2">
                  <button
                    onClick={() => onFeedback(1)}
                    className="btn-icon"
                    title="有帮助"
                  >
                    <ThumbsUp className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => onFeedback(-1)}
                    className="btn-icon"
                    title="无帮助"
                  >
                    <ThumbsDown className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function ToolInvocationsList({ invocations }: { invocations: ToolInvocation[] }) {
  return (
    <div className="mt-4 space-y-2">
      <div className="flex items-center gap-1.5 text-xs font-medium text-ink-400">
        <Wrench className="w-3.5 h-3.5" />
        <span>工具调用 ({invocations.length})</span>
      </div>
      <div className="space-y-2">
        {invocations.map((inv, i) => (
          <ToolInvocationCard key={i} invocation={inv} />
        ))}
      </div>
    </div>
  );
}

function ToolInvocationCard({ invocation }: { invocation: ToolInvocation }) {
  const isError = invocation.status === "error";
  const argsPreview = formatArgsOneLine(invocation.args);
  const body = isError ? invocation.error ?? "未知错误" : invocation.result ?? "";

  return (
    <details
      className={`group rounded-lg border text-sm ${
        isError
          ? "border-red-200 bg-red-50/40"
          : "border-cream-300 bg-cream-50/60"
      }`}
    >
      <summary className="cursor-pointer select-none flex items-center gap-2 px-3 py-2">
        {isError ? (
          <XCircle className="w-4 h-4 text-red-500 shrink-0" />
        ) : (
          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
        )}
        <code className="font-mono text-[13px] text-ink-700 font-medium">
          {invocation.name}
        </code>
        <span className="text-xs text-ink-400 truncate flex-1 min-w-0">
          {argsPreview}
        </span>
      </summary>
      <div className="px-3 pb-3 pt-1 space-y-2 border-t border-cream-300/60">
        {Object.keys(invocation.args).length > 0 && (
          <div>
            <div className="text-[11px] uppercase tracking-wide text-ink-400 mb-1">
              参数
            </div>
            <pre className="text-xs bg-cream-100/80 rounded p-2 overflow-x-auto text-ink-600 whitespace-pre-wrap break-all">
              {JSON.stringify(invocation.args, null, 2)}
            </pre>
          </div>
        )}
        <div>
          <div className="text-[11px] uppercase tracking-wide text-ink-400 mb-1">
            {isError ? "错误" : "返回摘要"}
          </div>
          <pre
            className={`text-xs rounded p-2 overflow-x-auto whitespace-pre-wrap break-all ${
              isError
                ? "bg-red-50 text-red-700"
                : "bg-cream-100/80 text-ink-600"
            }`}
          >
            {body || "(空)"}
          </pre>
        </div>
      </div>
    </details>
  );
}

function formatArgsOneLine(args: Record<string, unknown>): string {
  const keys = Object.keys(args);
  if (keys.length === 0) return "(无参数)";
  const parts = keys.slice(0, 3).map((k) => {
    const v = args[k];
    const repr =
      typeof v === "string"
        ? v.length > 30
          ? `"${v.slice(0, 30)}…"`
          : `"${v}"`
        : JSON.stringify(v);
    return `${k}=${repr}`;
  });
  if (keys.length > 3) parts.push("…");
  return parts.join(", ");
}

function CitationsList({ citations }: { citations: Citation[] }) {
  return (
    <details className="mt-4 group">
      <summary className="cursor-pointer text-xs font-medium text-ink-400 hover:text-terracotta-600 select-none">
        引用来源 ({citations.length})
      </summary>
      <ol className="mt-2 space-y-1.5 text-sm">
        {citations.map((c) => (
          <li
            key={c.index}
            className="flex gap-2 items-start text-ink-500 leading-relaxed"
          >
            <span className="citation-tag mt-0.5 shrink-0">{c.index}</span>
            <div className="flex-1 min-w-0">
              <div className="text-ink-600 font-medium text-[13px]">
                {c.heading_path}
              </div>
              <div className="text-xs text-ink-400 mt-0.5">{c.source_file}</div>
              <div className="text-xs text-ink-500 mt-1 line-clamp-2">
                {c.preview}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </details>
  );
}

function Toggle({
  icon,
  label,
  active,
  onChange,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      onClick={() => onChange(!active)}
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border transition ${
        active
          ? "bg-terracotta-500/10 text-terracotta-700 border-terracotta-500/30"
          : "text-ink-400 border-cream-300 hover:border-ink-400/40"
      }`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

function EmptyState() {
  return (
    <div className="text-center py-16">
      <div className="inline-flex w-12 h-12 rounded-2xl bg-terracotta-500 text-cream-50 items-center justify-center shadow-warm mb-4">
        <Sparkles className="w-6 h-6" strokeWidth={2.5} />
      </div>
      <h3 className="font-serif text-2xl text-ink-700 mb-2">从一个问题开始</h3>
      <p className="text-ink-400 text-sm max-w-md mx-auto leading-relaxed">
        基于上传的教材内容，我会为你解答疑问、生成图表，
        <br />
        并附上引用来源。
      </p>
    </div>
  );
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}
