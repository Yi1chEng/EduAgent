import { useState } from "react";
import {
  MessageCirclePlus,
  Trash2,
  BookOpen,
  MessageSquare,
  Sparkles,
} from "lucide-react";
import { deleteSession } from "../lib/api";
import type { SessionListItem } from "../lib/types";

interface Props {
  sessions: SessionListItem[];
  activeSessionId: string;
  currentView: "chat" | "knowledge";
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onOpenKnowledge: () => void;
  onSessionsChange: () => void;
}

export function Sidebar({
  sessions,
  activeSessionId,
  currentView,
  onSelectSession,
  onNewSession,
  onOpenKnowledge,
  onSessionsChange,
}: Props) {
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const handleDelete = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("确认删除此会话？")) return;
    setDeletingId(sessionId);
    try {
      await deleteSession(sessionId);
      onSessionsChange();
      if (sessionId === activeSessionId) onNewSession();
    } catch (err) {
      alert(`删除失败: ${err}`);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <aside className="w-72 shrink-0 h-full flex flex-col border-r border-cream-300 bg-cream-50/60 backdrop-blur">
      {/* 品牌头 */}
      <div className="px-5 py-5 border-b border-cream-300 flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-terracotta-500 grid place-items-center shadow-warm">
          <Sparkles className="w-4 h-4 text-cream-50" strokeWidth={2.5} />
        </div>
        <div>
          <h1 className="font-serif text-lg font-semibold text-ink-700 leading-none">
            EduAgent
          </h1>
          <p className="text-xs text-ink-400 mt-0.5">教育智能助手</p>
        </div>
      </div>

      {/* 操作按钮 */}
      <div className="px-3 pt-3 pb-2 space-y-1">
        <button onClick={onNewSession} className="btn-primary w-full justify-start">
          <MessageCirclePlus className="w-4 h-4" /> 新会话
        </button>
        <button
          onClick={onOpenKnowledge}
          className={`btn-ghost w-full justify-start ${
            currentView === "knowledge"
              ? "bg-cream-200 text-ink-700"
              : ""
          }`}
        >
          <BookOpen className="w-4 h-4" /> 知识库
        </button>
      </div>

      {/* 分隔标题 */}
      <div className="px-5 pt-3 pb-1.5 text-xs font-medium uppercase tracking-wider text-ink-400">
        历史会话
      </div>

      {/* 会话列表 */}
      <nav className="flex-1 overflow-y-auto px-2 pb-3">
        {sessions.length === 0 && (
          <p className="text-sm text-ink-400 px-3 py-4 text-center">
            暂无会话
          </p>
        )}
        {sessions.map((s) => {
          const active = s.session_id === activeSessionId && currentView === "chat";
          return (
            <button
              key={s.session_id}
              onClick={() => onSelectSession(s.session_id)}
              className={`group w-full text-left px-3 py-2.5 rounded-md transition-all flex items-start gap-2 ${
                active
                  ? "bg-terracotta-500/10 border border-terracotta-500/20"
                  : "hover:bg-cream-200 border border-transparent"
              }`}
            >
              <MessageSquare
                className={`w-4 h-4 mt-0.5 shrink-0 ${
                  active ? "text-terracotta-600" : "text-ink-400"
                }`}
              />
              <div className="flex-1 min-w-0">
                <p
                  className={`text-sm leading-snug truncate ${
                    active ? "text-ink-700 font-medium" : "text-ink-600"
                  }`}
                  title={s.last_message || ""}
                >
                  {s.title || s.last_message || "(空会话)"}
                </p>
                <p className="text-xs text-ink-400 mt-0.5">
                  {s.message_count} 条 · {formatTime(s.updated_at)}
                </p>
              </div>
              <button
                onClick={(e) => handleDelete(s.session_id, e)}
                className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-cream-300 text-ink-400 hover:text-terracotta-700"
                disabled={deletingId === s.session_id}
                title="删除会话"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </button>
          );
        })}
      </nav>

      <footer className="px-5 py-3 border-t border-cream-300 text-xs text-ink-400">
        基于 Claude 设计风格 · v0.1
      </footer>
    </aside>
  );
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    const now = new Date();
    const sameDay = d.toDateString() === now.toDateString();
    if (sameDay) return d.toTimeString().slice(0, 5);
    return `${d.getMonth() + 1}/${d.getDate()}`;
  } catch {
    return "";
  }
}
