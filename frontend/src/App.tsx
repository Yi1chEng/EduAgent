import { useEffect, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { ChatView } from "./components/ChatView";
import { KnowledgePanel } from "./components/KnowledgePanel";
import { listSessions } from "./lib/api";
import type { SessionListItem } from "./lib/types";

type View = "chat" | "knowledge";

export default function App() {
  const [view, setView] = useState<View>("chat");
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>(() => newSessionId());

  const refreshSessions = async () => {
    try {
      const list = await listSessions();
      setSessions(list);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    refreshSessions();
  }, []);

  const handleNewSession = () => {
    const id = newSessionId();
    setActiveSessionId(id);
    setView("chat");
  };

  return (
    <div className="h-screen w-screen flex bg-cream-100 text-ink-700">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        currentView={view}
        onSelectSession={(id) => {
          setActiveSessionId(id);
          setView("chat");
        }}
        onNewSession={handleNewSession}
        onOpenKnowledge={() => setView("knowledge")}
        onSessionsChange={refreshSessions}
      />
      <main className="flex-1 flex flex-col overflow-hidden">
        {view === "chat" ? (
          <ChatView
            sessionId={activeSessionId}
            onConversationUpdate={refreshSessions}
          />
        ) : (
          <KnowledgePanel />
        )}
      </main>
    </div>
  );
}

function newSessionId(): string {
  return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
