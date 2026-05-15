import { useEffect, useState } from "react";
import { ChevronDown, Wrench, AlertTriangle } from "lucide-react";
import { getToolsConfig } from "../lib/api";
import type { RiskLevel, ToolConfig } from "../lib/types";

interface Props {
  enabledTools: Record<string, boolean>;
  onChange: (enabled: Record<string, boolean>) => void;
}

/** 工具选择器：拉取 /api/tools，所有工具平铺渲染，默认全部禁用。 */
export function ToolSelector({ enabledTools, onChange }: Props) {
  const [tools, setTools] = useState<ToolConfig[] | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const list = await getToolsConfig();
        if (cancelled) return;
        setTools(list);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("加载工具列表失败:", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (!tools) {
    return <div className="text-xs text-ink-400 px-2 py-1">加载工具列表…</div>;
  }

  const enabledCount = tools.filter((t) => enabledTools[t.id]).length;

  const toggle = (id: string) => {
    onChange({ ...enabledTools, [id]: !enabledTools[id] });
  };

  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
      className="rounded-lg border border-cream-300 bg-cream-50/40 text-sm"
    >
      <summary className="cursor-pointer select-none flex items-center gap-2 px-3 py-2 text-ink-600">
        <Wrench className="w-3.5 h-3.5" />
        <span className="font-medium">工具</span>
        <span className="text-xs text-ink-400">
          已启用 {enabledCount} / {tools.length}
        </span>
        <ChevronDown
          className={`w-4 h-4 ml-auto transition-transform ${open ? "rotate-180" : ""}`}
        />
      </summary>
      <div className="px-3 pb-3 pt-1 space-y-1.5 border-t border-cream-300/60">
        {tools.map((t) => (
          <ToolRow
            key={t.id}
            tool={t}
            enabled={enabledTools[t.id] ?? false}
            onToggle={() => toggle(t.id)}
          />
        ))}
      </div>
    </details>
  );
}

interface ToolRowProps {
  tool: ToolConfig;
  enabled: boolean;
  onToggle: () => void;
}

function ToolRow({ tool, enabled, onToggle }: ToolRowProps) {
  return (
    <label className="flex items-start gap-2 px-2 py-1.5 rounded hover:bg-cream-100/80 cursor-pointer">
      <input
        type="checkbox"
        checked={enabled}
        onChange={onToggle}
        className="mt-0.5 accent-terracotta-500"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <code className="font-mono text-[12px] text-ink-700 font-medium">
            {tool.display_name}
          </code>
          <RiskBadge level={tool.risk_level} />
          <span className="text-[10px] text-ink-400 font-mono">{tool.mcp_server}</span>
        </div>
        <div className="text-xs text-ink-500 mt-0.5 leading-relaxed">
          {tool.description}
        </div>
      </div>
    </label>
  );
}

function RiskBadge({ level }: { level: RiskLevel }) {
  const cls =
    level === "HIGH"
      ? "bg-red-100 text-red-700 border-red-200"
      : level === "MEDIUM"
      ? "bg-amber-100 text-amber-700 border-amber-200"
      : "bg-emerald-100 text-emerald-700 border-emerald-200";
  return (
    <span
      className={`inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border ${cls}`}
    >
      {level === "HIGH" && <AlertTriangle className="w-2.5 h-2.5" />}
      {level}
    </span>
  );
}
