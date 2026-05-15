import { useEffect, useState } from "react";
import { ChevronDown, Wrench, AlertTriangle, Plug, Cpu } from "lucide-react";
import { getToolsConfig } from "../lib/api";
import type { RiskLevel, ToolConfig } from "../lib/types";

interface Props {
  enabledTools: Record<string, boolean>;
  onChange: (enabled: Record<string, boolean>) => void;
}

/** 工具选择器：拉取 /api/tools，按 internal/external 分组渲染开关。 */
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
        // 仅当父组件未提供启用集合时，才用 registry 默认值初始化
        if (Object.keys(enabledTools).length === 0) {
          const defaults: Record<string, boolean> = {};
          for (const t of list) defaults[t.id] = t.default_enabled;
          onChange(defaults);
        }
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("加载工具列表失败:", err);
      }
    })();
    return () => {
      cancelled = true;
    };
    // 仅在挂载时拉取一次
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!tools) {
    return (
      <div className="text-xs text-ink-400 px-2 py-1">加载工具列表…</div>
    );
  }

  const internal = tools.filter((t) => t.category === "internal");
  const external = tools.filter((t) => t.category === "external");
  const enabledCount = Object.values(enabledTools).filter(Boolean).length;

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
      <div className="px-3 pb-3 pt-1 space-y-3 border-t border-cream-300/60">
        <ToolGroup
          label="内置工具"
          icon={<Cpu className="w-3.5 h-3.5" />}
          tools={internal}
          enabledTools={enabledTools}
          onToggle={toggle}
        />
        <ToolGroup
          label="外部工具"
          icon={<Plug className="w-3.5 h-3.5" />}
          tools={external}
          enabledTools={enabledTools}
          onToggle={toggle}
        />
      </div>
    </details>
  );
}

interface ToolGroupProps {
  label: string;
  icon: React.ReactNode;
  tools: ToolConfig[];
  enabledTools: Record<string, boolean>;
  onToggle: (id: string) => void;
}

function ToolGroup({ label, icon, tools, enabledTools, onToggle }: ToolGroupProps) {
  if (tools.length === 0) return null;
  return (
    <div>
      <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-ink-400 mb-1.5">
        {icon}
        <span>{label}</span>
      </div>
      <div className="space-y-1.5">
        {tools.map((t) => (
          <ToolRow
            key={t.id}
            tool={t}
            enabled={enabledTools[t.id] ?? t.default_enabled}
            onToggle={() => onToggle(t.id)}
          />
        ))}
      </div>
    </div>
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
