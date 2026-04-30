import { useCallback, useEffect, useRef, useState } from "react";
import mermaid from "mermaid";
import {
  Copy,
  Check,
  Download,
  Maximize2,
  Minimize2,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

let initialized = false;

function ensureInit() {
  if (initialized) return;
  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    themeVariables: {
      primaryColor: "#FAF9F5",
      primaryTextColor: "#1A1A1A",
      primaryBorderColor: "#D97757",
      lineColor: "#C9623F",
      secondaryColor: "#F4F1EA",
      tertiaryColor: "#FDFCF8",
      fontFamily: "Inter, system-ui, sans-serif",
    },
    flowchart: { htmlLabels: true, curve: "basis", padding: 12 },
    mindmap: { padding: 16 },
    securityLevel: "loose",
  });
  initialized = true;
}

export function MermaidRenderer({ code }: { code: string }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const idRef = useRef(`m-${Math.random().toString(36).slice(2, 10)}`);
  const [svgText, setSvgText] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [scale, setScale] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [fullscreen, setFullscreen] = useState(false);
  const [copied, setCopied] = useState(false);

  // 渲染 mermaid → SVG 字符串
  useEffect(() => {
    ensureInit();
    let cancelled = false;
    setError("");
    (async () => {
      try {
        const { svg } = await mermaid.render(idRef.current, code);
        if (!cancelled) setSvgText(svg);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [code]);

  // 重置变换：每次代码变化 / 进出全屏后回到居中
  const reset = useCallback(() => {
    setScale(1);
    setTx(0);
    setTy(0);
  }, []);
  useEffect(() => {
    reset();
  }, [code, fullscreen, reset]);

  // 滚轮缩放（按住 Ctrl 或在容器内滚轮）
  const onWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (!e.ctrlKey && !e.metaKey) return; // 仅 Ctrl/Cmd + 滚轮缩放，避免劫持页面滚动
    e.preventDefault();
    const delta = -e.deltaY * 0.0015;
    setScale((s) => clamp(s * (1 + delta), 0.3, 4));
  };

  // 拖拽平移
  const dragRef = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);
  const onMouseDown = (e: React.MouseEvent) => {
    dragRef.current = { x: e.clientX, y: e.clientY, tx, ty };
  };
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      setTx(dragRef.current.tx + (e.clientX - dragRef.current.x));
      setTy(dragRef.current.ty + (e.clientY - dragRef.current.y));
    };
    const onUp = () => {
      dragRef.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  // ESC 退出全屏
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullscreen]);

  const copyDsl = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  };

  const downloadSvg = () => {
    if (!svgText) return;
    const blob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
    triggerDownload(blob, "diagram.svg");
  };

  const downloadPng = async () => {
    const svgEl = stageRef.current?.querySelector("svg") as SVGSVGElement | null;
    if (!svgEl) return;
    try {
      const blob = await svgToPngBlob(svgEl, 2);
      triggerDownload(blob, "diagram.png");
    } catch (e) {
      console.error(e);
    }
  };

  if (error) {
    return (
      <div className="my-4 p-4 bg-cream-50 border border-cream-300 rounded-lg">
        <div className="text-sm font-medium text-terracotta-700 mb-2">⚠️ 图表渲染失败</div>
        <div className="text-xs text-ink-500 mb-2 whitespace-pre-wrap break-words">{error}</div>
        <details className="text-xs">
          <summary className="cursor-pointer text-ink-400 hover:text-ink-600">查看原始 DSL</summary>
          <pre className="mt-2 p-3 bg-cream-200 rounded text-ink-600 whitespace-pre-wrap break-words overflow-x-auto">
            {code}
          </pre>
        </details>
      </div>
    );
  }

  const container = (
    <div
      ref={hostRef}
      className={
        fullscreen
          ? "fixed inset-0 z-50 bg-cream-50 flex flex-col"
          : "my-4 bg-cream-50 border border-cream-300 rounded-lg flex flex-col"
      }
    >
      {/* 工具栏 */}
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-cream-300 bg-cream-100/80 backdrop-blur rounded-t-lg">
        <ToolBtn title="缩小" onClick={() => setScale((s) => clamp(s / 1.2, 0.3, 4))}>
          <ZoomOut size={14} />
        </ToolBtn>
        <span className="px-2 text-xs tabular-nums text-ink-500 select-none">
          {(scale * 100).toFixed(0)}%
        </span>
        <ToolBtn title="放大" onClick={() => setScale((s) => clamp(s * 1.2, 0.3, 4))}>
          <ZoomIn size={14} />
        </ToolBtn>
        <ToolBtn title="重置视图" onClick={reset}>
          <RotateCcw size={14} />
        </ToolBtn>
        <div className="flex-1" />
        <ToolBtn title="复制 DSL" onClick={copyDsl}>
          {copied ? <Check size={14} className="text-emerald-600" /> : <Copy size={14} />}
        </ToolBtn>
        <ToolBtn title="导出 SVG" onClick={downloadSvg}>
          <Download size={14} />
          <span className="ml-1 text-[11px]">SVG</span>
        </ToolBtn>
        <ToolBtn title="导出 PNG" onClick={downloadPng}>
          <Download size={14} />
          <span className="ml-1 text-[11px]">PNG</span>
        </ToolBtn>
        <ToolBtn
          title={fullscreen ? "退出全屏" : "全屏"}
          onClick={() => setFullscreen((f) => !f)}
        >
          {fullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
        </ToolBtn>
      </div>

      {/* 画布 */}
      <div
        className={
          "relative overflow-hidden select-none " +
          (fullscreen ? "flex-1" : "h-[420px]") +
          " cursor-grab active:cursor-grabbing"
        }
        onWheel={onWheel}
        onMouseDown={onMouseDown}
      >
        <div
          ref={stageRef}
          className="absolute inset-0 flex items-center justify-center will-change-transform"
          style={{ transform: `translate(${tx}px, ${ty}px) scale(${scale})` }}
          dangerouslySetInnerHTML={{ __html: svgText }}
        />
      </div>
    </div>
  );

  return container;
}

function ToolBtn({
  title,
  onClick,
  children,
}: {
  title: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className="inline-flex items-center px-2 py-1 rounded hover:bg-cream-200 text-ink-600 hover:text-ink-900 transition-colors text-xs"
    >
      {children}
    </button>
  );
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function svgToPngBlob(svgEl: SVGSVGElement, scale = 2): Promise<Blob> {
  // 克隆并补齐 xmlns，确保 Image 能解析
  const clone = svgEl.cloneNode(true) as SVGSVGElement;
  if (!clone.getAttribute("xmlns")) clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const bbox = svgEl.getBoundingClientRect();
  const width = Math.max(1, Math.ceil(bbox.width));
  const height = Math.max(1, Math.ceil(bbox.height));
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  const xml = new XMLSerializer().serializeToString(clone);
  const svg64 = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(xml);

  const img = new Image();
  img.crossOrigin = "anonymous";
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = (e) => reject(e);
    img.src = svg64;
  });

  const canvas = document.createElement("canvas");
  canvas.width = width * scale;
  canvas.height = height * scale;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#FAF9F5";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.drawImage(img, 0, 0);

  return await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/png");
  });
}
