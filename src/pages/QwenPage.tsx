import { useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  BarChart3,
  ExternalLink,
  FileText,
  Focus,
  Maximize2,
  MousePointerClick,
  Search,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import QwenGraphCanvas, {
  type Granularity,
} from "../components/qwen/QwenGraphCanvas";
import QwenSidebar from "../components/qwen/QwenSidebar";
import QwenDetail from "../components/qwen/QwenDetail";
import { QWEN_PRESETS, SAMPLE_PROMPT, type QwenPresetName } from "../lib/qwenDemo";
import { qwenComponents, type QwenComponentId } from "../data/qwenComponents";

type RightTab = "inspector" | "stats";

export default function QwenPage() {
  const [selected, setSelected] = useState<QwenComponentId>("rope");
  const [prompt, setPrompt] = useState(SAMPLE_PROMPT);
  const [presetName, setPresetName] = useState<QwenPresetName>("Qwen2.5-7B");
  const [granularity, setGranularity] = useState<Granularity>("block");
  const [rightTab, setRightTab] = useState<RightTab>("inspector");
  const preset = QWEN_PRESETS[presetName];

  const modelLabel = (() => {
    const [brand, model] = presetName.split("-");
    return { brand, model: `-${model}` };
  })();

  const handleSelect = (id: QwenComponentId) => {
    setSelected(id);
    setRightTab("inspector");
  };

  const selectedMeta = qwenComponents[selected];

  return (
    <div className="flex flex-col gap-6">
      {/* Branded header */}
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-violet-300 font-semibold">
          <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
          Interactive model view
          <span className="ml-2 px-1.5 py-0.5 rounded bg-pink-500/20 border border-pink-400/40 text-pink-200">
            Qwen2.5
          </span>
        </div>
        <div className="flex items-end justify-between flex-wrap gap-4">
          <h1 className="text-4xl lg:text-6xl font-extrabold tracking-tight leading-[1]">
            <span className="text-ink-400">{modelLabel.brand}</span>
            <span className="bg-gradient-to-r from-violet-300 via-pink-300 to-cyan-300 bg-clip-text text-transparent">
              {modelLabel.model}
            </span>
          </h1>
          <div className="flex items-center gap-2 flex-wrap">
            <Link to="/llm" className="btn" title="Back to all LLM models">
              <ArrowLeft className="w-4 h-4" /> Back to models
            </Link>
            <a
              href="https://arxiv.org/abs/2412.15115"
              target="_blank"
              rel="noreferrer"
              className="btn"
              title="Qwen2.5 technical report"
            >
              <FileText className="w-4 h-4" /> Paper
            </a>
            <Link to="/llm/gpt" className="btn">
              <ExternalLink className="w-4 h-4" /> Compare with GPT
            </Link>
          </div>
        </div>
      </header>

      {/* Highlights strip — signature features */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Highlight color="#22d3ee" label="RMSNorm" detail="no centering, no bias" />
        <Highlight color="#ec4899" label="RoPE" detail="rotary positions" />
        <Highlight color="#facc15" label="GQA" detail="shared K/V" />
        <Highlight color="#7c5cff" label="SwiGLU" detail="gated FFN" />
      </section>

      {/* Prompt + preset bar */}
      <section className="card p-3.5 lg:p-4">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-3 items-center">
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold w-12">
              Prompt
            </span>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Type a prompt…"
              className="flex-1 px-3 py-2 rounded-lg bg-ink-950/70 border border-ink-700 focus:border-violet-400/60 focus:outline-none font-mono text-sm"
            />
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            {(Object.keys(QWEN_PRESETS) as QwenPresetName[]).map((p) => (
              <button
                key={p}
                onClick={() => setPresetName(p)}
                className={[
                  "px-2.5 py-1.5 rounded-lg text-[11px] font-medium border transition-all",
                  presetName === p
                    ? "bg-violet-500 text-white border-violet-400 shadow-[0_0_0_2px_rgba(124,92,255,0.25)]"
                    : "bg-ink-900/70 border-ink-700 text-ink-200 hover:border-violet-400/40",
                ].join(" ")}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Canvas window */}
      <section className="rounded-2xl border border-ink-800 bg-gradient-to-b from-ink-900/80 to-ink-950/80 shadow-[0_30px_80px_-30px_rgba(0,0,0,0.7)] overflow-hidden">
        {/* Chrome bar */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-ink-800/70 bg-ink-950/60">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-red-500/80" />
            <span className="w-3 h-3 rounded-full bg-yellow-500/80" />
            <span className="w-3 h-3 rounded-full bg-green-500/80" />
            <span className="ml-3 text-[11px] text-ink-400 font-mono">
              {presetName} · graph
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-[0.14em] text-ink-400 font-semibold mr-1">
              Granularity
            </span>
            <div className="flex rounded-lg bg-ink-900/70 border border-ink-800 p-0.5">
              <GranBtn label="Block" active={granularity === "block"} onClick={() => setGranularity("block")} />
              <GranBtn label="Op" active={granularity === "op"} onClick={() => setGranularity("op")} />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-[1fr_440px] gap-0">
          <div className="relative border-b xl:border-b-0 xl:border-r border-ink-800/70 min-h-[640px]">
            <div className="absolute top-3 right-3 z-10 flex flex-col gap-1.5">
              <ChromeBtn icon={<Search className="w-3.5 h-3.5" />} title="Search" />
              <ChromeBtn icon={<ZoomIn className="w-3.5 h-3.5" />} title="Zoom in" />
              <ChromeBtn icon={<ZoomOut className="w-3.5 h-3.5" />} title="Zoom out" />
              <ChromeBtn icon={<Maximize2 className="w-3.5 h-3.5" />} title="Fit" />
              <ChromeBtn icon={<Focus className="w-3.5 h-3.5" />} title="Center selection" />
            </div>
            <div className="p-5">
              <QwenGraphCanvas
                granularity={granularity}
                numLayers={preset.layers}
                selected={selected}
                onSelect={handleSelect}
              />
            </div>
            <div className="absolute bottom-3 left-4 right-4 flex items-center justify-between text-[10px] text-ink-500 font-mono pointer-events-none">
              <span>
                selected:{" "}
                <span className="text-ink-200">{selectedMeta?.full}</span>
              </span>
              <span>
                granularity:{" "}
                <span className="text-ink-200">{granularity.toUpperCase()}</span>
              </span>
            </div>
          </div>

          <aside className="bg-ink-950/40 flex flex-col max-h-[calc(100vh-160px)] xl:max-h-[900px] min-h-[640px]">
            <div className="sticky top-0 z-10 flex items-center justify-between gap-2 px-3 py-2.5 border-b border-ink-800/70 bg-ink-950/80 backdrop-blur">
              <div className="flex items-center gap-1 rounded-lg bg-ink-900/70 border border-ink-800 p-0.5">
                <TabBtn
                  active={rightTab === "inspector"}
                  onClick={() => setRightTab("inspector")}
                  icon={<MousePointerClick className="w-3.5 h-3.5" />}
                  label="Inspector"
                />
                <TabBtn
                  active={rightTab === "stats"}
                  onClick={() => setRightTab("stats")}
                  icon={<BarChart3 className="w-3.5 h-3.5" />}
                  label="Stats"
                />
              </div>
              {rightTab === "inspector" && (
                <div className="flex items-center gap-1.5 text-[10px] text-ink-400 font-mono pr-1 truncate">
                  <span className="w-1.5 h-1.5 rounded-full bg-violet-400 shrink-0" />
                  <span className="truncate">{selectedMeta?.short}</span>
                </div>
              )}
            </div>

            <div className="flex-1 overflow-y-auto p-4 lg:p-5">
              {rightTab === "inspector" ? (
                <QwenDetail
                  selected={selected}
                  prompt={prompt}
                  config={preset}
                  embedded
                />
              ) : (
                <QwenSidebar
                  modelName={presetName}
                  config={preset}
                />
              )}
            </div>

            {rightTab === "inspector" && (
              <div className="px-4 py-2 border-t border-ink-800/70 bg-ink-950/60 text-[10px] text-ink-500 flex items-center gap-2">
                <MousePointerClick className="w-3 h-3" />
                Click any node — RoPE & GQA & SwiGLU are the highlights.
              </div>
            )}
          </aside>
        </div>
      </section>
    </div>
  );
}

function Highlight({
  color,
  label,
  detail,
}: {
  color: string;
  label: string;
  detail: string;
}) {
  return (
    <div
      className="rounded-xl border p-3 flex flex-col gap-1"
      style={{
        background: `linear-gradient(180deg, ${color}1c, transparent)`,
        borderColor: `${color}55`,
      }}
    >
      <div className="text-sm font-bold" style={{ color }}>
        {label}
      </div>
      <div className="text-[11px] text-ink-300">{detail}</div>
    </div>
  );
}

function GranBtn({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "px-3 py-1 rounded-md text-[11px] font-semibold tracking-wider transition-all",
        active
          ? "bg-violet-500 text-white shadow-[0_0_0_2px_rgba(124,92,255,0.25)]"
          : "text-ink-300 hover:text-white",
      ].join(" ")}
    >
      {label.toUpperCase()}
    </button>
  );
}

function TabBtn({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold tracking-wide transition-all",
        active
          ? "bg-violet-500 text-white shadow-[0_0_0_2px_rgba(124,92,255,0.25)]"
          : "text-ink-300 hover:text-white",
      ].join(" ")}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

function ChromeBtn({
  icon,
  title,
}: {
  icon: React.ReactNode;
  title: string;
}) {
  return (
    <button
      title={title}
      className="w-7 h-7 rounded-md bg-ink-900/80 border border-ink-700 text-ink-300 hover:text-white hover:border-violet-400/50 hover:bg-ink-800/80 flex items-center justify-center transition-all backdrop-blur"
    >
      {icon}
    </button>
  );
}
