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
import GPTGraphCanvas, {
  type Granularity,
} from "../components/gpt/GPTGraphCanvas";
import GPTSidebar from "../components/gpt/GPTSidebar";
import GPTDetail from "../components/gpt/GPTDetail";
import {
  GPT_CONFIG_PRESETS,
  SAMPLE_PROMPT,
} from "../lib/gptDemo";
import { gptComponents, type GPTComponentId } from "../data/gptComponents";

type PresetName = keyof typeof GPT_CONFIG_PRESETS;
type RightTab = "inspector" | "stats";

export default function GPTPage() {
  const [selected, setSelected] = useState<GPTComponentId>("selfAttention");
  const [prompt, setPrompt] = useState(SAMPLE_PROMPT);
  const [presetName, setPresetName] = useState<PresetName>("GPT-2 small");
  const [granularity, setGranularity] = useState<Granularity>("block");
  const [rightTab, setRightTab] = useState<RightTab>("inspector");
  const preset = GPT_CONFIG_PRESETS[presetName];

  const modelLabel = (() => {
    switch (presetName) {
      case "GPT-2 small":
        return { brand: "GPT-2", model: "small (124M)" };
      case "GPT-2 medium":
        return { brand: "GPT-2", model: "medium (355M)" };
      case "GPT-2 large":
        return { brand: "GPT-2", model: "large (774M)" };
      case "GPT-3 175B":
        return { brand: "GPT-3", model: "175B" };
    }
  })();

  const handleSelect = (id: GPTComponentId) => {
    setSelected(id);
    setRightTab("inspector");
  };

  const selectedMeta = gptComponents[selected];

  return (
    <div className="flex flex-col gap-6">
      {/* Branded header */}
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-accent-soft font-semibold">
          <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
          Interactive model view
        </div>
        <div className="flex items-end justify-between flex-wrap gap-4">
          <h1 className="text-4xl lg:text-6xl font-extrabold tracking-tight leading-[1]">
            <span className="text-ink-400">{modelLabel.brand}</span>
            <span className="text-ink-400">/</span>
            <span className="bg-gradient-to-r from-white via-ink-100 to-ink-300 bg-clip-text text-transparent">
              {modelLabel.model}
            </span>
          </h1>
          <div className="flex items-center gap-2 flex-wrap">
            <Link to="/llm" className="btn" title="Back to all LLM models">
              <ArrowLeft className="w-4 h-4" /> Back to models
            </Link>
            <a
              href="https://arxiv.org/abs/2005.14165"
              target="_blank"
              rel="noreferrer"
              className="btn"
              title="GPT-3 paper"
            >
              <FileText className="w-4 h-4" /> Paper
            </a>
            <Link to="/softmax" className="btn">
              <ExternalLink className="w-4 h-4" /> Softmax viz
            </Link>
          </div>
        </div>
      </header>

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
              className="flex-1 px-3 py-2 rounded-lg bg-ink-950/70 border border-ink-700 focus:border-accent/60 focus:outline-none font-mono text-sm"
            />
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            {(Object.keys(GPT_CONFIG_PRESETS) as PresetName[]).map((p) => (
              <button
                key={p}
                onClick={() => setPresetName(p)}
                className={[
                  "px-2.5 py-1.5 rounded-lg text-[11px] font-medium border transition-all",
                  presetName === p
                    ? "bg-accent text-white border-accent shadow-glow"
                    : "bg-ink-900/70 border-ink-700 text-ink-200 hover:border-accent/40",
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
              {modelLabel.brand}/{modelLabel.model.replace(/\s.*$/, "")} · graph
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-[0.14em] text-ink-400 font-semibold mr-1">
              Granularity
            </span>
            <div className="flex rounded-lg bg-ink-900/70 border border-ink-800 p-0.5">
              <GranularityButton
                label="Block"
                active={granularity === "block"}
                onClick={() => setGranularity("block")}
              />
              <GranularityButton
                label="Op"
                active={granularity === "op"}
                onClick={() => setGranularity("op")}
              />
            </div>
          </div>
        </div>

        {/* Canvas + right panel */}
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_440px] gap-0">
          <div className="relative border-b xl:border-b-0 xl:border-r border-ink-800/70 min-h-[640px]">
            {/* Floating canvas controls */}
            <div className="absolute top-3 right-3 z-10 flex flex-col gap-1.5">
              <CanvasIconBtn icon={<Search className="w-3.5 h-3.5" />} title="Search" />
              <CanvasIconBtn icon={<ZoomIn className="w-3.5 h-3.5" />} title="Zoom in" />
              <CanvasIconBtn icon={<ZoomOut className="w-3.5 h-3.5" />} title="Zoom out" />
              <CanvasIconBtn icon={<Maximize2 className="w-3.5 h-3.5" />} title="Fit" />
              <CanvasIconBtn icon={<Focus className="w-3.5 h-3.5" />} title="Center selection" />
            </div>
            <div className="p-5">
              <GPTGraphCanvas
                granularity={granularity}
                numLayers={preset.layers}
                selected={selected}
                onSelect={handleSelect}
              />
            </div>
            {/* Footer status */}
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

          {/* Right panel: tabbed Inspector + Stats */}
          <aside className="bg-ink-950/40 flex flex-col max-h-[calc(100vh-160px)] xl:max-h-[900px] min-h-[640px]">
            {/* Tab header */}
            <div className="sticky top-0 z-10 flex items-center justify-between gap-2 px-3 py-2.5 border-b border-ink-800/70 bg-ink-950/80 backdrop-blur">
              <div className="flex items-center gap-1 rounded-lg bg-ink-900/70 border border-ink-800 p-0.5">
                <TabButton
                  active={rightTab === "inspector"}
                  onClick={() => setRightTab("inspector")}
                  icon={<MousePointerClick className="w-3.5 h-3.5" />}
                  label="Inspector"
                />
                <TabButton
                  active={rightTab === "stats"}
                  onClick={() => setRightTab("stats")}
                  icon={<BarChart3 className="w-3.5 h-3.5" />}
                  label="Stats"
                />
              </div>
              {rightTab === "inspector" && (
                <div className="flex items-center gap-1.5 text-[10px] text-ink-400 font-mono pr-1 truncate">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
                  <span className="truncate">{selectedMeta?.short}</span>
                </div>
              )}
            </div>

            {/* Tab content (scrollable) */}
            <div className="flex-1 overflow-y-auto p-4 lg:p-5">
              {rightTab === "inspector" ? (
                <GPTDetail
                  selected={selected}
                  prompt={prompt}
                  numHeads={preset.heads}
                  dModel={preset.d_model}
                  embedded
                />
              ) : (
                <GPTSidebar
                  modelName={`${modelLabel.brand}/${modelLabel.model} model`}
                  numLayers={preset.layers}
                  dModel={preset.d_model}
                  numHeads={preset.heads}
                  ffnDim={preset.ffn}
                />
              )}
            </div>

            {/* Hint banner */}
            {rightTab === "inspector" && (
              <div className="px-4 py-2 border-t border-ink-800/70 bg-ink-950/60 text-[10px] text-ink-500 flex items-center gap-2">
                <MousePointerClick className="w-3 h-3" />
                Click any node in the graph to update this panel.
              </div>
            )}
          </aside>
        </div>
      </section>
    </div>
  );
}

function GranularityButton({
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
          ? "bg-accent text-white shadow-glow"
          : "text-ink-300 hover:text-white",
      ].join(" ")}
    >
      {label.toUpperCase()}
    </button>
  );
}

function TabButton({
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
          ? "bg-accent text-white shadow-glow"
          : "text-ink-300 hover:text-white",
      ].join(" ")}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

function CanvasIconBtn({
  icon,
  title,
}: {
  icon: React.ReactNode;
  title: string;
}) {
  return (
    <button
      title={title}
      className="w-7 h-7 rounded-md bg-ink-900/80 border border-ink-700 text-ink-300 hover:text-white hover:border-accent/50 hover:bg-ink-800/80 flex items-center justify-center transition-all backdrop-blur"
    >
      {icon}
    </button>
  );
}
