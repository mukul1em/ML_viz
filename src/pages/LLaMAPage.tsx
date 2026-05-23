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
import QwenDetail from "../components/qwen/QwenDetail";
import LlamaSidebar from "../components/llama/LlamaSidebar";
import {
  LLAMA_PRESETS,
  SAMPLE_PROMPT,
  type LlamaPresetName,
} from "../lib/llamaDemo";
import {
  llamaComponents,
  type LlamaComponentId,
} from "../data/llamaComponents";
import type { QwenComponentId } from "../data/qwenComponents";

type RightTab = "inspector" | "stats";

const LLAMA_ACCENT = "#fb923c"; // orange-400

export default function LLaMAPage() {
  const [selected, setSelected] = useState<LlamaComponentId>("rope");
  const [prompt, setPrompt] = useState(SAMPLE_PROMPT);
  const [presetName, setPresetName] = useState<LlamaPresetName>("Llama-3.1-8B");
  const [granularity, setGranularity] = useState<Granularity>("block");
  const [rightTab, setRightTab] = useState<RightTab>("inspector");
  const preset = LLAMA_PRESETS[presetName];

  const modelLabel = (() => {
    const idx = presetName.indexOf("-");
    return {
      brand: presetName.slice(0, idx),
      model: presetName.slice(idx),
    };
  })();

  const handleSelect = (id: QwenComponentId) => {
    setSelected(id as LlamaComponentId);
    setRightTab("inspector");
  };

  const selectedMeta = llamaComponents[selected];

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-orange-300 font-semibold">
          <span className="w-1.5 h-1.5 rounded-full bg-orange-400 animate-pulse" />
          Interactive model view
          <span className="ml-2 px-1.5 py-0.5 rounded bg-orange-500/20 border border-orange-400/40 text-orange-200">
            Meta · open weights
          </span>
        </div>
        <div className="flex items-end justify-between flex-wrap gap-4">
          <h1 className="text-4xl lg:text-6xl font-extrabold tracking-tight leading-[1]">
            <span className="text-ink-400">{modelLabel.brand}</span>
            <span className="bg-gradient-to-r from-orange-300 via-pink-300 to-fuchsia-300 bg-clip-text text-transparent">
              {modelLabel.model}
            </span>
          </h1>
          <div className="flex items-center gap-2 flex-wrap">
            <Link to="/llm" className="btn" title="Back to all LLM models">
              <ArrowLeft className="w-4 h-4" /> Back to models
            </Link>
            <a
              href="https://arxiv.org/abs/2407.21783"
              target="_blank"
              rel="noreferrer"
              className="btn"
              title="The Llama 3 Herd of Models (Meta, 2024)"
            >
              <FileText className="w-4 h-4" /> Paper
            </a>
            <Link to="/llm/qwen" className="btn">
              <ExternalLink className="w-4 h-4" /> Compare with Qwen
            </Link>
          </div>
        </div>
      </header>

      {/* Why LLaMA matters strip */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Highlight
          color="#22d3ee"
          label="RMSNorm"
          detail="no centering, no bias"
        />
        <Highlight
          color="#ec4899"
          label={`RoPE θ=${preset.rope_theta.toLocaleString()}`}
          detail={`${(preset.context / 1024).toFixed(0)}k context`}
        />
        <Highlight
          color="#facc15"
          label="GQA"
          detail={`${preset.q_heads} Q : ${preset.kv_heads} KV`}
        />
        <Highlight color="#fb923c" label="SwiGLU" detail="bias-free 3-linear FFN" />
      </section>

      {/* The recipe everyone copies */}
      <section className="rounded-xl border border-orange-500/30 bg-gradient-to-r from-orange-500/10 via-pink-500/5 to-transparent p-4 flex items-start gap-3">
        <span className="mt-0.5 px-2 py-0.5 rounded text-[10px] uppercase tracking-wider font-bold bg-orange-500/20 text-orange-200 border border-orange-400/40">
          The reference recipe
        </span>
        <p className="text-sm text-ink-200 leading-relaxed">
          When LLaMA-1 dropped (Feb&nbsp;2023), it codified a clean post-GPT-2
          recipe: pre-norm RMSNorm, RoPE positional encoding, SwiGLU FFN, and
          no bias on any linear. LLaMA-2 added GQA. Almost every modern
          open-weight LM &mdash; Qwen, Mistral, DeepSeek, Gemma &mdash; ships
          the same five ingredients.
        </p>
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
              className="flex-1 px-3 py-2 rounded-lg bg-ink-950/70 border border-ink-700 focus:border-orange-400/60 focus:outline-none font-mono text-sm"
            />
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            {(Object.keys(LLAMA_PRESETS) as LlamaPresetName[]).map((p) => (
              <button
                key={p}
                onClick={() => setPresetName(p)}
                className={[
                  "px-2.5 py-1.5 rounded-lg text-[11px] font-medium border transition-all",
                  presetName === p
                    ? "bg-orange-500 text-white border-orange-400 shadow-[0_0_0_2px_rgba(251,146,60,0.25)]"
                    : "bg-ink-900/70 border-ink-700 text-ink-200 hover:border-orange-400/40",
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
                selected={selected as QwenComponentId}
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
                  <span
                    className="w-1.5 h-1.5 rounded-full shrink-0"
                    style={{ background: LLAMA_ACCENT }}
                  />
                  <span className="truncate">{selectedMeta?.short}</span>
                </div>
              )}
            </div>

            <div className="flex-1 overflow-y-auto p-4 lg:p-5">
              {rightTab === "inspector" ? (
                <QwenDetail
                  selected={selected as QwenComponentId}
                  prompt={prompt}
                  config={preset}
                  embedded
                  components={llamaComponents}
                  accentColor={LLAMA_ACCENT}
                />
              ) : (
                <LlamaSidebar
                  modelName={presetName}
                  config={preset}
                />
              )}
            </div>

            {rightTab === "inspector" && (
              <div className="px-4 py-2 border-t border-ink-800/70 bg-ink-950/60 text-[10px] text-ink-500 flex items-center gap-2">
                <MousePointerClick className="w-3 h-3" />
                Click any node — every linear is bias-free, RoPE handles position.
              </div>
            )}
          </aside>
        </div>
      </section>

      {/* Family tree */}
      <section className="card p-5">
        <div className="text-[10px] uppercase tracking-[0.18em] text-ink-400 font-bold mb-3">
          LLaMA family tree
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <FamilyCard
            era="2023 · Feb"
            name="LLaMA 1"
            features={[
              "RMSNorm + RoPE + SwiGLU",
              "Multi-head attention (no GQA)",
              "SentencePiece 32k",
              "2k context",
            ]}
            highlight={false}
          />
          <FamilyCard
            era="2023 · Jul"
            name="LLaMA 2"
            features={[
              "Adds GQA on 70B variant",
              "Same recipe otherwise",
              "4k context",
              "First major open-weights release",
            ]}
            highlight={false}
          />
          <FamilyCard
            era="2024+"
            name="LLaMA 3 · 3.1 · 3.2"
            features={[
              "tiktoken BPE 128k vocab",
              "GQA across all sizes",
              "RoPE θ=500k → 128k context",
              "Sizes from 1B to 405B",
            ]}
            highlight
          />
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

function FamilyCard({
  era,
  name,
  features,
  highlight,
}: {
  era: string;
  name: string;
  features: string[];
  highlight: boolean;
}) {
  return (
    <div
      className={
        highlight
          ? "rounded-xl border border-orange-400/40 bg-gradient-to-br from-orange-500/10 via-pink-500/5 to-transparent p-4 flex flex-col gap-2"
          : "rounded-xl border border-ink-800 bg-ink-950/50 p-4 flex flex-col gap-2"
      }
    >
      <div className="text-[10px] uppercase tracking-wider font-bold text-ink-400">
        {era}
      </div>
      <div className={highlight ? "text-lg font-bold text-orange-200" : "text-lg font-bold text-white"}>
        {name}
      </div>
      <ul className="flex flex-col gap-1 mt-1">
        {features.map((f) => (
          <li key={f} className="flex items-start gap-2 text-[11.5px] text-ink-300 leading-snug">
            <span
              className="mt-1.5 w-1 h-1 rounded-full shrink-0"
              style={{ background: highlight ? "#fb923c" : "#7c5cff" }}
            />
            {f}
          </li>
        ))}
      </ul>
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
          ? "bg-orange-500 text-white shadow-[0_0_0_2px_rgba(251,146,60,0.25)]"
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
          ? "bg-orange-500 text-white shadow-[0_0_0_2px_rgba(251,146,60,0.25)]"
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
      className="w-7 h-7 rounded-md bg-ink-900/80 border border-ink-700 text-ink-300 hover:text-white hover:border-orange-400/50 hover:bg-ink-800/80 flex items-center justify-center transition-all backdrop-blur"
    >
      {icon}
    </button>
  );
}
