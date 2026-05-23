import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  BarChart3,
  ExternalLink,
  EyeOff,
  FileText,
  Focus,
  Maximize2,
  MousePointerClick,
  Search,
  Shuffle,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import BertGraphCanvas, {
  type BertGranularity,
} from "../components/bert/BertGraphCanvas";
import BertSidebar from "../components/bert/BertSidebar";
import BertDetail from "../components/bert/BertDetail";
import {
  BERT_CONFIG_PRESETS,
  bertTokenize,
  SAMPLE_TEXT_A,
  SAMPLE_TEXT_B,
} from "../lib/bertDemo";
import { bertComponents, type BertComponentId } from "../data/bertComponents";

type PresetName = keyof typeof BERT_CONFIG_PRESETS;
type RightTab = "inspector" | "stats";

export default function BERTPage() {
  const [selected, setSelected] = useState<BertComponentId>("selfAttention");
  const [textA, setTextA] = useState(SAMPLE_TEXT_A);
  const [textB, setTextB] = useState(SAMPLE_TEXT_B);
  const [presetName, setPresetName] = useState<PresetName>("BERT-base");
  const [granularity, setGranularity] = useState<BertGranularity>("block");
  const [rightTab, setRightTab] = useState<RightTab>("inspector");
  const [maskedIndex, setMaskedIndex] = useState<number | null>(null);

  const preset = BERT_CONFIG_PRESETS[presetName];
  const selectedMeta = bertComponents[selected];

  // Tokens (without masking) — used for the mask picker controls.
  const tokensNoMask = useMemo(
    () => bertTokenize(textA, textB),
    [textA, textB]
  );

  const handleSelect = (id: BertComponentId) => {
    setSelected(id);
    setRightTab("inspector");
  };

  const modelLabel = (() => {
    switch (presetName) {
      case "BERT-base":
        return { brand: "BERT", model: "base (110M)" };
      case "BERT-large":
        return { brand: "BERT", model: "large (340M)" };
      case "DistilBERT":
        return { brand: "DistilBERT", model: "base (66M)" };
      case "RoBERTa-base":
        return { brand: "RoBERTa", model: "base (125M)" };
    }
  })();

  return (
    <div className="flex flex-col gap-6">
      {/* Branded header */}
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-cyan-300 font-semibold">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
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
            <Link to="/llm" className="btn">
              <ArrowLeft className="w-4 h-4" /> Back to models
            </Link>
            <a
              href="https://arxiv.org/abs/1810.04805"
              target="_blank"
              rel="noreferrer"
              className="btn"
            >
              <FileText className="w-4 h-4" /> Paper
            </a>
            <Link to="/llm/gpt" className="btn">
              <ExternalLink className="w-4 h-4" /> Compare to GPT
            </Link>
          </div>
        </div>
      </header>

      {/* Input + preset + masking controls */}
      <section className="card p-3.5 lg:p-4 flex flex-col gap-3">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          <LabeledInput
            label="Sentence A"
            value={textA}
            onChange={setTextA}
          />
          <LabeledInput
            label="Sentence B (optional)"
            value={textB}
            onChange={setTextB}
          />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-3 items-center">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold">
              Mask token
            </span>
            <div className="flex flex-wrap gap-1.5">
              {tokensNoMask.map((t, i) =>
                t.isSpecial ? null : (
                  <button
                    key={i}
                    onClick={() =>
                      setMaskedIndex(maskedIndex === i ? null : i)
                    }
                    className={[
                      "px-2 py-1 rounded-md text-[11px] font-mono border transition-all",
                      maskedIndex === i
                        ? "bg-orange-400/20 border-orange-400 text-orange-200 shadow-[0_0_0_2px_rgba(251,146,60,0.25)]"
                        : "bg-ink-900/70 border-ink-700 text-ink-300 hover:border-orange-400/40",
                    ].join(" ")}
                  >
                    {t.token}
                  </button>
                )
              )}
              {maskedIndex !== null && (
                <button
                  onClick={() => setMaskedIndex(null)}
                  className="px-2 py-1 rounded-md text-[11px] font-medium border bg-ink-900/70 border-ink-700 text-ink-300 hover:border-accent/40 flex items-center gap-1"
                >
                  <EyeOff className="w-3 h-3" /> clear
                </button>
              )}
              <button
                onClick={() => {
                  const valid = tokensNoMask
                    .map((t, i) => (t.isSpecial ? -1 : i))
                    .filter((i) => i >= 0);
                  if (!valid.length) return;
                  setMaskedIndex(valid[Math.floor(Math.random() * valid.length)]);
                }}
                className="px-2 py-1 rounded-md text-[11px] font-medium border bg-ink-900/70 border-ink-700 text-ink-300 hover:border-orange-400/40 flex items-center gap-1"
              >
                <Shuffle className="w-3 h-3" /> random
              </button>
            </div>
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            {(Object.keys(BERT_CONFIG_PRESETS) as PresetName[]).map((p) => (
              <button
                key={p}
                onClick={() => setPresetName(p)}
                className={[
                  "px-2.5 py-1.5 rounded-lg text-[11px] font-medium border transition-all",
                  presetName === p
                    ? "bg-cyan-500 text-white border-cyan-500 shadow-[0_0_0_3px_rgba(34,211,238,0.18)]"
                    : "bg-ink-900/70 border-ink-700 text-ink-200 hover:border-cyan-400/40",
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

        <div className="grid grid-cols-1 xl:grid-cols-[1fr_440px] gap-0">
          <div className="relative border-b xl:border-b-0 xl:border-r border-ink-800/70 min-h-[640px]">
            <div className="absolute top-3 right-3 z-10 flex flex-col gap-1.5">
              <CanvasIconBtn icon={<Search className="w-3.5 h-3.5" />} title="Search" />
              <CanvasIconBtn icon={<ZoomIn className="w-3.5 h-3.5" />} title="Zoom in" />
              <CanvasIconBtn icon={<ZoomOut className="w-3.5 h-3.5" />} title="Zoom out" />
              <CanvasIconBtn icon={<Maximize2 className="w-3.5 h-3.5" />} title="Fit" />
              <CanvasIconBtn icon={<Focus className="w-3.5 h-3.5" />} title="Center selection" />
            </div>
            <div className="p-5">
              <BertGraphCanvas
                granularity={granularity}
                numLayers={preset.layers}
                selected={selected}
                onSelect={handleSelect}
              />
            </div>
            <div className="absolute bottom-3 left-4 right-4 flex items-center justify-between text-[10px] text-ink-500 font-mono pointer-events-none">
              <span>
                selected: <span className="text-ink-200">{selectedMeta?.full}</span>
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
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shrink-0" />
                  <span className="truncate">{selectedMeta?.short}</span>
                </div>
              )}
            </div>

            <div className="flex-1 overflow-y-auto p-4 lg:p-5">
              {rightTab === "inspector" ? (
                <BertDetail
                  selected={selected}
                  textA={textA}
                  textB={textB}
                  numHeads={preset.heads}
                  dModel={preset.d_model}
                  maskedIndex={maskedIndex}
                  embedded
                />
              ) : (
                <BertSidebar
                  modelName={`${modelLabel.brand}/${modelLabel.model} model`}
                  numLayers={preset.layers}
                  dModel={preset.d_model}
                  numHeads={preset.heads}
                  ffnDim={preset.ffn}
                  vocabSize={preset.vocab}
                  contextLength={preset.ctx}
                />
              )}
            </div>

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

function LabeledInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold w-24 shrink-0">
        {label}
      </span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="flex-1 px-3 py-2 rounded-lg bg-ink-950/70 border border-ink-700 focus:border-cyan-400/60 focus:outline-none font-mono text-sm"
      />
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
          ? "bg-cyan-500 text-white shadow-[0_0_0_3px_rgba(34,211,238,0.18)]"
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
          ? "bg-cyan-500 text-white shadow-[0_0_0_3px_rgba(34,211,238,0.18)]"
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
      className="w-7 h-7 rounded-md bg-ink-900/80 border border-ink-700 text-ink-300 hover:text-white hover:border-cyan-400/50 hover:bg-ink-800/80 flex items-center justify-center transition-all backdrop-blur"
    >
      {icon}
    </button>
  );
}
