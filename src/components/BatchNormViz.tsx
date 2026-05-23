import { useMemo, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  batchNorm,
  groupNorm,
  layerNorm,
  makeBatch,
  rmsNorm,
  simulateRunningStats,
  type Stats,
} from "../lib/batchNorm";

type NormMode = "none" | "batch" | "layer" | "group" | "rms";

const B = 8;
const F = 6;
const EPS = 1e-5;

export default function BatchNormViz() {
  const [seed, setSeed] = useState(7);
  const [mode, setMode] = useState<NormMode>("batch");
  const [gamma, setGamma] = useState<number[]>(Array(F).fill(1));
  const [beta, setBeta] = useState<number[]>(Array(F).fill(0));
  const [selectedFeature, setSelectedFeature] = useState(0);

  const x = useMemo(() => makeBatch(B, F, seed), [seed]);

  const result = useMemo(() => {
    if (mode === "batch") return batchNorm(x, gamma, beta, EPS);
    if (mode === "layer") return layerNorm(x, gamma, beta, EPS);
    if (mode === "group") return groupNorm(x, 2, gamma, beta, EPS);
    if (mode === "rms") return rmsNorm(x, gamma, EPS);
    return null;
  }, [x, gamma, beta, mode]);

  const y = result ? result.y : x;
  const xhat = result ? result.xhat : x;
  const batchStats = useMemo<Stats[] | null>(() => {
    if (mode !== "batch") return null;
    return (result as ReturnType<typeof batchNorm>).stats;
  }, [result, mode]);

  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <ControlBar
        mode={mode}
        onMode={setMode}
        seed={seed}
        onReseed={() => setSeed((s) => s + 1)}
        gamma={gamma}
        beta={beta}
        onGamma={setGamma}
        onBeta={setBeta}
        selectedFeature={selectedFeature}
        onSelectFeature={setSelectedFeature}
      />

      <BatchMatrixView
        x={x}
        xhat={xhat}
        y={y}
        mode={mode}
        batchStats={batchStats}
        selectedFeature={selectedFeature}
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DistributionPanel x={x} y={y} selectedFeature={selectedFeature} mode={mode} />
        <RunningStatsPanel seed={seed} />
      </div>

      <PropertiesGrid />
      <UsageGrid />
      <FamilyComparison />
    </div>
  );
}

/* =============================== top =============================== */

function DefinitionCard() {
  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4">
      <div className="flex flex-col sm:flex-row sm:items-baseline gap-2 sm:gap-3">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold shrink-0">
          Definition
        </span>
        <p className="text-ink-100 text-sm leading-relaxed">
          <span className="text-indigo-300">Batch Normalization</span> standardizes each
          activation channel across the mini-batch to zero mean and unit variance, then
          re-introduces representational capacity with two learned per-channel
          parameters <InlineMath math={String.raw`\gamma, \beta`} />. Train uses batch
          statistics; eval uses a running EMA of those statistics.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-2 gap-3">
      <MathBox>
        <BlockMath math={String.raw`\mu_{j} = \tfrac{1}{m}\sum_{i=1}^{m} x_{ij},\quad \sigma_{j}^{2} = \tfrac{1}{m}\sum_{i=1}^{m}(x_{ij}-\mu_{j})^{2}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\hat{x}_{ij} = \frac{x_{ij}-\mu_{j}}{\sqrt{\sigma_{j}^{2}+\varepsilon}},\quad y_{ij} = \gamma_{j}\hat{x}_{ij} + \beta_{j}`} />
      </MathBox>
    </section>
  );
}

/* =============================== controls =============================== */

function ControlBar({
  mode,
  onMode,
  seed,
  onReseed,
  gamma,
  beta,
  onGamma,
  onBeta,
  selectedFeature,
  onSelectFeature,
}: {
  mode: NormMode;
  onMode: (m: NormMode) => void;
  seed: number;
  onReseed: () => void;
  gamma: number[];
  beta: number[];
  onGamma: (v: number[]) => void;
  onBeta: (v: number[]) => void;
  selectedFeature: number;
  onSelectFeature: (i: number) => void;
}) {
  const modes: { id: NormMode; label: string }[] = [
    { id: "none", label: "Raw" },
    { id: "batch", label: "Batch Norm" },
    { id: "layer", label: "Layer Norm" },
    { id: "group", label: "Group Norm (G=2)" },
    { id: "rms", label: "RMS Norm" },
  ];

  const setG = (i: number, v: number) => onGamma(gamma.map((g, j) => (j === i ? v : g)));
  const setB = (i: number, v: number) => onBeta(beta.map((b, j) => (j === i ? v : b)));

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold pr-1">
            Normalization
          </span>
          {modes.map((m) => (
            <button
              key={m.id}
              onClick={() => onMode(m.id)}
              className={`px-2.5 py-1 rounded text-[11.5px] font-mono border transition-colors ${
                mode === m.id
                  ? "border-indigo-400/60 text-white bg-indigo-500/15"
                  : "border-ink-700 text-ink-400 hover:text-white"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <button
          onClick={onReseed}
          className="px-2.5 py-1 rounded text-[11px] font-mono border border-ink-700 text-ink-400 hover:text-white hover:border-indigo-400/50"
        >
          ↻ resample (seed {seed})
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-3 items-start">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold pt-2">
          γ, β per feature
        </span>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {Array.from({ length: F }, (_, j) => {
            const selected = j === selectedFeature;
            return (
              <div
                key={j}
                onClick={() => onSelectFeature(j)}
                className={`rounded-lg border p-2.5 cursor-pointer transition-colors ${
                  selected
                    ? "border-indigo-400/60 bg-indigo-500/5"
                    : "border-ink-800 hover:border-ink-700"
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
                    Feature j = {j}
                  </span>
                  {selected && (
                    <span className="text-[9px] uppercase tracking-wider text-indigo-300 font-bold">
                      selected
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 font-mono text-[11px]">
                  <span className="w-6 text-ink-400">γ</span>
                  <input
                    type="range"
                    min={-2}
                    max={2}
                    step={0.05}
                    value={gamma[j]}
                    onChange={(e) => setG(j, parseFloat(e.target.value))}
                    onClick={(e) => e.stopPropagation()}
                    className="flex-1 accent-indigo-400"
                  />
                  <span className="w-10 text-right tabular-nums">{gamma[j].toFixed(2)}</span>
                </div>
                <div className="flex items-center gap-2 font-mono text-[11px]">
                  <span className="w-6 text-ink-400">β</span>
                  <input
                    type="range"
                    min={-2}
                    max={2}
                    step={0.05}
                    value={beta[j]}
                    onChange={(e) => setB(j, parseFloat(e.target.value))}
                    onClick={(e) => e.stopPropagation()}
                    className="flex-1 accent-cyan-400"
                  />
                  <span className="w-10 text-right tabular-nums">{beta[j].toFixed(2)}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

/* =============================== batch matrix view =============================== */

function BatchMatrixView({
  x,
  xhat,
  y,
  mode,
  batchStats,
  selectedFeature,
}: {
  x: number[][];
  xhat: number[][];
  y: number[][];
  mode: NormMode;
  batchStats: Stats[] | null;
  selectedFeature: number;
}) {
  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-4">
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
        Mini-batch · {B} samples × {F} features
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <MatrixBlock
          title="x (input)"
          accent="text-ink-300"
          matrix={x}
          highlightCol={mode === "batch" ? selectedFeature : undefined}
          highlightRow={mode === "layer" || mode === "rms" ? 0 : undefined}
        />
        <MatrixBlock
          title={mode === "none" ? "x̂ (n/a)" : "x̂ (normalized)"}
          accent="text-violet-300"
          matrix={mode === "none" ? x : xhat}
          highlightCol={mode === "batch" ? selectedFeature : undefined}
        />
        <MatrixBlock
          title={mode === "none" ? "y" : mode === "rms" ? "y = γ x̂" : "y = γ x̂ + β"}
          accent="text-cyan-300"
          matrix={y}
          highlightCol={mode === "batch" ? selectedFeature : undefined}
        />
      </div>

      {batchStats && (
        <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-3">
          <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold mb-2">
            Per-feature batch statistics (column-wise)
          </div>
          <div className="grid grid-cols-[auto_repeat(6,1fr)] gap-x-2 gap-y-1 text-[11.5px] font-mono items-baseline">
            <span className="text-ink-500">feature j</span>
            {batchStats.map((_, j) => (
              <span
                key={j}
                className={`text-center ${j === selectedFeature ? "text-indigo-300 font-bold" : "text-ink-400"}`}
              >
                {j}
              </span>
            ))}
            <span className="text-ink-500">μ_j</span>
            {batchStats.map((s, j) => (
              <span
                key={j}
                className={`text-center tabular-nums ${j === selectedFeature ? "text-indigo-300 font-bold" : "text-ink-200"}`}
              >
                {s.mean.toFixed(2)}
              </span>
            ))}
            <span className="text-ink-500">σ_j</span>
            {batchStats.map((s, j) => (
              <span
                key={j}
                className={`text-center tabular-nums ${j === selectedFeature ? "text-indigo-300 font-bold" : "text-ink-200"}`}
              >
                {s.std.toFixed(2)}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function MatrixBlock({
  title,
  accent,
  matrix,
  highlightCol,
  highlightRow,
}: {
  title: string;
  accent: string;
  matrix: number[][];
  highlightCol?: number;
  highlightRow?: number;
}) {
  // map values to a divergent color: negative violet, positive cyan, zero dark.
  const flat = matrix.flat();
  const maxAbs = Math.max(1e-6, ...flat.map((v) => Math.abs(v)));

  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-3">
      <div className={`text-[11px] font-mono mb-2 font-semibold ${accent}`}>{title}</div>
      <div className="overflow-auto">
        <table className="w-full text-[10.5px] font-mono tabular-nums">
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i} className={highlightRow === i ? "bg-violet-500/10" : ""}>
                {row.map((v, j) => {
                  const t = v / maxAbs;
                  const isCol = highlightCol === j;
                  const color = cellColor(t);
                  return (
                    <td
                      key={j}
                      className="p-0"
                      style={{
                        background: color,
                        outline: isCol ? "1px solid rgba(124,92,255,0.7)" : undefined,
                        outlineOffset: "-1px",
                      }}
                    >
                      <div className="px-1.5 py-1 text-center" style={{ color: textFor(t) }}>
                        {v.toFixed(2)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function cellColor(t: number): string {
  // t in [-1, 1]. Negative → violet, positive → cyan
  if (t >= 0) {
    const a = Math.min(t, 1);
    const r = Math.round(11 + (34 - 11) * a);
    const g = Math.round(13 + (211 - 13) * a);
    const b = Math.round(18 + (238 - 18) * a);
    return `rgba(${r}, ${g}, ${b}, ${0.18 + 0.55 * a})`;
  }
  const a = Math.min(-t, 1);
  const r = Math.round(11 + (124 - 11) * a);
  const g = Math.round(13 + (92 - 13) * a);
  const b = Math.round(18 + (255 - 18) * a);
  return `rgba(${r}, ${g}, ${b}, ${0.18 + 0.55 * a})`;
}

function textFor(t: number): string {
  const a = Math.min(Math.abs(t), 1);
  return a > 0.55 ? "#ffffff" : "#cbd5e1";
}

/* =============================== distribution panel =============================== */

function DistributionPanel({
  x,
  y,
  selectedFeature,
  mode,
}: {
  x: number[][];
  y: number[][];
  selectedFeature: number;
  mode: NormMode;
}) {
  const before = x.map((r) => r[selectedFeature]);
  const after = y.map((r) => r[selectedFeature]);
  const meanB = avg(before);
  const stdB = Math.sqrt(avg(before.map((v) => (v - meanB) ** 2)));
  const meanA = avg(after);
  const stdA = Math.sqrt(avg(after.map((v) => (v - meanA) ** 2)));

  const W = 380;
  const H = 220;
  const PAD = { l: 32, r: 12, t: 14, b: 24 };
  const all = [...before, ...after];
  const lo = Math.min(-1, ...all) - 0.5;
  const hi = Math.max(1, ...all) + 0.5;
  const sx = (v: number) => PAD.l + ((v - lo) / (hi - lo)) * (W - PAD.l - PAD.r);
  const sy = (i: number) => PAD.t + i * 22;

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Feature {selectedFeature} · before vs after
        </span>
        <span className="text-[10px] font-mono text-ink-500">
          {mode === "none" ? "no normalization" : modeLabel(mode)}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        {/* x axis */}
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke="rgba(255,255,255,0.1)" />
        {[-3, -2, -1, 0, 1, 2, 3].filter((v) => v > lo && v < hi).map((v) => (
          <g key={v}>
            <line x1={sx(v)} y1={H - PAD.b} x2={sx(v)} y2={H - PAD.b + 3} stroke="rgba(255,255,255,0.1)" />
            <text x={sx(v)} y={H - PAD.b + 13} textAnchor="middle" fontSize={9} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
              {v}
            </text>
          </g>
        ))}
        {/* before row */}
        <text x={PAD.l - 4} y={sy(1) + 4} textAnchor="end" fontSize={10} fill="#a78bfa" fontFamily="JetBrains Mono, monospace">x</text>
        {before.map((v, i) => (
          <circle key={`b-${i}`} cx={sx(v)} cy={sy(1)} r={4} fill="#7c5cff" opacity={0.85} />
        ))}
        <line x1={sx(meanB - stdB)} y1={sy(1)} x2={sx(meanB + stdB)} y2={sy(1)} stroke="#7c5cff" strokeOpacity={0.4} strokeWidth={2} />
        <circle cx={sx(meanB)} cy={sy(1)} r={5} fill="none" stroke="#fff" strokeWidth={1.5} />
        <text x={sx(meanB) + 8} y={sy(1) - 6} fontSize={9} fill="#fff" fontFamily="JetBrains Mono, monospace">
          μ={meanB.toFixed(2)}, σ={stdB.toFixed(2)}
        </text>

        {/* after row */}
        <text x={PAD.l - 4} y={sy(4) + 4} textAnchor="end" fontSize={10} fill="#67e8f9" fontFamily="JetBrains Mono, monospace">y</text>
        {after.map((v, i) => (
          <circle key={`a-${i}`} cx={sx(v)} cy={sy(4)} r={4} fill="#22d3ee" opacity={0.85} />
        ))}
        <line x1={sx(meanA - stdA)} y1={sy(4)} x2={sx(meanA + stdA)} y2={sy(4)} stroke="#22d3ee" strokeOpacity={0.4} strokeWidth={2} />
        <circle cx={sx(meanA)} cy={sy(4)} r={5} fill="none" stroke="#fff" strokeWidth={1.5} />
        <text x={sx(meanA) + 8} y={sy(4) - 6} fontSize={9} fill="#fff" fontFamily="JetBrains Mono, monospace">
          μ={meanA.toFixed(2)}, σ={stdA.toFixed(2)}
        </text>
      </svg>
      <p className="text-[11px] text-ink-400 leading-snug">
        Dots are individual samples; horizontal bars span <InlineMath math={String.raw`\mu \pm \sigma`} />.
        BN&apos;s job: align <InlineMath math={String.raw`y`} /> to{" "}
        <InlineMath math={String.raw`\mu \approx \beta,\;\sigma \approx |\gamma|`} />.
      </p>
    </section>
  );
}

function avg(a: number[]): number {
  return a.reduce((s, v) => s + v, 0) / a.length;
}

function modeLabel(m: NormMode): string {
  switch (m) {
    case "batch":
      return "BatchNorm";
    case "layer":
      return "LayerNorm";
    case "group":
      return "GroupNorm";
    case "rms":
      return "RMSNorm";
    default:
      return "raw";
  }
}

/* =============================== running stats panel =============================== */

function RunningStatsPanel({ seed }: { seed: number }) {
  const batches = useMemo(() => {
    const list: number[][][] = [];
    for (let k = 0; k < 60; k++) list.push(makeBatch(B, F, seed * 100 + k));
    return list;
  }, [seed]);
  const ema = useMemo(() => simulateRunningStats(batches, 0.1), [batches]);

  const W = 380;
  const H = 220;
  const PAD = { l: 36, r: 12, t: 14, b: 24 };
  const T = ema.runningMean.length;
  const all = ema.runningMean.flat();
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const sx = (t: number) => PAD.l + (t / (T - 1)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - lo) / (hi - lo + 1e-6)) * (H - PAD.t - PAD.b);

  const colors = ["#7c5cff", "#22d3ee", "#34d399", "#f472b6", "#fbbf24", "#818cf8"];

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Train vs Eval · running mean μ̄_j via EMA
        </span>
        <span className="text-[10px] font-mono text-ink-500">momentum 0.1 · 60 batches</span>
      </div>
      <BlockMath math={String.raw`\bar\mu_{j} \leftarrow (1-\alpha)\,\bar\mu_{j} + \alpha\,\mu_{j}^{(B)}`} />
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke="rgba(255,255,255,0.1)" />
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke="rgba(255,255,255,0.1)" />
        {/* zero line */}
        {lo < 0 && hi > 0 && (
          <line x1={PAD.l} y1={sy(0)} x2={W - PAD.r} y2={sy(0)} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 4" />
        )}
        {Array.from({ length: F }, (_, j) => {
          const d = ema.runningMean
            .map((m, t) => `${t === 0 ? "M" : "L"} ${sx(t).toFixed(2)} ${sy(m[j]).toFixed(2)}`)
            .join(" ");
          return <path key={j} d={d} stroke={colors[j]} strokeWidth={1.6} fill="none" />;
        })}
        <text x={W - PAD.r} y={H - PAD.b - 4} textAnchor="end" fontSize={9} fill="#5b6478" fontFamily="JetBrains Mono, monospace">
          step
        </text>
      </svg>
      <p className="text-[11px] text-ink-400 leading-snug">
        Each colored line is one feature&apos;s running mean — the value frozen into{" "}
        <span className="text-indigo-300">eval mode</span> for inference. Variance is
        tracked identically.
      </p>
    </section>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="Per-channel">
        <InlineMath math={String.raw`\text{One }(\gamma, \beta)\text{ per feature; shared across batch.}`} />
      </PropCard>
      <PropCard title="Invariant to affine recale">
        <InlineMath math={String.raw`\mathrm{BN}(ax+b) = \mathrm{BN}(x)`} />
      </PropCard>
      <PropCard title="Train vs eval">
        <InlineMath math={String.raw`\text{train: } \mu_{B},\sigma_{B}\;\;|\;\;\text{eval: } \bar\mu,\bar\sigma`} />
      </PropCard>
      <PropCard title="Param cost">
        <InlineMath math={String.raw`2F\;\text{params}; \; 2F\;\text{running buffers}`} />
      </PropCard>
      <PropCard title="Gradient w.r.t. γ">
        <InlineMath math={String.raw`\partial L / \partial \gamma_{j} = \sum_{i}(\partial L/\partial y_{ij})\,\hat x_{ij}`} />
      </PropCard>
      <PropCard title="Acts as regularizer">
        <InlineMath math={String.raw`\text{Batch noise} \Rightarrow \text{implicit regularization.}`} />
      </PropCard>
    </section>
  );
}

function PropCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-3.5">
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold mb-2">
        {title}
      </div>
      <div className="text-ink-100 text-base">{children}</div>
    </div>
  );
}

/* =============================== usage =============================== */

function UsageGrid() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <UsageCard title="Where used" accent="text-indigo-300">
        <UsageItem head="CNN backbones" body="ResNet, Inception, EfficientNet — BN between conv and ReLU." />
        <UsageItem head="Object detection / segmentation" body="usually inherited from a BN-pretrained backbone." />
        <UsageItem head="GANs" body="BN in generators (with care); discriminators often use Spectral / Instance Norm." />
        <UsageItem head="Style transfer" body="Instance Norm (per-sample / per-channel) replaces BN." />
        <UsageItem head="Transformers / LLMs" body="LayerNorm (or RMSNorm in modern LLMs) — BN rarely used." />
        <UsageItem head="Tabular / MLPs" body="LayerNorm or BN — both common; BN if batches are large." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-cyan-300">
        <UsageItem
          head="Faster convergence"
          body="bigger learning rates become safe; loss surface gets smoother (Santurkar et al., 2018)."
        />
        <UsageItem
          head="Reduces internal covariate shift"
          body="the original (debated) motivation: stabilize layer-input distributions."
        />
        <UsageItem
          head="Implicit regularization"
          body="batch noise has a dropout-like effect; sometimes obviates dropout."
        />
        <UsageItem
          head="Decouples scale"
          body={
            <>
              Activations stay <InlineMath math={String.raw`\mathcal{O}(1)`} /> regardless of weights — prevents
              activation blow-up.
            </>
          }
        />
        <UsageItem
          head="Recoverable identity"
          body={
            <>
              Setting <InlineMath math={String.raw`\gamma=\sigma, \beta=\mu`} /> recovers the input ⇒ no
              loss of capacity.
            </>
          }
        />
        <UsageItem
          head="Composable"
          body="works under any optimizer; one of the few interventions that everyone agrees on."
        />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem
          head="Batch-size dependent"
          body="small / micro-batches make μ̂_B, σ̂_B noisy — GroupNorm / LayerNorm preferred there."
        />
        <UsageItem
          head="Train ≠ eval"
          body="running stats can drift from true distribution; BN can hurt at inference time."
        />
        <UsageItem
          head="Synchronization cost"
          body="distributed training needs SyncBN across devices for correct stats."
        />
        <UsageItem
          head="Bad for RNNs / sequences"
          body="batch stats change with sequence length; LayerNorm wins."
        />
        <UsageItem
          head="Information leakage"
          body="examples within a batch interact through μ, σ — bad for contrastive / RL."
        />
        <UsageItem
          head="Not transferable"
          body="domain shift at test time invalidates running stats."
        />
      </UsageCard>
    </section>
  );
}

function UsageCard({
  title,
  accent,
  children,
}: {
  title: string;
  accent: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-2.5">
      <div className={`text-[10px] uppercase tracking-wider font-bold ${accent}`}>
        {title}
      </div>
      {children}
    </div>
  );
}

function UsageItem({ head, body }: { head: string; body: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 border-l-2 border-ink-800 pl-2.5">
      <div className="text-ink-100 text-[12px] font-mono font-semibold">{head}</div>
      <div className="text-ink-400 text-[11.5px] leading-snug">{body}</div>
    </div>
  );
}

/* =============================== family comparison =============================== */

function FamilyComparison() {
  const SCHEMES: { id: string; name: string; formula: string; pattern: (i: number, j: number) => boolean }[] = [
    {
      id: "bn",
      name: "BatchNorm",
      formula: String.raw`\mu_j, \sigma_j\;\text{over batch axis}`,
      pattern: (_i, j) => j === 0, // highlight column 0
    },
    {
      id: "ln",
      name: "LayerNorm",
      formula: String.raw`\mu_i, \sigma_i\;\text{over feature axis}`,
      pattern: (i, _j) => i === 0, // highlight row 0
    },
    {
      id: "in",
      name: "InstanceNorm",
      formula: String.raw`\text{(spatial NLP-flat: one cell per }(i,j)\text{)}`,
      pattern: (i, j) => i === 0 && j === 0,
    },
    {
      id: "gn",
      name: "GroupNorm (G=2)",
      formula: String.raw`\text{groups of features per sample}`,
      pattern: (i, j) => i === 0 && j < 3, // row 0, first half features
    },
    {
      id: "rms",
      name: "RMSNorm",
      formula: String.raw`\text{rms}_i\;\text{over feature axis, no centering}`,
      pattern: (i, _j) => i === 0,
    },
  ];

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
        Normalization family · which cells share statistics
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        {SCHEMES.map((s) => (
          <div key={s.id} className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-3 flex flex-col gap-2">
            <div className="flex items-baseline justify-between">
              <span className="text-[11px] text-ink-100 font-mono font-semibold">{s.name}</span>
            </div>
            <SchemeGrid pattern={s.pattern} />
            <div className="text-[11px] text-ink-400 leading-snug">
              <InlineMath math={s.formula} />
            </div>
          </div>
        ))}
      </div>
      <p className="text-[11px] text-ink-400 leading-snug">
        Highlighted cells share one set of <InlineMath math={String.raw`(\mu, \sigma)`} />. BN aggregates
        across the <span className="text-indigo-300">batch</span>; LN / RMS across the{" "}
        <span className="text-violet-300">features</span>; GN within feature groups;
        InstanceNorm not at all in a flat (B, F) layout (it shines on B×C×HW).
      </p>
    </section>
  );
}

function SchemeGrid({ pattern }: { pattern: (i: number, j: number) => boolean }) {
  const Br = 5;
  const Fc = 6;
  return (
    <div
      className="grid gap-[2px]"
      style={{ gridTemplateColumns: `repeat(${Fc}, 1fr)` }}
    >
      {Array.from({ length: Br }, (_, i) =>
        Array.from({ length: Fc }, (_, j) => {
          const on = pattern(i, j);
          return (
            <div
              key={`${i}-${j}`}
              style={{
                aspectRatio: "1 / 1",
                background: on ? "rgba(124,92,255,0.8)" : "rgba(255,255,255,0.05)",
                border: on ? "1px solid #a78bfa" : "1px solid rgba(255,255,255,0.07)",
                borderRadius: 2,
              }}
            />
          );
        })
      )}
    </div>
  );
}

/* =============================== shared =============================== */

function MathBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-3 flex items-center justify-center">
      {children}
    </div>
  );
}
