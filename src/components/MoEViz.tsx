import { useMemo, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  MOE_PRESETS,
  activationRatio,
  activeMoEParams,
  expertUtilization,
  loadBalanceLoss,
  randomMatrix,
  routerScores,
  topKGate,
  totalMoEParams,
  type MoEConfig,
} from "../lib/moe";

const C_AXIS = "rgba(255,255,255,0.10)";
const C_GATE = "#a78bfa";
const C_ACTIVE = "#22d3ee";
const C_INACTIVE = "#3b3f55";
const C_HOT = "#f472b6";
const C_OK = "#34d399";
const C_BAD = "#fbbf24";

const TOKEN_NAMES = ["the", "cat", "sat", "matrix", "GPU", "loss", "gradient", "neural"];

export default function MoEViz() {
  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <RouterLab />
      <ParamsCalculator />
      <ModelTable />

      <PropertiesGrid />
      <UsageGrid />
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
          A <span className="text-violet-300">Mixture of Experts</span> layer
          replaces the dense FFN with{" "}
          <span className="text-cyan-300">N expert FFNs</span> plus a small{" "}
          <em>router</em>. Each token's representation goes through{" "}
          <em>only the top-k experts</em> chosen by the router. Total parameter
          count is N×, but compute per token stays at k× — letting you build
          trillion-parameter models that run at the cost of a ~50B dense model.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`g(x) = \mathrm{softmax}(W_g\,x)\in\mathbb{R}^{N}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`y = \sum_{e\in\mathrm{top}_k(g)} \tilde{g}_e \cdot E_e(x)`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`L_{\text{aux}} = \alpha N \sum_e f_e P_e`} />
      </MathBox>
    </section>
  );
}

/* =============================== router lab =============================== */

function RouterLab() {
  const dModel = 8;
  const [nExperts, setNExperts] = useState(8);
  const [topK, setTopK] = useState(2);
  const [seed, setSeed] = useState(7);

  const tokens = useMemo(() => randomMatrix(TOKEN_NAMES.length, dModel, seed), [seed]);
  const gates = useMemo(() => randomMatrix(nExperts, dModel, seed + 1000), [nExperts, seed]);

  const scoresFull = useMemo(() => routerScores(tokens, gates), [tokens, gates]);
  const scoresGated = useMemo(() => scoresFull.map((row) => topKGate(row, topK)), [scoresFull, topK]);
  const util = useMemo(() => expertUtilization(scoresGated), [scoresGated]);
  const aux = useMemo(() => loadBalanceLoss(scoresFull, scoresGated, 0.01), [scoresFull, scoresGated]);

  // Imbalance: max / mean utilization. 1.0 = perfectly balanced.
  const mean = util.reduce((a, b) => a + b, 0) / Math.max(util.length, 1);
  const imbalance = mean > 0 ? Math.max(...util) / mean : 0;

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Router · top-k gating · utilization
        </span>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="text-[10px] font-mono text-ink-400 hover:text-white border border-ink-700 px-2 py-0.5 rounded"
        >
          re-roll tokens & gates
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <SliderRow label="# experts N" value={nExperts} onChange={(v) => { const n = Math.round(v); setNExperts(n); setTopK(Math.min(topK, n)); }} min={2} max={16} step={1} fmt={(v) => v.toFixed(0)} color={C_ACTIVE} />
        <SliderRow label="top-k" value={topK} onChange={(v) => setTopK(Math.min(Math.round(v), nExperts))} min={1} max={Math.min(nExperts, 6)} step={1} fmt={(v) => v.toFixed(0)} color={C_HOT} />
      </div>

      <RouterHeatmap scoresGated={scoresGated} scoresFull={scoresFull} tokenNames={TOKEN_NAMES} />

      <UtilizationBars util={util} />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 font-mono text-[12px]">
        <Stat label="active experts / token" value={`${topK} of ${nExperts}`} color={C_ACTIVE} />
        <Stat label="capacity utilization" value={`${(activationRatio({ dModel: 1, dFF: 1, nExperts, topK, nLayers: 1 }) * 100).toFixed(0)}%`} color={C_ACTIVE} hint="k / N" />
        <Stat label="imbalance (max / mean)" value={`${imbalance.toFixed(2)}×`} color={imbalance < 1.6 ? C_OK : C_BAD} hint="1.00 = balanced" />
        <Stat label="L_aux (α=0.01)" value={aux.toExponential(2)} color={C_GATE} hint="add to main loss" />
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        The router is a tiny linear layer (<InlineMath math={String.raw`W_g \in \mathbb{R}^{N\times d}`} />)
        that scores every expert per token, picks the top-k, and routes the
        token's hidden state through them. Without the load-balance loss, the
        gradient gradient quickly concentrates on a few experts — wasting most
        of the parameters.
      </p>
    </section>
  );
}

function RouterHeatmap({
  scoresGated,
  scoresFull,
  tokenNames,
}: {
  scoresGated: number[][];
  scoresFull: number[][];
  tokenNames: string[];
}) {
  const T = scoresGated.length;
  const N = scoresGated[0]?.length ?? 0;
  const W = 720;
  const H = Math.max(120, 24 + T * 28);
  const PAD = { l: 72, r: 8, t: 18, b: 28 };
  const innerW = W - PAD.l - PAD.r;
  const cellW = innerW / N;
  const cellH = (H - PAD.t - PAD.b) / T;

  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <text x={PAD.l - 8} y={PAD.t - 4} fontSize={10} fill="#7a8094" textAnchor="end" fontFamily="JetBrains Mono, monospace">
          token
        </text>
        <text x={PAD.l + 4} y={PAD.t - 4} fontSize={10} fill={C_GATE} fontFamily="JetBrains Mono, monospace" fontWeight={700}>
          expert →
        </text>

        {scoresGated.map((row, t) => (
          <g key={t}>
            <text x={PAD.l - 6} y={PAD.t + t * cellH + cellH * 0.65} fontSize={10} fill="#cbd5e1" textAnchor="end" fontFamily="JetBrains Mono, monospace">
              {tokenNames[t] ?? `t${t}`}
            </text>
            {row.map((v, e) => {
              const active = v > 1e-6;
              const full = scoresFull[t][e];
              const fill = active
                ? `rgba(34,211,238,${0.25 + 0.75 * v})`
                : `rgba(255,255,255,${0.04 + 0.18 * full})`;
              return (
                <g key={e}>
                  <rect
                    x={PAD.l + e * cellW + 1}
                    y={PAD.t + t * cellH + 1}
                    width={cellW - 2}
                    height={cellH - 2}
                    rx={3}
                    fill={fill}
                    stroke={active ? C_ACTIVE : "transparent"}
                    strokeOpacity={active ? 0.7 : 0}
                  />
                  {active && (
                    <text x={PAD.l + e * cellW + cellW / 2} y={PAD.t + t * cellH + cellH * 0.65} textAnchor="middle" fontSize={9} fill="#0b0f19" fontFamily="JetBrains Mono, monospace" fontWeight={700}>
                      {(v * 100).toFixed(0)}%
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        ))}

        {Array.from({ length: N }, (_, e) => (
          <text key={e} x={PAD.l + e * cellW + cellW / 2} y={H - PAD.b + 12} textAnchor="middle" fontSize={9} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
            E{e}
          </text>
        ))}
      </svg>
    </div>
  );
}

function UtilizationBars({ util }: { util: number[] }) {
  const max = Math.max(...util, 1e-9);
  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2.5 flex flex-col gap-1">
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold mb-1">
        Per-expert utilization (mean gate weight across tokens)
      </div>
      <div className="flex flex-col gap-1">
        {util.map((u, e) => {
          const pct = u / max;
          const isHot = pct > 0.85;
          const isCold = pct < 0.1;
          return (
            <div key={e} className="flex items-center gap-2 font-mono text-[11px]">
              <span className="w-10 text-ink-500">E{e}</span>
              <div className="flex-1 h-3.5 rounded bg-ink-950/80 border border-ink-800/60 overflow-hidden">
                <div
                  className="h-full rounded"
                  style={{
                    width: `${pct * 100}%`,
                    background: isHot ? C_HOT : isCold ? C_INACTIVE : C_ACTIVE,
                    opacity: 0.85,
                  }}
                />
              </div>
              <span className="w-14 text-right tabular-nums">{(u * 100).toFixed(1)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* =============================== params calculator =============================== */

function ParamsCalculator() {
  const [dModel, setDModel] = useState(4096);
  const [dFF, setDFF] = useState(14336);
  const [nExperts, setNExperts] = useState(8);
  const [topK, setTopK] = useState(2);
  const [nLayers, setNLayers] = useState(32);

  const cfg: MoEConfig = { dModel, dFF, nExperts, topK, nLayers };
  const totalPerLayer = totalMoEParams(cfg);
  const activePerLayer = activeMoEParams(cfg);
  const totalAll = totalPerLayer * nLayers;
  const activeAll = activePerLayer * nLayers;

  const ratio = activationRatio(cfg);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Active vs total parameters
        </span>
        <BlockMath math={String.raw`\#\text{params}_{\text{active}} = k \cdot 2 d_{model} d_{ff} \cdot N_l`} />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
        <SliderRow label="d_model" value={dModel} onChange={(v) => setDModel(Math.round(v))} min={512} max={8192} step={64} fmt={(v) => v.toFixed(0)} color={C_GATE} />
        <SliderRow label="d_ff" value={dFF} onChange={(v) => setDFF(Math.round(v))} min={1024} max={32768} step={256} fmt={(v) => v.toFixed(0)} color={C_GATE} />
        <SliderRow label="# experts" value={nExperts} onChange={(v) => { const n = Math.round(v); setNExperts(n); setTopK(Math.min(topK, n)); }} min={2} max={256} step={1} fmt={(v) => v.toFixed(0)} color={C_ACTIVE} />
        <SliderRow label="top-k" value={topK} onChange={(v) => setTopK(Math.min(Math.round(v), nExperts))} min={1} max={Math.min(nExperts, 16)} step={1} fmt={(v) => v.toFixed(0)} color={C_HOT} />
        <SliderRow label="layers" value={nLayers} onChange={(v) => setNLayers(Math.round(v))} min={1} max={120} step={1} fmt={(v) => v.toFixed(0)} color={C_OK} />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 font-mono text-[12px]">
        <Stat label="active / token" value={formatParams(activeAll)} color={C_ACTIVE} hint={`${(ratio * 100).toFixed(1)}% of total`} />
        <Stat label="total (all experts)" value={formatParams(totalAll)} color={C_GATE} />
        <Stat label="dense-equivalent" value={`~${formatParams(activeAll)}`} color={C_OK} hint="compute cost" />
        <Stat label="storage cost" value={formatParams(totalAll)} color={C_BAD} hint="HBM / disk" />
      </div>

      <ParamsBar active={activeAll} total={totalAll} />

      <p className="text-[11px] text-ink-400 leading-snug">
        Mixtral 8×7B is "47B params total, 13B active". The active count
        determines your decoding FLOPs (and roughly latency on a single GPU);
        the total determines how much HBM you need to hold the model. MoE wins
        when memory is plentiful but compute is the bottleneck — which is the
        case for inference-heavy production serving.
      </p>
    </section>
  );
}

function ParamsBar({ active, total }: { active: number; total: number }) {
  const pct = total > 0 ? active / total : 0;
  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2.5">
      <div className="flex items-center gap-3 text-[11px] font-mono">
        <span className="w-24 text-ink-400">active / total</span>
        <div className="flex-1 h-5 rounded bg-ink-950 border border-ink-800/60 overflow-hidden flex">
          <div className="h-full" style={{ width: `${pct * 100}%`, background: C_ACTIVE, opacity: 0.85 }} />
          <div className="h-full" style={{ width: `${(1 - pct) * 100}%`, background: C_INACTIVE, opacity: 0.75 }} />
        </div>
        <span className="w-24 text-right tabular-nums">{(pct * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}

/* =============================== model table =============================== */

function ModelTable() {
  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Real MoE models
        </span>
      </div>
      <div className="overflow-x-auto -mx-2 px-2">
        <table className="w-full text-[12px] font-mono">
          <thead>
            <tr className="text-left text-ink-500 border-b border-ink-800/60">
              <th className="py-1.5 pr-3">Model</th>
              <th className="py-1.5 pr-3 text-right">total</th>
              <th className="py-1.5 pr-3 text-right">active</th>
              <th className="py-1.5 pr-3 text-right">N</th>
              <th className="py-1.5 pr-3 text-right">top-k</th>
              <th className="py-1.5 pr-3">notes</th>
            </tr>
          </thead>
          <tbody>
            {MOE_PRESETS.map((m) => (
              <tr key={m.name} className="border-b border-ink-900/60 hover:bg-ink-900/40">
                <td className="py-1.5 pr-3 text-ink-100">{m.name}</td>
                <td className="py-1.5 pr-3 text-right tabular-nums text-ink-200">{m.totalParams}</td>
                <td className="py-1.5 pr-3 text-right tabular-nums" style={{ color: C_ACTIVE }}>{m.activeParams}</td>
                <td className="py-1.5 pr-3 text-right tabular-nums">{m.nExperts}</td>
                <td className="py-1.5 pr-3 text-right tabular-nums">{m.topK}</td>
                <td className="py-1.5 pr-3 text-ink-400 text-[11px]">{m.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="Router parameters">
        <InlineMath math={String.raw`W_g \in \mathbb{R}^{N \times d_{model}}`} />
      </PropCard>
      <PropCard title="Active params / token">
        <InlineMath math={String.raw`k \cdot 2\,d_{model}\,d_{ff}`} />
      </PropCard>
      <PropCard title="Total params / layer">
        <InlineMath math={String.raw`N \cdot 2\,d_{model}\,d_{ff}`} />
      </PropCard>
      <PropCard title="Load-balance loss">
        <InlineMath math={String.raw`L_{\text{aux}} = \alpha N \sum f_e P_e`} />
      </PropCard>
      <PropCard title="Top-k entropy">
        <InlineMath math={String.raw`H(\tilde{g}) \le \log k`} />
      </PropCard>
      <PropCard title="Capacity factor">
        <InlineMath math={String.raw`C = \frac{T\,k}{N}\cdot c\quad(c\in[1,1.5])`} />
      </PropCard>
      <PropCard title="Decode FLOPs">
        <InlineMath math={String.raw`\propto k\,(2\,d_{model}\,d_{ff})`} />
      </PropCard>
      <PropCard title="Memory cost">
        <InlineMath math={String.raw`\propto N\,(2\,d_{model}\,d_{ff})`} />
      </PropCard>
      <PropCard title="Quality scaling">
        <InlineMath math={String.raw`\text{MoE} \approx \text{dense at}\;4{-}8\!\times\!\text{params}`} />
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
      <UsageCard title="Where used" accent="text-violet-300">
        <UsageItem head="Switch Transformer" body="Google's 1.6T params at 26B active — original top-1 MoE." />
        <UsageItem head="Mixtral 8×7B / 8×22B" body="Mistral's top-2 routing, 8 experts per FFN layer." />
        <UsageItem head="DeepSeek-V3" body="256 fine-grained experts + 1 shared, auxiliary-loss-free balancing." />
        <UsageItem head="Qwen-MoE" body="60 small experts, top-4 routing — fine-grained = better quality / param." />
        <UsageItem head="Gemini Pro / 1.5" body="reportedly MoE; sparse activation rumored." />
        <UsageItem head="Arctic 480B" body="dense+MoE hybrid by Snowflake (128 experts, top-2)." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-cyan-300">
        <UsageItem head="Compute-decoupled scaling" body="grow params without growing per-token FLOPs." />
        <UsageItem head="Better quality / FLOP" body="MoE matches dense at 4-8× more params, ~same compute." />
        <UsageItem head="Specialization" body="experts learn distinct token / topic distributions." />
        <UsageItem head="Inference-friendly with cache" body={<>only k expert FFNs need to load per token (k·d² gemms)</>} />
        <UsageItem head="Fine-grained outperforms coarse" body="many small experts > few large experts at iso-FLOPs." />
        <UsageItem head="Memory is cheaper than compute" body="HBM keeps growing; FLOPs/s tops out per GPU." />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem head="Load imbalance" body="without aux loss, gradient concentrates on 1-2 experts." />
        <UsageItem head="Distributed training complexity" body="all-to-all communication of tokens-to-experts is expensive." />
        <UsageItem head="Memory cost stays large" body="must keep all N experts in HBM (or page across nodes)." />
        <UsageItem head="Inference latency variance" body="batched inference may have token-expert imbalance per batch." />
        <UsageItem head="Routing is non-differentiable" body="top-k uses straight-through; small gradient noise." />
        <UsageItem head="Fine-tuning is harder" body="adapter / instruction-tuning on MoE is less mature than dense." />
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

/* =============================== shared =============================== */

function Stat({
  label,
  value,
  color,
  hint,
}: {
  label: string;
  value: string;
  color?: string;
  hint?: string;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-wider text-ink-500">{label}</span>
      <span className="text-base font-semibold tabular-nums" style={color ? { color } : undefined}>
        {value}
      </span>
      {hint && <span className="text-[10px] text-ink-500">{hint}</span>}
    </div>
  );
}

function SliderRow({
  label,
  value,
  onChange,
  min,
  max,
  step,
  color,
  fmt,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  color?: string;
  fmt?: (v: number) => string;
}) {
  return (
    <div className="flex items-center gap-2 font-mono text-[12px]">
      <span className="w-28 shrink-0 text-ink-400" style={color ? { color } : undefined}>
        {label}
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1"
        style={color ? { accentColor: color } : undefined}
      />
      <span className="w-16 text-right tabular-nums">
        {fmt ? fmt(value) : value.toFixed(2)}
      </span>
    </div>
  );
}

function MathBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-3 flex items-center justify-center">
      {children}
    </div>
  );
}

function formatParams(n: number): string {
  const units = ["", "K", "M", "B", "T"];
  let v = n;
  let u = 0;
  while (v >= 1000 && u < units.length - 1) {
    v /= 1000;
    u += 1;
  }
  return `${v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)}${units[u]}`;
}

// keep linter happy with unused constant
void C_AXIS;
