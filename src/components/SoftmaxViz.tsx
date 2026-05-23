import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine,
} from "recharts";
import { motion } from "framer-motion";
import { Dice5, Minus, Plus, RotateCcw, Snowflake, Flame } from "lucide-react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import Slider from "./Slider";
import { maxEntropy, softmax } from "../lib/softmax";
import SoftmaxAdvancedLab from "./SoftmaxAdvancedLab";

const PALETTE = [
  "#7c5cff",
  "#22d3ee",
  "#f472b6",
  "#fbbf24",
  "#34d399",
  "#f87171",
  "#a78bfa",
  "#60a5fa",
  "#fb7185",
  "#facc15",
];

const PRESETS: { label: string; logits: number[]; desc: string }[] = [
  {
    label: "Uniform",
    logits: [0, 0, 0, 0],
    desc: "All logits equal → maximum entropy.",
  },
  {
    label: "Confident",
    logits: [4, 1, 0.5, -0.5],
    desc: "One logit dominates → sharp peak.",
  },
  {
    label: "Two-way tie",
    logits: [3, 3, 0, 0],
    desc: "Two close winners → bimodal output.",
  },
  {
    label: "Spread",
    logits: [2.5, 1.5, 0.5, -0.5, -1.5],
    desc: "Gradual gradient across classes.",
  },
];

const fmt = (v: number, d = 3) =>
  Number.isFinite(v) ? v.toFixed(d) : "—";

export default function SoftmaxViz() {
  const [logits, setLogits] = useState<number[]>([2.0, 1.0, 0.1, -0.5]);
  const [temperature, setTemperature] = useState(1);
  const [showStable, setShowStable] = useState(true);

  const result = useMemo(
    () => softmax(logits, temperature),
    [logits, temperature]
  );

  const chartData = useMemo(
    () =>
      result.steps.map((s, i) => ({
        name: `z${subscript(i + 1)}`,
        index: i,
        logit: s.logit,
        prob: s.prob,
        probPct: s.prob * 100,
      })),
    [result]
  );

  const maxAbsLogit = Math.max(
    4,
    ...logits.map((v) => Math.abs(v))
  );
  const logitDomain: [number, number] = [
    -Math.ceil(maxAbsLogit),
    Math.ceil(maxAbsLogit),
  ];

  const updateLogit = (i: number, v: number) =>
    setLogits((prev) => prev.map((x, idx) => (idx === i ? v : x)));

  const addClass = () => {
    if (logits.length >= 10) return;
    setLogits((prev) => [...prev, 0]);
  };

  const removeClass = () => {
    if (logits.length <= 2) return;
    setLogits((prev) => prev.slice(0, -1));
  };

  const randomize = () => {
    setLogits((prev) =>
      prev.map(() => +(Math.random() * 6 - 3).toFixed(2))
    );
  };

  const reset = () => {
    setLogits([2.0, 1.0, 0.1, -0.5]);
    setTemperature(1);
  };

  const applyPreset = (preset: number[]) => setLogits([...preset]);

  const entropy = result.entropy;
  const maxH = maxEntropy(logits.length);
  const entropyRatio = maxH > 0 ? Math.min(1, entropy / maxH) : 0;

  return (
    <div className="flex flex-col gap-10">
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
      {/* Controls */}
      <div className="xl:col-span-4 flex flex-col gap-5">
        <section className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold tracking-wide text-ink-100">
              Logits
            </h3>
            <div className="flex items-center gap-1.5">
              <button
                className="btn !px-2 !py-1.5"
                onClick={removeClass}
                disabled={logits.length <= 2}
                aria-label="Remove class"
                title="Remove class"
              >
                <Minus className="w-4 h-4" />
              </button>
              <div className="chip !px-2.5 !py-1 text-xs font-mono">
                n = {logits.length}
              </div>
              <button
                className="btn !px-2 !py-1.5"
                onClick={addClass}
                disabled={logits.length >= 10}
                aria-label="Add class"
                title="Add class"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            {logits.map((v, i) => (
              <Slider
                key={i}
                label={`Class ${i + 1}`}
                value={v}
                min={logitDomain[0]}
                max={logitDomain[1]}
                step={0.01}
                onChange={(nv) => updateLogit(i, nv)}
                formatValue={(x) => x.toFixed(2)}
                accent={PALETTE[i % PALETTE.length]}
              />
            ))}
          </div>

          <div className="flex flex-wrap gap-2 mt-5">
            <button className="btn" onClick={randomize} title="Randomize">
              <Dice5 className="w-4 h-4" /> Randomize
            </button>
            <button className="btn" onClick={reset} title="Reset">
              <RotateCcw className="w-4 h-4" /> Reset
            </button>
          </div>
        </section>

        <section className="card p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold tracking-wide text-ink-100">
              Temperature
            </h3>
            <div className="chip font-mono">
              <Snowflake className="w-3 h-3 text-cyan-300" />
              T = {temperature.toFixed(2)}
              <Flame className="w-3 h-3 text-orange-300" />
            </div>
          </div>
          <Slider
            value={temperature}
            min={0.05}
            max={5}
            step={0.01}
            onChange={setTemperature}
            hint="Low T sharpens the distribution toward argmax. High T flattens it toward uniform."
          />
          <div className="grid grid-cols-3 gap-2 mt-4 text-[11px]">
            <button
              className="btn !py-1.5 justify-center"
              onClick={() => setTemperature(0.2)}
            >
              Sharp · 0.2
            </button>
            <button
              className="btn !py-1.5 justify-center"
              onClick={() => setTemperature(1)}
            >
              Default · 1
            </button>
            <button
              className="btn !py-1.5 justify-center"
              onClick={() => setTemperature(3)}
            >
              Soft · 3
            </button>
          </div>
        </section>

        <section className="card p-5">
          <h3 className="text-sm font-semibold tracking-wide text-ink-100 mb-3">
            Presets
          </h3>
          <div className="grid grid-cols-2 gap-2">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                className="text-left p-3 rounded-xl bg-ink-800/60 hover:bg-ink-700/70 border border-ink-700/60 transition"
                onClick={() => applyPreset(p.logits)}
                title={p.desc}
              >
                <div className="text-sm font-semibold">{p.label}</div>
                <div className="text-[11px] text-ink-400 mt-0.5 leading-snug">
                  {p.desc}
                </div>
              </button>
            ))}
          </div>
        </section>
      </div>

      {/* Visualizations */}
      <div className="xl:col-span-8 flex flex-col gap-5">
        {/* Stats */}
        <section className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard
            label="Predicted class"
            value={`#${result.argmax + 1}`}
            sub={`p = ${(result.steps[result.argmax]?.prob ?? 0).toFixed(3)}`}
            color={PALETTE[result.argmax % PALETTE.length]}
          />
          <StatCard
            label="Max probability"
            value={`${(
              (result.steps[result.argmax]?.prob ?? 0) * 100
            ).toFixed(1)}%`}
            sub="Confidence of top class"
          />
          <StatCard
            label="Entropy"
            value={`${entropy.toFixed(3)} nats`}
            sub={`max ≈ ${maxH.toFixed(3)} (n=${logits.length})`}
            progress={entropyRatio}
          />
          <StatCard
            label="Σ probabilities"
            value={result.steps
              .reduce((a, b) => a + b.prob, 0)
              .toFixed(4)}
            sub="Always exactly 1"
          />
        </section>

        {/* Charts */}
        <section className="card p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold tracking-wide text-ink-100">
              Logits → Probabilities
            </h3>
            <div className="text-[11px] text-ink-400">
              Drag sliders to see the soft transformation in real time.
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <ChartPanel title="Input logits (zᵢ)">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart
                  data={chartData}
                  margin={{ top: 10, right: 10, bottom: 5, left: -10 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.06)"
                  />
                  <XAxis
                    dataKey="name"
                    stroke="#8b94a8"
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis
                    stroke="#8b94a8"
                    tick={{ fontSize: 12 }}
                    domain={logitDomain}
                  />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.18)" />
                  <Tooltip
                    cursor={{ fill: "rgba(124,92,255,0.08)" }}
                    formatter={(v) => [Number(v).toFixed(3), "logit"]}
                  />
                  <Bar dataKey="logit" radius={[6, 6, 0, 0]} maxBarSize={44}>
                    {chartData.map((_, i) => (
                      <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartPanel>

            <ChartPanel title="Output probabilities (σ(z)ᵢ)">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart
                  data={chartData}
                  margin={{ top: 10, right: 10, bottom: 5, left: -10 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.06)"
                  />
                  <XAxis
                    dataKey="name"
                    stroke="#8b94a8"
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis
                    stroke="#8b94a8"
                    tick={{ fontSize: 12 }}
                    domain={[0, 1]}
                    tickFormatter={(v) => `${Math.round(v * 100)}%`}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(34,211,238,0.08)" }}
                    formatter={(v) => [
                      `${(Number(v) * 100).toFixed(2)}%`,
                      "probability",
                    ]}
                  />
                  <Bar dataKey="prob" radius={[6, 6, 0, 0]} maxBarSize={44}>
                    {chartData.map((_, i) => (
                      <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartPanel>
          </div>
        </section>

        {/* Step-by-step computation */}
        <section className="card p-5">
          <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
            <h3 className="text-sm font-semibold tracking-wide text-ink-100">
              Step-by-step computation
            </h3>
            <label className="flex items-center gap-2 text-[12px] text-ink-300 select-none cursor-pointer">
              <input
                type="checkbox"
                checked={showStable}
                onChange={(e) => setShowStable(e.target.checked)}
                className="accent-violet-500"
              />
              Show numerically stable form
            </label>
          </div>

          <div className="rounded-xl bg-ink-950/60 border border-ink-800 p-4 text-sm overflow-x-auto">
            <BlockMath
              math={
                showStable
                  ? String.raw`\sigma(z)_i = \frac{\exp\!\left(\frac{z_i - \max_k z_k}{T}\right)}{\sum_{j} \exp\!\left(\frac{z_j - \max_k z_k}{T}\right)}`
                  : String.raw`\sigma(z)_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}`
              }
            />
          </div>

          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-ink-400 text-xs uppercase tracking-wider">
                  <th className="py-2 pr-3 font-medium">i</th>
                  <th className="py-2 pr-3 font-medium">zᵢ</th>
                  <th className="py-2 pr-3 font-medium">zᵢ / T</th>
                  {showStable && (
                    <th className="py-2 pr-3 font-medium">
                      zᵢ/T − max
                    </th>
                  )}
                  <th className="py-2 pr-3 font-medium">exp(·)</th>
                  <th className="py-2 pr-3 font-medium">σ(z)ᵢ</th>
                  <th className="py-2 pr-3 font-medium">σ(z)ᵢ × 100%</th>
                </tr>
              </thead>
              <tbody>
                {result.steps.map((s, i) => {
                  const color = PALETTE[i % PALETTE.length];
                  const isMax = i === result.argmax;
                  return (
                    <tr
                      key={i}
                      className="border-t border-ink-800/80 font-mono"
                    >
                      <td className="py-2 pr-3">
                        <span
                          className="inline-flex items-center gap-2"
                          style={{ color }}
                        >
                          <span className="w-2 h-2 rounded-full" style={{ background: color }} />
                          {i + 1}
                        </span>
                      </td>
                      <td className="py-2 pr-3">{fmt(s.logit)}</td>
                      <td className="py-2 pr-3">
                        {fmt(s.logit / Math.max(temperature, 1e-6))}
                      </td>
                      {showStable && (
                        <td className="py-2 pr-3 text-ink-300">
                          {fmt(s.shifted)}
                        </td>
                      )}
                      <td className="py-2 pr-3">{fmt(s.exp, 4)}</td>
                      <td
                        className={`py-2 pr-3 ${
                          isMax ? "text-white" : "text-ink-200"
                        }`}
                      >
                        {fmt(s.prob, 4)}
                      </td>
                      <td className="py-2 pr-3">
                        <ProbBar
                          value={s.prob}
                          color={color}
                          highlight={isMax}
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
              <tfoot>
                <tr className="border-t border-ink-800 text-xs text-ink-400">
                  <td className="py-2 pr-3">Σ</td>
                  <td className="py-2 pr-3 font-mono">
                    {fmt(logits.reduce((a, b) => a + b, 0))}
                  </td>
                  <td colSpan={showStable ? 2 : 1}></td>
                  <td className="py-2 pr-3 font-mono">
                    {fmt(result.sumExp, 4)}
                  </td>
                  <td className="py-2 pr-3 font-mono text-ink-200">1.0000</td>
                  <td></td>
                </tr>
              </tfoot>
            </table>
          </div>

          <p className="text-xs text-ink-400 mt-3 leading-relaxed">
            <InlineMath math={String.raw`\sum_i \sigma(z)_i = 1`} /> always
            holds, so softmax outputs form a valid probability distribution
            over the classes.
          </p>
        </section>

        {/* Intuition */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-4" id="intuition-cards">
          <InsightCard
            title="Translation invariance"
            body={
              <>
                Adding the same constant <InlineMath math="c" /> to every logit
                doesn't change the output:
                <span className="block mt-1 font-mono text-ink-200">
                  σ(z + c) = σ(z)
                </span>
                That's why subtracting <InlineMath math={String.raw`\max_k z_k`} /> is safe — and prevents
                overflow in <InlineMath math={String.raw`\exp`} />.
              </>
            }
          />
          <InsightCard
            title="Temperature shapes confidence"
            body={
              <>
                <InlineMath math={String.raw`T \to 0`} /> drives softmax to a one-hot
                argmax. <InlineMath math={String.raw`T \to \infty`} /> drives it to a uniform
                distribution. <InlineMath math="T = 1" /> is the standard
                softmax used in cross-entropy.
              </>
            }
          />
          <InsightCard
            title="Gradient (for cross-entropy)"
            body={
              <>
                With one-hot label <InlineMath math="y" />, the gradient of
                cross-entropy w.r.t. logits is beautifully simple:
                <span className="block mt-1">
                  <InlineMath math={String.raw`\frac{\partial L}{\partial z_i} = \sigma(z)_i - y_i`} />
                </span>
                Errors flow directly back through the difference between
                predicted and true probabilities.
              </>
            }
          />
        </section>
      </div>
    </div>
    <SoftmaxAdvancedLab
      logits={logits}
      temperature={temperature}
      onSetLogits={setLogits}
      onSetTemperature={setTemperature}
    />
    </div>
  );
}

function StatCard({
  label,
  value,
  sub,
  color,
  progress,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
  progress?: number;
}) {
  return (
    <div className="card p-4">
      <div className="text-[11px] uppercase tracking-wider text-ink-400">
        {label}
      </div>
      <div
        className="text-xl font-bold mt-1"
        style={color ? { color } : undefined}
      >
        {value}
      </div>
      {sub && <div className="text-[11px] text-ink-400 mt-0.5">{sub}</div>}
      {typeof progress === "number" && (
        <div className="mt-2 h-1.5 rounded-full bg-ink-800 overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-accent to-teal"
            initial={false}
            animate={{ width: `${Math.round(progress * 100)}%` }}
            transition={{ type: "spring", stiffness: 180, damping: 24 }}
          />
        </div>
      )}
    </div>
  );
}

function ChartPanel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl bg-ink-950/40 border border-ink-800/80 p-3">
      <div className="text-xs text-ink-300 mb-2 px-1">{title}</div>
      {children}
    </div>
  );
}

function ProbBar({
  value,
  color,
  highlight,
}: {
  value: number;
  color: string;
  highlight?: boolean;
}) {
  return (
    <div className="flex items-center gap-2 min-w-[140px]">
      <div className="flex-1 h-2 rounded-full bg-ink-800 overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ background: color }}
          initial={false}
          animate={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
          transition={{ type: "spring", stiffness: 220, damping: 26 }}
        />
      </div>
      <span
        className={`tabular-nums text-xs w-12 text-right ${
          highlight ? "text-white font-semibold" : "text-ink-300"
        }`}
      >
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function InsightCard({
  title,
  body,
}: {
  title: string;
  body: React.ReactNode;
}) {
  return (
    <div className="card p-4 text-sm leading-relaxed text-ink-200">
      <div className="font-semibold text-ink-100 mb-1.5">{title}</div>
      <div className="text-ink-300">{body}</div>
    </div>
  );
}

function subscript(n: number) {
  const map: Record<string, string> = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
  };
  return String(n)
    .split("")
    .map((d) => map[d] ?? d)
    .join("");
}
