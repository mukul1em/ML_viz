import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Bar,
  BarChart,
} from "recharts";
import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Dice5,
  Grid3x3,
  Server,
  Sigma,
  Sparkles,
  Target,
  Thermometer,
  Triangle,
  WifiOff,
  Workflow,
} from "lucide-react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  API_BASE,
  api,
  useApiStatus,
  type ApiStatus,
  type StabilityResp,
} from "../lib/api";
import {
  crossEntropyGradient,
  crossEntropyLoss,
  jacobian,
  maxEntropy,
  sampleCategorical,
  softmaxProbs,
  temperatureSweep,
} from "../lib/softmax";

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

interface Props {
  logits: number[];
  temperature: number;
  onSetLogits?: (next: number[]) => void;
  onSetTemperature?: (next: number) => void;
}

export default function SoftmaxAdvancedLab({
  logits,
  temperature,
  onSetLogits,
  onSetTemperature,
}: Props) {
  const status = useApiStatus();

  return (
    <div className="flex flex-col gap-10">
      <Header status={status} />
      <TemperatureSweepSection
        logits={logits}
        temperature={temperature}
        apiOnline={status === "online"}
        onSetTemperature={onSetTemperature}
      />
      <SimplexSection
        logits={logits}
        temperature={temperature}
        onSetLogits={onSetLogits}
      />
      <JacobianSection
        logits={logits}
        temperature={temperature}
        apiOnline={status === "online"}
      />
      <CrossEntropySection
        logits={logits}
        temperature={temperature}
        apiOnline={status === "online"}
      />
      <SamplingSection logits={logits} temperature={temperature} />
      <StabilitySection apiOnline={status === "online"} />
    </div>
  );
}

/* =============================== HEADER + API STATUS =============================== */

function Header({ status }: { status: ApiStatus }) {
  return (
    <section className="flex flex-col gap-3">
      <Eyebrow icon={<Cpu className="w-3 h-3" />}>The math lab</Eyebrow>
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-2xl lg:text-3xl font-extrabold tracking-tight">
            Softmax, deeper
          </h2>
          <p className="text-ink-300 max-w-3xl leading-relaxed mt-1.5">
            Beyond the basic bars: temperature sweeps, the probability
            simplex, the Jacobian, cross-entropy gradients, sampling, and
            numerical-stability surprises. All powered by a real Python
            backend you can hit directly.
          </p>
        </div>
        <ApiStatusBadge status={status} />
      </div>
      <ApiInstructions status={status} />
    </section>
  );
}

function ApiStatusBadge({ status }: { status: ApiStatus }) {
  const cfg: Record<
    ApiStatus,
    { color: string; bg: string; icon: React.ReactNode; label: string }
  > = {
    checking: {
      color: "#fbbf24",
      bg: "#fbbf2422",
      icon: <Activity className="w-3.5 h-3.5 animate-pulse" />,
      label: "checking…",
    },
    online: {
      color: "#34d399",
      bg: "#34d39922",
      icon: <CheckCircle2 className="w-3.5 h-3.5" />,
      label: "Python API online",
    },
    offline: {
      color: "#f87171",
      bg: "#f8717122",
      icon: <WifiOff className="w-3.5 h-3.5" />,
      label: "Python API offline · using local math",
    },
  };
  const c = cfg[status];
  return (
    <div
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-mono border"
      style={{ color: c.color, background: c.bg, borderColor: `${c.color}66` }}
    >
      {c.icon}
      <span>{c.label}</span>
    </div>
  );
}

function ApiInstructions({ status }: { status: ApiStatus }) {
  if (status === "online") {
    return (
      <div className="rounded-xl border border-emerald-400/30 bg-emerald-400/[0.05] p-3 text-sm text-ink-200 leading-relaxed">
        <div className="flex items-center gap-2 mb-1">
          <Server className="w-4 h-4 text-emerald-300" />
          <span className="font-semibold">Connected to Python at {API_BASE}</span>
        </div>
        <div className="text-[12px] text-ink-300">
          Every visualization below pulls from the real numpy backend.{" "}
          <a
            href={`${API_BASE}/docs`}
            target="_blank"
            rel="noreferrer"
            className="underline text-emerald-300 hover:text-emerald-200"
          >
            Open Swagger UI →
          </a>
        </div>
      </div>
    );
  }
  return (
    <div className="rounded-xl border border-amber-400/30 bg-amber-400/[0.05] p-3 text-sm text-ink-200 leading-relaxed">
      <div className="flex items-center gap-2 mb-1.5">
        <AlertTriangle className="w-4 h-4 text-amber-300" />
        <span className="font-semibold">Start the Python backend to unlock live math</span>
      </div>
      <p className="text-[12.5px] text-ink-300 mb-2">
        Visualizations still work with built-in JavaScript math; the backend
        lets you call <code className="text-amber-200 font-mono">curl</code>{" "}
        directly to play with values.
      </p>
      <pre className="text-[11px] font-mono bg-ink-950/70 border border-ink-800 rounded-lg p-2 overflow-x-auto">{`cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000`}</pre>
    </div>
  );
}

/* ============================== TEMPERATURE SWEEP ============================== */

function TemperatureSweepSection({
  logits,
  temperature,
  apiOnline,
  onSetTemperature,
}: {
  logits: number[];
  temperature: number;
  apiOnline: boolean;
  onSetTemperature?: (t: number) => void;
}) {
  const [rows, setRows] = useState<{ T: number; probs: number[]; entropy: number; max_prob: number }[]>([]);
  const [loading, setLoading] = useState(false);
  const aborter = useRef<AbortController | null>(null);

  useEffect(() => {
    aborter.current?.abort();
    const ctrl = new AbortController();
    aborter.current = ctrl;
    setLoading(true);

    const fetchRows = async () => {
      if (apiOnline) {
        try {
          const data = await api.sweep(logits, 0.05, 5, 100, ctrl.signal);
          if (!ctrl.signal.aborted) setRows(data.rows);
        } catch {
          if (!ctrl.signal.aborted) {
            // Fallback locally
            const local = temperatureSweep(logits, 0.05, 5, 100);
            setRows(local.map((r) => ({ T: r.T, probs: r.probs, entropy: r.entropy, max_prob: r.maxProb })));
          }
        }
      } else {
        const local = temperatureSweep(logits, 0.05, 5, 100);
        setRows(local.map((r) => ({ T: r.T, probs: r.probs, entropy: r.entropy, max_prob: r.maxProb })));
      }
      if (!ctrl.signal.aborted) setLoading(false);
    };
    fetchRows();
    return () => ctrl.abort();
  }, [logits, apiOnline]);

  const chartData = useMemo(
    () =>
      rows.map((r) => {
        const o: Record<string, number> = { T: r.T, entropy: r.entropy };
        r.probs.forEach((p, i) => (o[`c${i}`] = p));
        return o;
      }),
    [rows]
  );

  const maxH = maxEntropy(logits.length);

  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Thermometer className="w-3 h-3" />}>Temperature sweep</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        How every probability bends with T
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Vary <InlineMath math="T" /> from very cold (sharp argmax) to hot
        (uniform). Each line is one class's probability, and the entropy
        curve below tells you how spread-out the distribution is overall.
        Click anywhere on the chart to jump T to that value.
      </p>

      <div className="grid grid-cols-1 xl:grid-cols-[2fr_1fr] gap-4">
        <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="text-xs text-ink-300">
              Probabilities · σ(z/T)ᵢ
            </div>
            <div className="text-[11px] text-ink-400 font-mono">
              {loading ? "…" : `${rows.length} points`}
            </div>
          </div>
          <div onClick={(e) => {
            // figure out fraction along the chart and convert to T
            const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
            const xFrac = (e.clientX - rect.left) / rect.width;
            if (xFrac >= 0 && xFrac <= 1 && onSetTemperature) {
              const T = 0.05 + (5 - 0.05) * xFrac;
              onSetTemperature(Number(T.toFixed(3)));
            }
          }} className="cursor-crosshair">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData} margin={{ top: 8, right: 12, left: -8, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="T"
                  type="number"
                  domain={[0.05, 5]}
                  ticks={[0.1, 0.5, 1, 2, 3, 4, 5]}
                  tickFormatter={(v) => `${v}`}
                  stroke="#8b94a8"
                  tick={{ fontSize: 11 }}
                  label={{ value: "T (temperature)", position: "insideBottom", offset: -2, fill: "#8b94a8", fontSize: 11 }}
                />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={(v) => `${Math.round(v * 100)}%`}
                  stroke="#8b94a8"
                  tick={{ fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{ background: "#0f1320", border: "1px solid #2a2f3f", borderRadius: 8 }}
                  labelFormatter={(v) => `T = ${Number(v).toFixed(3)}`}
                  formatter={(v, name) => {
                    const nm = String(name);
                    const num = Number(v);
                    if (nm === "entropy") return [`${num.toFixed(3)} nats`, "entropy"];
                    const idx = Number(nm.replace("c", ""));
                    return [`${(num * 100).toFixed(1)}%`, `class ${idx + 1}`];
                  }}
                />
                <ReferenceLine
                  x={temperature}
                  stroke="#fff"
                  strokeOpacity={0.6}
                  strokeDasharray="4 3"
                  label={{ value: `T = ${temperature.toFixed(2)}`, fill: "#fff", fontSize: 11, position: "top" }}
                />
                {logits.map((_, i) => (
                  <Line
                    key={i}
                    type="monotone"
                    dataKey={`c${i}`}
                    stroke={PALETTE[i % PALETTE.length]}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4">
          <div className="text-xs text-ink-300 mb-2">
            Entropy H(σ(z/T))
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData} margin={{ top: 8, right: 12, left: -8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis
                dataKey="T"
                type="number"
                domain={[0.05, 5]}
                ticks={[0.1, 0.5, 1, 2, 3, 4, 5]}
                tickFormatter={(v) => `${v}`}
                stroke="#8b94a8"
                tick={{ fontSize: 11 }}
              />
              <YAxis
                domain={[0, Math.ceil(maxH * 1.05)]}
                stroke="#8b94a8"
                tick={{ fontSize: 11 }}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <Tooltip
                contentStyle={{ background: "#0f1320", border: "1px solid #2a2f3f", borderRadius: 8 }}
                labelFormatter={(v) => `T = ${Number(v).toFixed(3)}`}
                formatter={(v) => [`${Number(v).toFixed(3)} nats`, "entropy"]}
              />
              <ReferenceLine y={maxH} stroke="#34d399" strokeOpacity={0.4} strokeDasharray="2 2" label={{ value: `max H = ln(n) ≈ ${maxH.toFixed(2)}`, fill: "#34d399", fontSize: 10, position: "insideTopRight" }} />
              <ReferenceLine
                x={temperature}
                stroke="#fff"
                strokeOpacity={0.6}
                strokeDasharray="4 3"
              />
              <Line
                type="monotone"
                dataKey="entropy"
                stroke="#fbbf24"
                strokeWidth={2.5}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-1">
        <InsightCard
          color="#22d3ee"
          title="T → 0 (cold)"
          body="The largest logit takes nearly all the mass. Softmax collapses to argmax. Entropy crashes to 0."
        />
        <InsightCard
          color="#a78bfa"
          title="T = 1 (default)"
          body="Standard cross-entropy regime. Probabilities reflect the raw confidence of the logits."
        />
        <InsightCard
          color="#fbbf24"
          title="T → ∞ (hot)"
          body="Differences between logits vanish. Distribution → uniform. Entropy → ln(n), its maximum."
        />
      </div>
    </section>
  );
}

/* ============================== PROBABILITY SIMPLEX ============================== */

function SimplexSection({
  logits,
  temperature,
  onSetLogits,
}: {
  logits: number[];
  temperature: number;
  onSetLogits?: (next: number[]) => void;
}) {
  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Triangle className="w-3 h-3" />}>The probability simplex</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        The space of all valid distributions
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Every output of softmax is a point on the <strong>(n−1)-simplex</strong>:
        all non-negative vectors that sum to 1. For{" "}
        <InlineMath math="n=3" /> that's a triangle we can <em>draw</em>.
        Drag a vertex to set a one-hot logit, or move the temperature
        slider — watch the point move.
      </p>

      {logits.length === 3 ? (
        <SimplexCanvas logits={logits} temperature={temperature} />
      ) : (
        <div className="rounded-2xl border border-ink-800/80 bg-ink-950/40 p-6 text-sm text-ink-300 flex items-center gap-3">
          <Triangle className="w-5 h-5 text-violet-300" />
          <span>
            The simplex viz needs exactly 3 classes. You currently have{" "}
            <strong>{logits.length}</strong>.
          </span>
          {onSetLogits && (
            <button
              className="btn ml-auto"
              onClick={() => onSetLogits([2, 1, 0])}
            >
              Set n = 3
            </button>
          )}
        </div>
      )}
    </section>
  );
}

function SimplexCanvas({
  logits,
  temperature,
}: {
  logits: number[];
  temperature: number;
}) {
  const probs = useMemo(() => softmaxProbs(logits, temperature), [logits, temperature]);
  // Compute a temperature-sweep trail for the same logits
  const trail = useMemo(
    () => temperatureSweep(logits, 0.05, 5, 60).map((r) => r.probs),
    [logits]
  );

  const size = 360;
  const cx = size / 2;
  const cy = size * 0.55;
  const r = size * 0.4;
  const SQ32 = Math.sqrt(3) / 2;

  // Vertex mapping: v1 (top) = class 1, v2 (bottom-right) = class 2, v3 (bottom-left) = class 3
  const project = (p: number[]) => ({
    x: cx + r * SQ32 * (p[1] - p[2]),
    y: cy + r * (0.5 - 1.5 * p[0]),
  });

  const pt = project(probs);
  const v1 = { x: cx, y: cy - r };
  const v2 = { x: cx + r * SQ32, y: cy + r * 0.5 };
  const v3 = { x: cx - r * SQ32, y: cy + r * 0.5 };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[auto_1fr] gap-6">
      <div className="rounded-2xl bg-ink-950/70 border border-ink-800/80 p-4 flex justify-center">
        <svg viewBox={`0 0 ${size} ${size * 0.92}`} width={size} height={size * 0.92}>
          {/* Argmax regions (3 quadrants meeting at centroid) */}
          <ArgmaxRegions v1={v1} v2={v2} v3={v3} cx={cx} cy={cy + r * 0.5 / 3} />

          {/* Iso-prob lines (lines of constant p_i) */}
          {[0.25, 0.5, 0.75].map((lvl) => (
            <g key={lvl}>
              <line
                x1={v2.x + (v1.x - v2.x) * lvl}
                y1={v2.y + (v1.y - v2.y) * lvl}
                x2={v3.x + (v1.x - v3.x) * lvl}
                y2={v3.y + (v1.y - v3.y) * lvl}
                stroke="rgba(124,92,255,0.18)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
            </g>
          ))}

          {/* Triangle outline */}
          <polygon
            points={`${v1.x},${v1.y} ${v2.x},${v2.y} ${v3.x},${v3.y}`}
            fill="rgba(255,255,255,0.02)"
            stroke="rgba(255,255,255,0.3)"
            strokeWidth={1.5}
          />

          {/* Temperature trail */}
          <polyline
            points={trail
              .map((p) => {
                const q = project(p);
                return `${q.x},${q.y}`;
              })
              .join(" ")}
            fill="none"
            stroke="#fbbf24"
            strokeOpacity={0.4}
            strokeWidth={1.5}
            strokeDasharray="3 2"
          />

          {/* Vertex labels */}
          <VertexLabel
            v={v1}
            color={PALETTE[0]}
            label="(1, 0, 0)"
            classLabel="class 1"
            anchor="middle"
            dy={-12}
          />
          <VertexLabel
            v={v2}
            color={PALETTE[1]}
            label="(0, 1, 0)"
            classLabel="class 2"
            anchor="start"
            dx={10}
            dy={6}
          />
          <VertexLabel
            v={v3}
            color={PALETTE[2]}
            label="(0, 0, 1)"
            classLabel="class 3"
            anchor="end"
            dx={-10}
            dy={6}
          />

          {/* Centroid (uniform) marker */}
          <circle cx={cx} cy={cy + (r * 0.5) / 3} r={3} fill="rgba(255,255,255,0.3)" />
          <text x={cx + 8} y={cy + (r * 0.5) / 3 + 4} fontSize={9} fill="#8b94a8">
            uniform (1/3,1/3,1/3)
          </text>

          {/* Current point */}
          <motion.circle
            cx={pt.x}
            cy={pt.y}
            r={10}
            fill="rgba(255,255,255,0.15)"
            initial={false}
            animate={{ cx: pt.x, cy: pt.y }}
            transition={{ type: "spring", stiffness: 180, damping: 22 }}
          />
          <motion.circle
            cx={pt.x}
            cy={pt.y}
            r={6}
            fill="#fff"
            stroke="#7c5cff"
            strokeWidth={2}
            initial={false}
            animate={{ cx: pt.x, cy: pt.y }}
            transition={{ type: "spring", stiffness: 180, damping: 22 }}
          />
        </svg>
      </div>

      <div className="flex flex-col gap-3">
        <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4">
          <div className="text-[11px] uppercase tracking-wider text-ink-400 font-semibold mb-2">
            Where is the point?
          </div>
          <div className="text-2xl font-mono">
            ({probs[0].toFixed(3)},{" "}
            <span style={{ color: PALETTE[1] }}>{probs[1].toFixed(3)}</span>,{" "}
            <span style={{ color: PALETTE[2] }}>{probs[2].toFixed(3)}</span>)
          </div>
          <div className="text-[11px] text-ink-400 mt-2 leading-relaxed">
            All three coordinates sum to{" "}
            <span className="text-ink-200">
              {(probs[0] + probs[1] + probs[2]).toFixed(4)}
            </span>{" "}
            — the simplex constraint, always satisfied by softmax.
          </div>
        </div>

        <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4">
          <div className="text-[11px] uppercase tracking-wider text-ink-400 font-semibold mb-2 flex items-center gap-2">
            <span className="inline-block w-3 h-1.5 rounded-full bg-amber-300" />
            The temperature trail
          </div>
          <p className="text-sm text-ink-300 leading-relaxed">
            As <InlineMath math="T" /> varies from 0.05 → 5, the softmax
            output traces a curve from the nearest <em>corner</em> (T → 0,
            one-hot) to the <em>centroid</em> (T → ∞, uniform).
          </p>
        </div>

        <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4">
          <div className="text-[11px] uppercase tracking-wider text-ink-400 font-semibold mb-2">
            Argmax regions
          </div>
          <p className="text-sm text-ink-300 leading-relaxed">
            The three colored zones show which class wins for points in
            that region. The boundaries are where two classes tie.
          </p>
        </div>
      </div>
    </div>
  );
}

function ArgmaxRegions({
  v1,
  v2,
  v3,
  cx,
  cy,
}: {
  v1: { x: number; y: number };
  v2: { x: number; y: number };
  v3: { x: number; y: number };
  cx: number;
  cy: number;
}) {
  // Midpoints of opposite edges
  const m12 = { x: (v1.x + v2.x) / 2, y: (v1.y + v2.y) / 2 };
  const m13 = { x: (v1.x + v3.x) / 2, y: (v1.y + v3.y) / 2 };
  const m23 = { x: (v2.x + v3.x) / 2, y: (v2.y + v3.y) / 2 };
  const c = { x: cx, y: cy };
  return (
    <g>
      <polygon
        points={`${v1.x},${v1.y} ${m12.x},${m12.y} ${c.x},${c.y} ${m13.x},${m13.y}`}
        fill={`${PALETTE[0]}1a`}
      />
      <polygon
        points={`${v2.x},${v2.y} ${m23.x},${m23.y} ${c.x},${c.y} ${m12.x},${m12.y}`}
        fill={`${PALETTE[1]}1a`}
      />
      <polygon
        points={`${v3.x},${v3.y} ${m13.x},${m13.y} ${c.x},${c.y} ${m23.x},${m23.y}`}
        fill={`${PALETTE[2]}1a`}
      />
    </g>
  );
}

function VertexLabel({
  v,
  color,
  label,
  classLabel,
  anchor,
  dx = 0,
  dy = 0,
}: {
  v: { x: number; y: number };
  color: string;
  label: string;
  classLabel: string;
  anchor: "start" | "middle" | "end";
  dx?: number;
  dy?: number;
}) {
  return (
    <g>
      <circle cx={v.x} cy={v.y} r={7} fill={color} stroke="#fff" strokeWidth={1.5} />
      <text
        x={v.x + dx}
        y={v.y + dy}
        textAnchor={anchor}
        fontSize={11}
        fontWeight={700}
        fill={color}
        fontFamily="JetBrains Mono, monospace"
      >
        {label}
      </text>
      <text
        x={v.x + dx}
        y={v.y + dy + 11}
        textAnchor={anchor}
        fontSize={9}
        fill="#8b94a8"
      >
        {classLabel}
      </text>
    </g>
  );
}

/* ============================== JACOBIAN ============================== */

function JacobianSection({
  logits,
  temperature,
  apiOnline,
}: {
  logits: number[];
  temperature: number;
  apiOnline: boolean;
}) {
  const [J, setJ] = useState<number[][]>(() => {
    const p = softmaxProbs(logits, temperature);
    const J0 = jacobian(p);
    return J0.map((row) => row.map((v) => v / Math.max(temperature, 1e-6)));
  });
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (apiOnline) {
      api
        .jacobian(logits, temperature)
        .then((d) => {
          if (!cancelled) setJ(d.jacobian);
        })
        .catch(() => {
          if (!cancelled) {
            const p = softmaxProbs(logits, temperature);
            const J0 = jacobian(p);
            setJ(J0.map((row) => row.map((v) => v / Math.max(temperature, 1e-6))));
          }
        });
    } else {
      const p = softmaxProbs(logits, temperature);
      const J0 = jacobian(p);
      setJ(J0.map((row) => row.map((v) => v / Math.max(temperature, 1e-6))));
    }
    return () => {
      cancelled = true;
    };
  }, [logits, temperature, apiOnline]);

  const n = J.length;
  const cell = Math.max(40, Math.min(70, 480 / Math.max(n, 1)));
  const labelW = 36;
  const labelH = 36;
  const w = labelW + n * cell;
  const h = labelH + n * cell;
  const absMax = Math.max(...J.flat().map((v) => Math.abs(v)), 0.001);

  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Grid3x3 className="w-3 h-3" />}>The Jacobian</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        How a logit twitch ripples through every probability
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Softmax couples its outputs: nudge one logit and{" "}
        <em>every</em> probability changes. The Jacobian{" "}
        <InlineMath math={String.raw`J_{ij} = \partial \sigma(z)_i / \partial z_j`} />{" "}
        is what backprop actually multiplies by. There's a beautifully
        simple closed form.
      </p>
      <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4">
        <BlockMath
          math={String.raw`J_{ij} \;=\; \frac{1}{T}\,\sigma(z)_i \,\bigl(\delta_{ij} - \sigma(z)_j\bigr)`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-4">
        <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 overflow-auto">
          <svg width={w} height={h} className="block" style={{ maxWidth: "100%" }}>
            {/* Column labels (j) */}
            {J.map((_, j) => (
              <text
                key={`cl-${j}`}
                x={labelW + j * cell + cell / 2}
                y={labelH - 10}
                textAnchor="middle"
                fontSize={11}
                fontWeight={600}
                fill="#a3a8b8"
                fontFamily="JetBrains Mono, monospace"
              >
                z{subscript(j + 1)}
              </text>
            ))}
            {/* Row labels (i) */}
            {J.map((_, i) => (
              <text
                key={`rl-${i}`}
                x={labelW - 8}
                y={labelH + i * cell + cell / 2 + 4}
                textAnchor="end"
                fontSize={11}
                fontWeight={600}
                fill="#a3a8b8"
                fontFamily="JetBrains Mono, monospace"
              >
                p{subscript(i + 1)}
              </text>
            ))}
            {/* Cells */}
            {J.map((row, i) =>
              row.map((v, j) => {
                const t = Math.abs(v) / absMax;
                const isPos = v >= 0;
                const bg = isPos
                  ? `rgba(52, 211, 153, ${0.15 + t * 0.7})`
                  : `rgba(248, 113, 113, ${0.15 + t * 0.7})`;
                const isHov = hover?.i === i && hover?.j === j;
                return (
                  <g key={`${i}-${j}`}>
                    <rect
                      x={labelW + j * cell + 2}
                      y={labelH + i * cell + 2}
                      width={cell - 4}
                      height={cell - 4}
                      rx={6}
                      fill={bg}
                      stroke={isHov ? "#fff" : "rgba(255,255,255,0.06)"}
                      strokeWidth={isHov ? 1.5 : 1}
                      onMouseEnter={() => setHover({ i, j })}
                      onMouseLeave={() => setHover(null)}
                    />
                    <text
                      x={labelW + j * cell + cell / 2}
                      y={labelH + i * cell + cell / 2 + 4}
                      textAnchor="middle"
                      fontSize={Math.min(12, cell / 4.2)}
                      fontFamily="JetBrains Mono, monospace"
                      fill={t > 0.5 ? "#0a0b10" : "#e6e8ee"}
                      pointerEvents="none"
                    >
                      {v.toFixed(3)}
                    </text>
                  </g>
                );
              })
            )}
          </svg>
        </div>

        <div className="flex flex-col gap-3">
          <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4">
            <div className="text-[11px] uppercase tracking-wider text-ink-400 font-semibold mb-2">
              Hover
            </div>
            {hover ? (
              <div className="space-y-1 text-sm">
                <div className="font-mono">
                  J[{hover.i + 1},{hover.j + 1}] ={" "}
                  <span className={J[hover.i][hover.j] >= 0 ? "text-emerald-300" : "text-red-300"}>
                    {J[hover.i][hover.j].toFixed(4)}
                  </span>
                </div>
                <div className="text-[11px] text-ink-400">
                  ∂σ(z)<sub>{hover.i + 1}</sub> / ∂z<sub>{hover.j + 1}</sub>
                </div>
                {hover.i === hover.j ? (
                  <div className="text-[11px] text-emerald-300">
                    Diagonal: p<sub>i</sub>(1−p<sub>i</sub>) — always{" "}
                    <span className="font-mono">≥ 0</span>.
                  </div>
                ) : (
                  <div className="text-[11px] text-red-300">
                    Off-diagonal: −p<sub>i</sub>p<sub>j</sub> — always{" "}
                    <span className="font-mono">≤ 0</span>.
                  </div>
                )}
              </div>
            ) : (
              <div className="text-[11px] text-ink-400">Hover a cell to inspect.</div>
            )}
          </div>

          <InsightCard
            color="#34d399"
            title="Diagonal entries"
            body={
              <>
                When you push the logit of class <InlineMath math="i" /> up,
                its own probability goes up by{" "}
                <InlineMath math={String.raw`p_i(1-p_i)`} /> per unit. Always
                non-negative.
              </>
            }
          />
          <InsightCard
            color="#f87171"
            title="Off-diagonal entries"
            body={
              <>
                And when class <InlineMath math="i" /> wins more, every other
                class <InlineMath math="j" /> loses{" "}
                <InlineMath math={String.raw`p_i p_j`} /> of mass.
                Conservation of probability, baked in.
              </>
            }
          />
        </div>
      </div>
    </section>
  );
}

/* ============================== CROSS ENTROPY ============================== */

function CrossEntropySection({
  logits,
  temperature,
  apiOnline,
}: {
  logits: number[];
  temperature: number;
  apiOnline: boolean;
}) {
  const [target, setTarget] = useState(0);
  const [loss, setLoss] = useState(0);
  const [grad, setGrad] = useState<number[]>([]);
  const [probs, setProbs] = useState<number[]>(() => softmaxProbs(logits, temperature));

  useEffect(() => {
    setTarget((t) => Math.min(t, logits.length - 1));
  }, [logits.length]);

  useEffect(() => {
    let cancelled = false;
    const computeLocal = () => {
      const p = softmaxProbs(logits, temperature);
      const tgt = Math.min(target, p.length - 1);
      setProbs(p);
      setLoss(crossEntropyLoss(p, tgt));
      setGrad(crossEntropyGradient(p, tgt));
    };
    if (apiOnline) {
      api
        .crossEntropy(logits, Math.min(target, logits.length - 1), temperature)
        .then((d) => {
          if (cancelled) return;
          setProbs(d.probabilities);
          setLoss(d.loss);
          setGrad(d.gradient);
        })
        .catch(computeLocal);
    } else {
      computeLocal();
    }
    return () => {
      cancelled = true;
    };
  }, [logits, temperature, target, apiOnline]);

  const data = grad.map((g, i) => ({
    name: `c${i + 1}`,
    grad: g,
    p: probs[i],
    isTarget: i === target,
  }));

  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Target className="w-3 h-3" />}>Cross-entropy</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        Why softmax + CE has such a simple gradient
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Pair softmax with the categorical cross-entropy loss and the
        gradient with respect to the logits collapses to a single beautiful
        expression — no derivative of softmax in sight. This is{" "}
        <em>the</em> reason these two are always used together.
      </p>

      <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4">
        <BlockMath
          math={String.raw`L = -\sum_i y_i \log \sigma(z)_i \quad\implies\quad \frac{\partial L}{\partial z_i} = \sigma(z)_i - y_i`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-4">
        <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4">
          <div className="text-[11px] uppercase tracking-wider text-ink-400 font-semibold mb-3">
            Pick the true class
          </div>
          <div className="flex flex-wrap gap-2">
            {logits.map((_, i) => (
              <button
                key={i}
                onClick={() => setTarget(i)}
                className={[
                  "px-3 py-1.5 rounded-md text-sm font-mono border transition-all",
                  i === target
                    ? "text-white"
                    : "text-ink-300 border-ink-700 hover:border-violet-400/50",
                ].join(" ")}
                style={{
                  background: i === target ? `${PALETTE[i % PALETTE.length]}33` : "transparent",
                  borderColor: i === target ? PALETTE[i % PALETTE.length] : undefined,
                }}
              >
                y = class {i + 1}
              </button>
            ))}
          </div>

          <div className="mt-5 grid grid-cols-2 gap-3">
            <Metric
              label="Loss L"
              value={Number.isFinite(loss) ? loss.toFixed(4) : "∞"}
              color="#fbbf24"
              hint="nats. Lower is better."
            />
            <Metric
              label="p_target"
              value={(probs[target] ?? 0).toFixed(4)}
              color={PALETTE[target % PALETTE.length]}
              hint={`L = −ln(p_target)`}
            />
          </div>
        </div>

        <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4">
          <div className="text-[11px] uppercase tracking-wider text-ink-400 font-semibold mb-2">
            Gradient ∂L/∂z = σ(z) − y
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data} margin={{ top: 8, right: 12, left: -10, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="name" stroke="#8b94a8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#8b94a8" tick={{ fontSize: 11 }} domain={[-1, 1]} />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" />
              <Tooltip
                contentStyle={{ background: "#0f1320", border: "1px solid #2a2f3f", borderRadius: 8 }}
                formatter={(v) => [Number(v).toFixed(4), "gradient"]}
              />
              <Bar dataKey="grad" radius={[6, 6, 0, 0]} maxBarSize={44}>
                {data.map((d, i) => (
                  <Cell
                    key={i}
                    fill={d.grad >= 0 ? "#f87171" : "#34d399"}
                    stroke={d.isTarget ? "#fff" : "transparent"}
                    strokeWidth={d.isTarget ? 2 : 0}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="text-[11px] text-ink-400 mt-1 leading-relaxed">
            Positive bars (red): probability that the model wants to push <em>down</em>.
            Negative bar (green): the target — the model wants to push it <em>up</em>.
            At convergence on a confident correct prediction, every bar approaches zero.
          </div>
        </div>
      </div>
    </section>
  );
}

/* ============================== SAMPLING ============================== */

function SamplingSection({
  logits,
  temperature,
}: {
  logits: number[];
  temperature: number;
}) {
  const [n, setN] = useState(1000);
  const [seed, setSeed] = useState(42);
  const probs = useMemo(() => softmaxProbs(logits, temperature), [logits, temperature]);
  const { empirical } = useMemo(
    () => sampleCategorical(probs, n, seed),
    [probs, n, seed]
  );

  const data = probs.map((p, i) => ({
    name: `c${i + 1}`,
    theory: p,
    empirical: empirical[i] ?? 0,
  }));

  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Dice5 className="w-3 h-3" />}>Sampling</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        Watch the law of large numbers settle in
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Sample <InlineMath math="N" /> indices from the categorical
        distribution defined by softmax. With <InlineMath math="N" /> small
        the histogram is jagged; as <InlineMath math="N" /> grows it
        converges to the true probabilities.
      </p>

      <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4">
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_auto] gap-4 items-center mb-3">
          <div>
            <label className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold">
              N samples: <span className="text-ink-100 font-mono">{n.toLocaleString()}</span>
            </label>
            <input
              type="range"
              min={10}
              max={50000}
              step={10}
              value={n}
              onChange={(e) => setN(Number(e.target.value))}
              className="w-full mt-1 accent-violet-500"
            />
          </div>
          <div>
            <label className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold block mb-1">
              Seed
            </label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-24 px-2 py-1.5 rounded-lg bg-ink-950/70 border border-ink-700 font-mono text-sm focus:outline-none focus:border-violet-400/60"
            />
          </div>
          <button className="btn" onClick={() => setSeed((s) => s + 1)}>
            <Sparkles className="w-3.5 h-3.5" /> Re-roll
          </button>
        </div>

        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data} margin={{ top: 8, right: 12, left: -10, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="name" stroke="#8b94a8" tick={{ fontSize: 11 }} />
            <YAxis
              stroke="#8b94a8"
              tick={{ fontSize: 11 }}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
              domain={[0, 1]}
            />
            <Tooltip
              contentStyle={{ background: "#0f1320", border: "1px solid #2a2f3f", borderRadius: 8 }}
              formatter={(v, key) => [
                `${(Number(v) * 100).toFixed(2)}%`,
                String(key) === "theory" ? "theoretical" : "empirical",
              ]}
            />
            <Bar dataKey="theory" fill="#7c5cff" name="theory" radius={[4, 4, 0, 0]} maxBarSize={28} />
            <Bar dataKey="empirical" fill="#fbbf24" name="empirical" radius={[4, 4, 0, 0]} maxBarSize={28} />
          </BarChart>
        </ResponsiveContainer>

        <div className="flex items-center gap-4 mt-2 text-[11px]">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-violet-500" /> theoretical p
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-amber-400" /> empirical (out of {n.toLocaleString()})
          </span>
        </div>
      </div>
    </section>
  );
}

/* ============================== STABILITY ============================== */

function StabilitySection({ apiOnline }: { apiOnline: boolean }) {
  const [extremes, setExtremes] = useState<number[]>([1000, 999, 998]);
  const [resp, setResp] = useState<StabilityResp | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (apiOnline) {
      api
        .stability(extremes, 1.0)
        .then((d) => !cancelled && setResp(d))
        .catch(() => {
          if (cancelled) return;
          setResp(simulateStabilityLocal(extremes));
        });
    } else {
      setResp(simulateStabilityLocal(extremes));
    }
    return () => {
      cancelled = true;
    };
  }, [extremes, apiOnline]);

  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Sigma className="w-3 h-3" />}>Numerical stability</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        Why every framework subtracts the max
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Try pushing the logits to extremes. The naive formula starts
        producing <code className="text-amber-300 font-mono">Infinity</code>{" "}
        and <code className="text-red-300 font-mono">NaN</code>; the
        max-shifted version stays rock-solid. Same answer mathematically,
        wildly different in float64.
      </p>

      <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4">
        <div className="flex items-center gap-2 flex-wrap mb-4">
          <span className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold">
            Try presets:
          </span>
          {[
            { label: "Tiny", v: [2, 1, 0] },
            { label: "Big (~1000)", v: [1000, 999, 998] },
            { label: "Huge (10⁴)", v: [10_000, 9_995, 9_990] },
            { label: "Insane (10⁶)", v: [1_000_000, 999_999, 999_998] },
          ].map((p) => (
            <button key={p.label} className="btn" onClick={() => setExtremes(p.v)}>
              {p.label}
            </button>
          ))}
          <div className="text-[11px] font-mono text-ink-400 ml-2">
            current: [{extremes.join(", ")}]
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <StabilityCard
            ok={!!resp?.naive.ok}
            title="Naive  e^(zᵢ) / Σ e^(zⱼ)"
            color="#f87171"
            entries={
              resp
                ? extremes.map((z, i) => ({
                    z,
                    exp: resp.naive.exp_vals[i],
                    prob: resp.naive.probs[i],
                  }))
                : []
            }
            sumExp={resp?.naive.sum_exp ?? null}
            maxExp={resp?.naive.max_exp ?? null}
          />
          <StabilityCard
            ok={!!resp?.stable.ok}
            title="Stable  e^(zᵢ − max z) / Σ e^(zⱼ − max z)"
            color="#34d399"
            entries={
              resp
                ? extremes.map((z, i) => ({
                    z,
                    exp: resp.stable.exp_vals[i],
                    prob: resp.stable.probs[i],
                  }))
                : []
            }
            sumExp={resp?.stable.sum_exp ?? null}
            maxExp={resp?.stable.max_exp ?? null}
          />
        </div>
      </div>

      <div className="rounded-xl border border-violet-400/30 bg-violet-400/[0.05] p-4 text-sm leading-relaxed text-ink-200">
        <span className="text-violet-300 font-semibold">The trick →</span>{" "}
        Because softmax is invariant to a shift,{" "}
        <InlineMath math={String.raw`\sigma(z) = \sigma(z - c)`} /> for any{" "}
        <InlineMath math="c" />. Pick{" "}
        <InlineMath math={String.raw`c = \max_k z_k`} /> and the largest
        exponent becomes <InlineMath math="e^0 = 1" />. Overflow gone.
      </div>
    </section>
  );
}

function simulateStabilityLocal(logits: number[]): StabilityResp {
  const maxZ = Math.max(...logits);
  const eShift = logits.map((z) => Math.exp(z - maxZ));
  const sumShift = eShift.reduce((a, b) => a + b, 0);
  const probsStable = eShift.map((e) => e / sumShift);

  const eNaive = logits.map((z) => Math.exp(z));
  const sumNaive = eNaive.reduce((a, b) => a + b, 0);
  const ok = eNaive.every(Number.isFinite) && Number.isFinite(sumNaive);

  return {
    naive: {
      ok,
      exp_vals: eNaive.map((v) => (Number.isFinite(v) ? v : null)),
      sum_exp: Number.isFinite(sumNaive) ? sumNaive : null,
      probs: ok ? eNaive.map((e) => e / sumNaive) : eNaive.map(() => null),
      max_exp: ok ? Math.max(...eNaive) : null,
    },
    stable: {
      ok: true,
      exp_vals: eShift,
      sum_exp: sumShift,
      probs: probsStable,
      max_exp: Math.max(...eShift),
      shifted_logits: logits.map((z) => z - maxZ),
    },
  };
}

function StabilityCard({
  ok,
  title,
  color,
  entries,
  sumExp,
  maxExp,
}: {
  ok: boolean;
  title: string;
  color: string;
  entries: { z: number; exp: number | null; prob: number | null }[];
  sumExp: number | null;
  maxExp: number | null;
}) {
  return (
    <div
      className="rounded-xl border p-4 flex flex-col gap-3"
      style={{
        background: ok ? `${color}0a` : "#f8717111",
        borderColor: ok ? `${color}55` : "#f8717188",
      }}
    >
      <div className="flex items-center gap-2">
        <div
          className="w-6 h-6 rounded-md flex items-center justify-center"
          style={{ background: ok ? `${color}33` : "#f8717133" }}
        >
          {ok ? (
            <CheckCircle2 className="w-4 h-4" style={{ color }} />
          ) : (
            <AlertTriangle className="w-4 h-4 text-red-300" />
          )}
        </div>
        <div className="font-mono text-sm" style={{ color: ok ? color : "#f87171" }}>
          {title}
        </div>
      </div>

      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-ink-400 text-left">
            <th className="font-medium pb-1">zᵢ</th>
            <th className="font-medium pb-1">exp(·)</th>
            <th className="font-medium pb-1">p</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e, i) => (
            <tr key={i} className="border-t border-ink-800/60">
              <td className="py-1">{e.z.toExponential(2)}</td>
              <td className="py-1">{fmtMaybe(e.exp)}</td>
              <td className="py-1">{fmtMaybe(e.prob)}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr className="border-t border-ink-700">
            <td className="py-1 text-ink-400">Σ exp</td>
            <td className="py-1 text-ink-200" colSpan={2}>
              {fmtMaybe(sumExp)} {maxExp != null && <span className="text-ink-400 ml-2">(max = {fmtMaybe(maxExp)})</span>}
            </td>
          </tr>
        </tfoot>
      </table>

      {!ok && (
        <div className="text-[11px] text-red-300 flex items-center gap-1.5">
          <AlertTriangle className="w-3 h-3" />
          Overflow — float64 ran out of range.
        </div>
      )}
    </div>
  );
}

/* =================================================================== */
/* ============================= UTILITIES =========================== */
/* =================================================================== */

function Eyebrow({
  children,
  icon,
}: {
  children: React.ReactNode;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-violet-300 font-semibold">
      {icon ?? <span className="w-1.5 h-1.5 rounded-full bg-violet-400" />}
      {children}
    </div>
  );
}

function InsightCard({
  color,
  title,
  body,
}: {
  color: string;
  title: string;
  body: React.ReactNode;
}) {
  return (
    <div
      className="rounded-xl border p-4"
      style={{
        background: `${color}0a`,
        borderColor: `${color}55`,
      }}
    >
      <div className="font-bold text-white mb-1.5">{title}</div>
      <div className="text-sm text-ink-300 leading-relaxed">{body}</div>
    </div>
  );
}

function Metric({
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
    <div className="rounded-xl bg-ink-950/70 border border-ink-800/80 p-3">
      <div className="text-[10px] uppercase tracking-wider text-ink-400">
        {label}
      </div>
      <div className="text-xl font-bold font-mono mt-1" style={color ? { color } : undefined}>
        {value}
      </div>
      {hint && <div className="text-[11px] text-ink-400 mt-1">{hint}</div>}
    </div>
  );
}

function fmtMaybe(v: number | null | undefined): string {
  if (v === null || v === undefined) return "∞ / NaN";
  if (!Number.isFinite(v)) return "∞";
  if (Math.abs(v) >= 1e6 || (Math.abs(v) < 1e-3 && v !== 0)) return v.toExponential(3);
  return v.toFixed(4);
}

function subscript(n: number): string {
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

// Suppress unused-icon imports we want available
const _icons = [ArrowRight, Workflow];
void _icons;
