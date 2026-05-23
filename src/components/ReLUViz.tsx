import { useMemo, useRef, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  VARIANTS,
  sample,
  type VariantId,
  type VariantSpec,
} from "../lib/relu";

const C_AXIS = "rgba(255,255,255,0.10)";

const DEFAULT_ON: VariantId[] = ["relu", "leaky", "gelu", "silu"];

export default function ReLUViz() {
  const [x, setX] = useState(1);
  const [active, setActive] = useState<Set<VariantId>>(new Set(DEFAULT_ON));

  const visible = useMemo(() => VARIANTS.filter((v) => active.has(v.id)), [active]);

  const toggle = (id: VariantId) => {
    const next = new Set(active);
    next.has(id) ? next.delete(id) : next.add(id);
    if (next.size === 0) next.add("relu");
    setActive(next);
  };

  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <VariantToggleBar active={active} onToggle={toggle} />

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-4">
        <MainPlot x={x} onX={setX} visible={visible} />
        <ValuesPanel x={x} variants={visible} onPick={setX} />
      </div>

      <PropertiesGrid />

      <UsageGrid />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DerivativePlot visible={visible} x={x} onX={setX} />
        <DyingReluPanel />
      </div>

      <VariantFormulaTable visible={visible} />
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
          The <span className="text-violet-300">Rectified Linear Unit</span> is the
          identity for positive inputs and zero elsewhere — a{" "}
          <em>piecewise-linear, half-wave rectifier</em>. Its modern descendants
          (Leaky · ELU · GELU · SiLU · Mish · Softplus) keep the cheap right-half
          slope while smoothing or leaking the dead negative half.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`\mathrm{ReLU}(x) \;=\; \max(0,\,x)`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\mathrm{ReLU}(x) \;=\; \tfrac{x + |x|}{2}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\frac{d}{dx}\mathrm{ReLU}(x) \;=\; \mathbb{1}[x > 0]`} />
      </MathBox>
    </section>
  );
}

/* =============================== variant toggles =============================== */

function VariantToggleBar({
  active,
  onToggle,
}: {
  active: Set<VariantId>;
  onToggle: (id: VariantId) => void;
}) {
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-2.5 flex flex-wrap items-center gap-1.5">
      <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold pr-1">
        Variants
      </span>
      {VARIANTS.map((v) => {
        const on = active.has(v.id);
        return (
          <button
            key={v.id}
            onClick={() => onToggle(v.id)}
            className="px-2.5 py-1 rounded text-[11.5px] font-mono border transition-colors flex items-center gap-1.5"
            style={{
              borderColor: on ? v.color : "rgba(255,255,255,0.10)",
              color: on ? "#fff" : "#7a8094",
              background: on ? v.color + "1f" : "transparent",
            }}
          >
            <span
              className="inline-block w-2.5 h-2.5 rounded-sm"
              style={{ background: on ? v.color : "transparent", outline: `1px solid ${v.color}` }}
            />
            {v.name}
          </button>
        );
      })}
    </div>
  );
}

/* =============================== main plot =============================== */

function MainPlot({
  x,
  onX,
  visible,
}: {
  x: number;
  onX: (v: number) => void;
  visible: VariantSpec[];
}) {
  const W = 720;
  const H = 380;
  const PAD = { l: 36, r: 16, t: 12, b: 28 };
  const xMin = -4;
  const xMax = 4;
  const yMin = -1.5;
  const yMax = 4;

  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);
  const xPx = sx(x);

  const paths = useMemo(
    () =>
      visible.map((v) => ({
        v,
        d: buildPath(sample(v.fn, xMin, xMax, 240), sx, sy),
      })),
    [visible]
  );

  const dragRef = useRef(false);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const setXFromEvent = (clientX: number) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = ((clientX - rect.left) / rect.width) * W;
    const newX = xMin + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (xMax - xMin);
    onX(Math.max(xMin, Math.min(xMax, +newX.toFixed(3))));
  };

  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-3 overflow-hidden">
      <div className="flex items-center justify-between text-[11px] mb-2 px-1">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          f(x)
        </span>
        <span className="text-ink-500 font-mono">click or drag the curve</span>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full block cursor-crosshair select-none"
        onMouseDown={(e) => {
          dragRef.current = true;
          setXFromEvent(e.clientX);
        }}
        onMouseMove={(e) => {
          if (dragRef.current) setXFromEvent(e.clientX);
        }}
        onMouseUp={() => (dragRef.current = false)}
        onMouseLeave={() => (dragRef.current = false)}
      >
        <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[-1, 0, 1, 2, 3, 4]} xTicks={[-4, -2, 0, 2, 4]} W={W} H={H} PAD={PAD} />
        {/* y=0 zero axis */}
        <line x1={sx(xMin)} y1={sy(0)} x2={sx(xMax)} y2={sy(0)} stroke={C_AXIS} />
        {/* x=0 */}
        <line x1={sx(0)} y1={PAD.t} x2={sx(0)} y2={H - PAD.b} stroke={C_AXIS} />
        {/* y = x reference */}
        <line
          x1={sx(0)}
          y1={sy(0)}
          x2={sx(xMax)}
          y2={sy(xMax)}
          stroke="rgba(255,255,255,0.08)"
          strokeDasharray="2 4"
        />

        {/* curves */}
        {paths.map(({ v, d }) => (
          <path key={v.id} d={d} stroke={v.color} strokeWidth={2.2} fill="none" />
        ))}

        {/* cursor */}
        <line x1={xPx} y1={PAD.t} x2={xPx} y2={H - PAD.b} stroke="#ffffff" strokeOpacity={0.45} strokeWidth={1} />
        {paths.map(({ v }) => {
          const y = v.fn(x);
          if (y < yMin || y > yMax) return null;
          return <circle key={v.id} cx={xPx} cy={sy(y)} r={4} fill={v.color} stroke="#fff" strokeWidth={1.5} />;
        })}
      </svg>
    </div>
  );
}

/* =============================== values panel =============================== */

function ValuesPanel({
  x,
  variants,
  onPick,
}: {
  x: number;
  variants: VariantSpec[];
  onPick: (v: number) => void;
}) {
  const presets = [-4, -2, -1, 0, 1, 2, 4];
  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-3 flex flex-col gap-2 font-mono text-sm">
      <Row label="x" value={x.toFixed(3)} color="#fff" />
      <hr className="border-ink-800/80 my-1" />
      {variants.map((v) => (
        <Row
          key={v.id}
          label={`${v.name}(x)`}
          value={v.fn(x).toFixed(4)}
          color={v.color}
        />
      ))}
      <hr className="border-ink-800/80 my-1" />
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold pb-1">
        Derivatives
      </div>
      {variants.map((v) => (
        <Row
          key={v.id + "-d"}
          label={`${v.name}′`}
          value={v.deriv(x).toFixed(4)}
          color={v.color}
          dim
        />
      ))}
      <hr className="border-ink-800/80 my-1" />
      <div className="flex flex-wrap gap-1">
        {presets.map((p) => (
          <button
            key={p}
            onClick={() => onPick(p)}
            className={`px-2 py-0.5 rounded text-[11px] font-mono border transition-colors ${
              Math.abs(p - x) < 1e-3
                ? "border-violet-400/60 text-white bg-violet-500/10"
                : "border-ink-700 text-ink-400 hover:text-white hover:border-violet-400/50"
            }`}
          >
            x = {p}
          </button>
        ))}
      </div>
    </div>
  );
}

function Row({
  label,
  value,
  color,
  dim,
}: {
  label: string;
  value: string;
  color?: string;
  dim?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className={`text-xs ${dim ? "text-ink-500" : "text-ink-400"}`}>{label}</span>
      <span
        className={`tabular-nums ${dim ? "font-medium opacity-80" : "font-semibold"}`}
        style={color ? { color } : undefined}
      >
        {value}
      </span>
    </div>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="Range">
        <InlineMath math={String.raw`\mathrm{ReLU}(x) \in [0,\,\infty)`} />
      </PropCard>
      <PropCard title="Identity at +">
        <InlineMath math={String.raw`x > 0 \;\Rightarrow\; \mathrm{ReLU}(x) = x`} />
      </PropCard>
      <PropCard title="Sparsity">
        <InlineMath math={String.raw`\Pr[\mathrm{ReLU}(z)=0] = \Pr[z \le 0]`} />
      </PropCard>
      <PropCard title="Homogeneous">
        <InlineMath math={String.raw`\mathrm{ReLU}(\lambda x) = \lambda\,\mathrm{ReLU}(x),\;\lambda \ge 0`} />
      </PropCard>
      <PropCard title="Non-differentiable">
        <InlineMath math={String.raw`\partial \mathrm{ReLU}(0) = [0,\,1]`} />
      </PropCard>
      <PropCard title="Cheap">
        <InlineMath math={String.raw`O(1):\; \text{compare + select}`} />
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

/* =============================== where / why / limitations =============================== */

function UsageGrid() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <UsageCard title="Where used" accent="text-violet-300">
        <UsageItem head="CNN hidden activations" body="AlexNet, VGG, ResNet — the default since 2012." />
        <UsageItem head="MLP hidden layers" body="standard non-linearity in feedforward stacks." />
        <UsageItem head="Transformer FFN" body={<>GELU (BERT, GPT-2), SiLU (LLaMA, Qwen via SwiGLU).</>} />
        <UsageItem head="Object detection backbones" body="YOLO, Faster-RCNN, RetinaNet." />
        <UsageItem head="Diffusion U-Nets" body="SiLU / Swish in residual blocks." />
        <UsageItem head="Mobile / edge" body="ReLU6 — bounded for fixed-point quantization." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-cyan-300">
        <UsageItem
          head="Mitigates vanishing gradients"
          body={
            <>
              Derivative is exactly <InlineMath math={String.raw`1`} /> for{" "}
              <InlineMath math={String.raw`x>0`} /> — gradient propagates unattenuated.
            </>
          }
        />
        <UsageItem head="Cheap to compute" body="single comparison; no exp/log on the fast path." />
        <UsageItem
          head="Induces sparsity"
          body={
            <>
              About <InlineMath math={String.raw`50\%`} /> of units fire on
              zero-mean input — implicit regularization.
            </>
          }
        />
        <UsageItem
          head="Piecewise-linear"
          body="composition stays piecewise-linear ⇒ exact characterization of NN regions."
        />
        <UsageItem
          head="Scale-invariant on +"
          body={<InlineMath math={String.raw`\mathrm{ReLU}(\lambda x) = \lambda \mathrm{ReLU}(x)`} />}
        />
        <UsageItem
          head="Empirically faster training"
          body="6× speedup over tanh on AlexNet (Krizhevsky et al., 2012)."
        />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem
          head="Dying ReLU"
          body={
            <>
              If a unit gets stuck at <InlineMath math={String.raw`x \le 0`} />, its
              gradient is <InlineMath math={String.raw`0`} /> — never recovers.
            </>
          }
        />
        <UsageItem
          head="Not zero-centered"
          body="positive-only output biases downstream gradient sign."
        />
        <UsageItem
          head="Unbounded above"
          body="can produce huge activations ⇒ pair with normalization."
        />
        <UsageItem
          head="Non-smooth at 0"
          body="kink prevents some second-order methods; smoothed variants exist."
        />
        <UsageItem
          head="Half-input wasted"
          body="Leaky / ELU / GELU / SiLU recover the negative half."
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

/* =============================== derivative plot =============================== */

function DerivativePlot({
  visible,
  x,
  onX,
}: {
  visible: VariantSpec[];
  x: number;
  onX: (v: number) => void;
}) {
  const W = 400;
  const H = 280;
  const PAD = { l: 32, r: 12, t: 12, b: 24 };
  const xMin = -4;
  const xMax = 4;
  const yMin = -0.1;
  const yMax = 1.3;

  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const paths = useMemo(
    () =>
      visible.map((v) => ({
        v,
        d: buildPath(sample(v.deriv, xMin, xMax, 220), sx, sy),
      })),
    [visible]
  );

  const dragRef = useRef(false);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const setXFromEvent = (clientX: number) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = ((clientX - rect.left) / rect.width) * W;
    const newX = xMin + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (xMax - xMin);
    onX(Math.max(xMin, Math.min(xMax, +newX.toFixed(3))));
  };
  const xPx = sx(x);

  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-3 flex flex-col gap-1">
      <div className="flex items-center justify-between px-1">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          f ′(x) — gradient flow
        </span>
        <span className="text-ink-500 text-[10px] font-mono">click to scrub</span>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full block cursor-crosshair select-none"
        onMouseDown={(e) => {
          dragRef.current = true;
          setXFromEvent(e.clientX);
        }}
        onMouseMove={(e) => {
          if (dragRef.current) setXFromEvent(e.clientX);
        }}
        onMouseUp={() => (dragRef.current = false)}
        onMouseLeave={() => (dragRef.current = false)}
      >
        <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[0, 0.5, 1]} xTicks={[-4, -2, 0, 2, 4]} W={W} H={H} PAD={PAD} compact />
        <line x1={sx(xMin)} y1={sy(0)} x2={sx(xMax)} y2={sy(0)} stroke={C_AXIS} />
        <line x1={sx(xMin)} y1={sy(1)} x2={sx(xMax)} y2={sy(1)} stroke="rgba(255,255,255,0.08)" strokeDasharray="2 4" />
        {paths.map(({ v, d }) => (
          <path key={v.id} d={d} stroke={v.color} strokeWidth={2} fill="none" />
        ))}
        <line x1={xPx} y1={PAD.t} x2={xPx} y2={H - PAD.b} stroke="#ffffff" strokeOpacity={0.35} />
      </svg>
      <p className="text-[11px] text-ink-400 leading-snug px-1">
        ReLU′ is a hard <InlineMath math={String.raw`0/1`} /> step. Leaky lifts the
        dead half to <InlineMath math={String.raw`\alpha`} />. GELU · SiLU · Mish are
        smooth — sometimes <InlineMath math={String.raw`>1`} /> just past zero.
      </p>
    </div>
  );
}

/* =============================== dying ReLU =============================== */

function DyingReluPanel() {
  const W = 400;
  const H = 280;
  const PAD = { l: 32, r: 12, t: 12, b: 24 };
  const xMin = -4;
  const xMax = 4;
  const yMin = -2.5;
  const yMax = 4;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const reluPath = buildPath(sample((x) => Math.max(0, x), xMin, xMax, 200), sx, sy);
  const leakyPath = buildPath(
    sample((x) => (x >= 0 ? x : 0.1 * x), xMin, xMax, 200),
    sx,
    sy
  );

  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-3 flex flex-col gap-1">
      <div className="flex items-center gap-2 px-1">
        <span className="text-[10px] uppercase tracking-wider text-amber-300 font-bold">
          Dying ReLU
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[-2, 0, 2, 4]} xTicks={[-4, -2, 0, 2, 4]} W={W} H={H} PAD={PAD} compact />
        {/* shaded dead zone */}
        <rect
          x={sx(xMin)}
          y={PAD.t}
          width={sx(0) - sx(xMin)}
          height={H - PAD.t - PAD.b}
          fill="rgba(239,68,68,0.06)"
        />
        <text x={sx(-2)} y={PAD.t + 12} fontSize={10} fill="#f87171" textAnchor="middle" fontFamily="JetBrains Mono, monospace">
          dead zone (∇=0)
        </text>
        <line x1={sx(xMin)} y1={sy(0)} x2={sx(xMax)} y2={sy(0)} stroke={C_AXIS} />
        <path d={leakyPath} stroke="#22d3ee" strokeWidth={2} fill="none" strokeDasharray="5 4" />
        <path d={reluPath} stroke="#7c5cff" strokeWidth={2.4} fill="none" />
      </svg>
      <p className="text-[11px] text-ink-400 leading-snug px-1">
        With <InlineMath math={String.raw`\mathrm{ReLU}'(x)=0`} /> for{" "}
        <InlineMath math={String.raw`x \le 0`} />, units that drift into the dead zone
        can never produce a gradient signal again. Leaky ReLU&apos;s slope of{" "}
        <InlineMath math={String.raw`\alpha`} /> in that region is the simplest fix.
      </p>
    </div>
  );
}

/* =============================== formula table =============================== */

function VariantFormulaTable({ visible }: { visible: VariantSpec[] }) {
  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 overflow-hidden">
      <div className="px-4 py-2 border-b border-ink-800/80 text-[10px] uppercase tracking-wider text-ink-400 font-bold">
        Variant cheatsheet
      </div>
      <div className="grid grid-cols-[auto_1fr_1fr] text-sm">
        <div className="contents text-[10px] uppercase tracking-wider text-ink-500 font-bold">
          <div className="px-4 py-2 border-b border-ink-800/60">Name</div>
          <div className="px-4 py-2 border-b border-ink-800/60">f(x)</div>
          <div className="px-4 py-2 border-b border-ink-800/60">f ′(x)</div>
        </div>
        {visible.map((v) => (
          <div key={v.id} className="contents">
            <div className="px-4 py-3 border-b border-ink-800/40 flex items-center gap-2 font-mono">
              <span
                className="inline-block w-2.5 h-2.5 rounded-sm"
                style={{ background: v.color }}
              />
              <span className="text-ink-100 text-[12px]">{v.name}</span>
            </div>
            <div className="px-4 py-3 border-b border-ink-800/40">
              <InlineMath math={v.formula} />
            </div>
            <div className="px-4 py-3 border-b border-ink-800/40">
              <InlineMath math={v.derivFormula} />
            </div>
          </div>
        ))}
      </div>
    </section>
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

interface GridProps {
  sx: (v: number) => number;
  sy: (v: number) => number;
  xMin: number;
  xMax: number;
  yTicks: number[];
  xTicks?: number[];
  W: number;
  H: number;
  PAD: { l: number; r: number; t: number; b: number };
  compact?: boolean;
}

function Grid({ sx, sy, xMin, xMax, yTicks, xTicks, W, H, PAD, compact }: GridProps) {
  const defaultXTicks = [-4, -2, 0, 2, 4].filter((v) => v >= xMin && v <= xMax);
  const xt = xTicks ?? defaultXTicks;
  const tickFs = compact ? 9 : 10;
  return (
    <g>
      <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke={C_AXIS} />
      <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke={C_AXIS} />
      {yTicks.map((v) => (
        <g key={`y-${v}`}>
          <line x1={PAD.l - 4} y1={sy(v)} x2={W - PAD.r} y2={sy(v)} stroke="rgba(255,255,255,0.04)" />
          <text
            x={PAD.l - 6}
            y={sy(v) + 3}
            textAnchor="end"
            fontSize={tickFs}
            fill="#7a8094"
            fontFamily="JetBrains Mono, monospace"
          >
            {fmt(v)}
          </text>
        </g>
      ))}
      {xt.map((v) => (
        <g key={`x-${v}`}>
          <line x1={sx(v)} y1={H - PAD.b} x2={sx(v)} y2={H - PAD.b + 4} stroke={C_AXIS} />
          <text
            x={sx(v)}
            y={H - PAD.b + 14}
            textAnchor="middle"
            fontSize={tickFs}
            fill="#7a8094"
            fontFamily="JetBrains Mono, monospace"
          >
            {fmt(v)}
          </text>
        </g>
      ))}
    </g>
  );
}

function buildPath(
  pts: { x: number; y: number }[],
  sx: (v: number) => number,
  sy: (v: number) => number
): string {
  return pts
    .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x).toFixed(2)} ${sy(p.y).toFixed(2)}`)
    .join(" ");
}

function fmt(v: number): string {
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(2).replace(/\.?0+$/, "");
}
