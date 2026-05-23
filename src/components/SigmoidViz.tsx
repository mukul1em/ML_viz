import { useMemo, useRef, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  bce,
  hardSigmoid,
  logit,
  sample,
  sigmoid,
  sigmoidDeriv,
  tanh,
} from "../lib/sigmoid";

const C_SIGMOID = "#7c5cff";
const C_DERIV = "#22d3ee";
const C_TANH = "#34d399";
const C_HARD = "#f472b6";
const C_AXIS = "rgba(255,255,255,0.10)";

export default function SigmoidViz() {
  const [x, setX] = useState(1);

  const sx = sigmoid(x);
  const sxp = sigmoidDeriv(x);

  return (
    <div className="flex flex-col gap-5">
      {/* Definition + key identities */}
      <DefinitionCard />
      <DefinitionRow />

      {/* Main plot + values */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_260px] gap-4">
        <MainPlot x={x} onX={setX} />
        <ValuesPanel x={x} sx={sx} sxp={sxp} onPick={setX} />
      </div>

      {/* Properties — math-only cards */}
      <PropertiesGrid />

      {/* Where / Why / Limitations */}
      <UsageGrid />

      {/* Three focused sub-charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <DerivativePanel />
        <InversePanel />
        <BCEPanel />
      </div>

      {/* Family comparison */}
      <FamilyPlot />
    </div>
  );
}

/* =============================== top row =============================== */

function DefinitionCard() {
  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4">
      <div className="flex flex-col sm:flex-row sm:items-baseline gap-2 sm:gap-3">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold shrink-0">
          Definition
        </span>
        <p className="text-ink-100 text-sm leading-relaxed">
          The <span className="text-violet-300">logistic function</span>: a smooth,
          monotonically increasing map{" "}
          <InlineMath math={String.raw`\mathbb{R} \to (0,1)`} /> that converts a
          real-valued <em>logit</em>{" "}
          <InlineMath math={String.raw`x = \ln \tfrac{p}{1-p}`} /> back into a{" "}
          <em>probability</em> <InlineMath math={String.raw`p`} />. Equivalently, it is
          the cumulative distribution function of the standard logistic distribution
          and the canonical inverse-link of the Bernoulli GLM.
        </p>
      </div>
    </section>
  );
}

function DefinitionRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-2 gap-3">
      <MathBox>
        <BlockMath math={String.raw`\sigma(x) \;=\; \frac{1}{1 + e^{-x}} \;=\; \frac{e^{x}}{1 + e^{x}}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\sigma'(x) \;=\; \sigma(x)\,\bigl(1 - \sigma(x)\bigr)`} />
      </MathBox>
    </section>
  );
}

/* =============================== where / why / limits =============================== */

function UsageGrid() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <UsageCard title="Where used" accent="text-violet-300">
        <UsageItem head="Binary classifier head" body="logistic regression, NN output layer" />
        <UsageItem head="LSTM / GRU gates" body={<>input · forget · output (LSTM); update · reset (GRU)</>} />
        <UsageItem head="Mixture-of-Experts router" body="soft gating before top-k" />
        <UsageItem head="Calibration" body="Platt scaling on classifier logits" />
        <UsageItem head="Diffusion schedules" body="sigmoid β-schedule for noise" />
        <UsageItem head="SiLU / Swish" body={<InlineMath math={String.raw`\mathrm{SiLU}(x) = x\,\sigma(x)`} />} />
        <UsageItem head="GAN discriminator" body="real-vs-fake probability output" />
      </UsageCard>

      <UsageCard title="Why used" accent="text-cyan-300">
        <UsageItem
          head="Probabilistic output"
          body={
            <>
              <InlineMath math={String.raw`\sigma(x) \in (0,1)`} /> ⇒ valid Bernoulli
              parameter.
            </>
          }
        />
        <UsageItem
          head="Smooth & differentiable"
          body="∞-differentiable, no kinks ⇒ gradient flows everywhere."
        />
        <UsageItem
          head="Cheap gradient"
          body={
            <>
              <InlineMath math={String.raw`\sigma' = \sigma(1-\sigma)`} /> reuses the
              forward value — one extra multiply.
            </>
          }
        />
        <UsageItem
          head="Log-odds bijection"
          body={
            <>
              Inverse <InlineMath math={String.raw`\mathrm{logit}`} /> links additive
              models to probabilities.
            </>
          }
        />
        <UsageItem
          head="Soft 0/1 gate"
          body="continuous relaxation of a hard switch ⇒ end-to-end trainable."
        />
        <UsageItem
          head="Connection to softmax"
          body={
            <InlineMath math={String.raw`\sigma(x) = \mathrm{softmax}([x, 0])_1`} />
          }
        />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem
          head="Vanishing gradient"
          body={
            <>
              <InlineMath math={String.raw`\max \sigma' = \tfrac14`} /> at{" "}
              <InlineMath math={String.raw`x=0`} />; decays exponentially in{" "}
              <InlineMath math={String.raw`|x|`} />.
            </>
          }
        />
        <UsageItem
          head="Not zero-centered"
          body="output mean ≈ 0.5 ⇒ correlated gradient updates, slow convergence."
        />
        <UsageItem
          head="Saturation"
          body={
            <>
              For <InlineMath math={String.raw`|x| > 5`} />,{" "}
              <InlineMath math={String.raw`\sigma' \approx 0`} /> — neurons stop
              learning.
            </>
          }
        />
        <UsageItem
          head="Hidden layers ⇒ avoid"
          body="modern stacks prefer ReLU / GELU / SiLU; σ stays at gates & outputs."
        />
        <UsageItem
          head="exp(·) cost"
          body="more expensive than ReLU per element, requires stable formulation."
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

function UsageItem({
  head,
  body,
}: {
  head: string;
  body: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-0.5 border-l-2 border-ink-800 pl-2.5">
      <div className="text-ink-100 text-[12px] font-mono font-semibold">{head}</div>
      <div className="text-ink-400 text-[11.5px] leading-snug">{body}</div>
    </div>
  );
}

/* =============================== main plot =============================== */

function MainPlot({ x, onX }: { x: number; onX: (v: number) => void }) {
  const W = 720;
  const H = 360;
  const PAD = { l: 36, r: 16, t: 12, b: 28 };
  const xMin = -8;
  const xMax = 8;
  const yMin = -0.05;
  const yMax = 1.05;

  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);
  const xPx = sx(x);
  const sigVal = sigmoid(x);
  const derVal = sigmoidDeriv(x);

  const sigPath = useMemo(() => buildPath(sample(sigmoid, xMin, xMax, 220), sx, sy), []);
  const derPath = useMemo(() => buildPath(sample(sigmoidDeriv, xMin, xMax, 220), sx, sy), []);

  const dragRef = useRef(false);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const setXFromEvent = (clientX: number) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = ((clientX - rect.left) / rect.width) * W;
    const vbX = px;
    const newX = xMin + ((vbX - PAD.l) / (W - PAD.l - PAD.r)) * (xMax - xMin);
    onX(Math.max(xMin, Math.min(xMax, +newX.toFixed(3))));
  };

  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-3 overflow-hidden">
      <div className="flex items-center justify-between text-[11px] mb-2 px-1">
        <Legend swatches={[
          { color: C_SIGMOID, label: "σ(x)" },
          { color: C_DERIV, label: "σ′(x)", dash: true },
        ]} />
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
        <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[0, 0.25, 0.5, 0.75, 1]} W={W} H={H} PAD={PAD} />

        {/* Reference: y=0.5 */}
        <line x1={sx(xMin)} y1={sy(0.5)} x2={sx(xMax)} y2={sy(0.5)} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 4" />
        {/* y=1 and y=0 asymptotes */}
        <line x1={sx(xMin)} y1={sy(1)} x2={sx(xMax)} y2={sy(1)} stroke="rgba(124,92,255,0.18)" strokeDasharray="2 5" />
        <line x1={sx(xMin)} y1={sy(0)} x2={sx(xMax)} y2={sy(0)} stroke="rgba(124,92,255,0.18)" strokeDasharray="2 5" />

        {/* σ(x) */}
        <path d={sigPath} stroke={C_SIGMOID} strokeWidth={2.4} fill="none" />
        {/* σ'(x) */}
        <path d={derPath} stroke={C_DERIV} strokeWidth={2} fill="none" strokeDasharray="6 4" />

        {/* cursor */}
        <line x1={xPx} y1={PAD.t} x2={xPx} y2={H - PAD.b} stroke="#ffffff" strokeOpacity={0.45} strokeWidth={1} />
        <circle cx={xPx} cy={sy(sigVal)} r={5.5} fill={C_SIGMOID} stroke="#fff" strokeWidth={2} />
        <circle cx={xPx} cy={sy(derVal)} r={4} fill={C_DERIV} stroke="#fff" strokeWidth={1.5} />

        {/* readouts near cursor */}
        <g pointerEvents="none">
          <text x={xPx + 8} y={sy(sigVal) - 8} fontSize={11} fontFamily="JetBrains Mono, monospace" fill={C_SIGMOID}>
            σ = {sigVal.toFixed(4)}
          </text>
          <text x={xPx + 8} y={sy(derVal) - 8} fontSize={11} fontFamily="JetBrains Mono, monospace" fill={C_DERIV}>
            σ′ = {derVal.toFixed(4)}
          </text>
        </g>
      </svg>
    </div>
  );
}

/* =============================== values panel =============================== */

function ValuesPanel({
  x,
  sx,
  sxp,
  onPick,
}: {
  x: number;
  sx: number;
  sxp: number;
  onPick: (v: number) => void;
}) {
  const oneMinus = 1 - sx;
  const lg = logit(Math.min(Math.max(sx, 1e-9), 1 - 1e-9));
  const presets = [0, 1, 2, -2, 4, -4, 8];
  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-3 flex flex-col gap-2 font-mono text-sm">
      <Row label="x" value={x.toFixed(3)} color="#ffffff" />
      <Row label="σ(x)" value={sx.toFixed(5)} color={C_SIGMOID} />
      <Row label="1 − σ(x)" value={oneMinus.toFixed(5)} color="#a78bfa" />
      <Row label="σ′(x)" value={sxp.toFixed(5)} color={C_DERIV} />
      <Row label="logit(σ(x))" value={lg.toFixed(3)} color="#34d399" hint="≈ x" />
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
  hint,
}: {
  label: string;
  value: string;
  color?: string;
  hint?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-ink-400 text-xs">{label}</span>
      <span className="tabular-nums font-semibold" style={color ? { color } : undefined}>
        {value}
        {hint && <span className="text-ink-500 text-[10px] ml-1">{hint}</span>}
      </span>
    </div>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="Range">
        <InlineMath math={String.raw`\sigma(x) \in (0,\,1)`} />
      </PropCard>
      <PropCard title="Symmetry">
        <InlineMath math={String.raw`1 - \sigma(x) \;=\; \sigma(-x)`} />
      </PropCard>
      <PropCard title="Fixed point">
        <InlineMath math={String.raw`\sigma(0) = \tfrac12, \quad \sigma'(0) = \tfrac14`} />
      </PropCard>
      <PropCard title="Asymptotes">
        <InlineMath math={String.raw`\sigma(x) \to 1 \;\text{as } x \to +\infty, \; \sigma(x) \to 0 \;\text{as } x \to -\infty`} />
      </PropCard>
      <PropCard title="Inverse (logit)">
        <InlineMath math={String.raw`\sigma^{-1}(p) \;=\; \ln \tfrac{p}{1-p}`} />
      </PropCard>
      <PropCard title="Related">
        <InlineMath math={String.raw`\tanh(x) \;=\; 2\sigma(2x) - 1`} />
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

/* =============================== sub-charts =============================== */

function DerivativePanel() {
  const W = 320;
  const H = 200;
  const PAD = { l: 30, r: 8, t: 10, b: 24 };
  const xMin = -8;
  const xMax = 8;
  const yMin = 0;
  const yMax = 0.3;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);
  const path = buildPath(sample(sigmoidDeriv, xMin, xMax, 200), sx, sy);

  return (
    <SubCard title="Derivative">
      <BlockMath math={String.raw`\sigma'(x) = \sigma(x)(1-\sigma(x))`} />
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[0, 0.1, 0.2, 0.25]} W={W} H={H} PAD={PAD} compact />
        {/* Max reference */}
        <line x1={sx(xMin)} y1={sy(0.25)} x2={sx(xMax)} y2={sy(0.25)} stroke="rgba(34,211,238,0.4)" strokeDasharray="2 4" />
        <text x={W - PAD.r - 4} y={sy(0.25) - 4} fontSize={9} textAnchor="end" fill="#22d3ee" fontFamily="JetBrains Mono, monospace">
          max = 1/4
        </text>
        <path d={path} stroke={C_DERIV} strokeWidth={2} fill="none" />
        <circle cx={sx(0)} cy={sy(0.25)} r={3.5} fill={C_DERIV} />
      </svg>
      <p className="text-[11px] text-ink-400 leading-snug">
        Peaks at <InlineMath math={String.raw`x=0, \sigma'(0) = \tfrac14`} />.
        Below ~0.01 once |x| &gt; 4 — the <span className="text-amber-300">vanishing-gradient</span> regime.
      </p>
    </SubCard>
  );
}

function InversePanel() {
  const W = 320;
  const H = 200;
  const PAD = { l: 36, r: 8, t: 10, b: 24 };
  const pMin = 0.01;
  const pMax = 0.99;
  const yMin = -5;
  const yMax = 5;
  const sx = (v: number) => PAD.l + ((v - pMin) / (pMax - pMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const path = buildPath(sample(logit, pMin, pMax, 200), sx, sy);

  return (
    <SubCard title="Inverse · logit">
      <BlockMath math={String.raw`\text{logit}(p) = \ln\frac{p}{1-p}`} />
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <Grid
          sx={sx}
          sy={sy}
          xMin={pMin}
          xMax={pMax}
          yTicks={[-4, -2, 0, 2, 4]}
          xTicks={[0.1, 0.3, 0.5, 0.7, 0.9]}
          W={W}
          H={H}
          PAD={PAD}
          compact
          xLabel="p"
        />
        <line x1={sx(0.5)} y1={PAD.t} x2={sx(0.5)} y2={H - PAD.b} stroke="rgba(255,255,255,0.08)" strokeDasharray="2 4" />
        <path d={path} stroke="#34d399" strokeWidth={2} fill="none" />
        <circle cx={sx(0.5)} cy={sy(0)} r={3.5} fill="#34d399" />
      </svg>
      <p className="text-[11px] text-ink-400 leading-snug">
        Maps probabilities back to logits.{" "}
        <InlineMath math={String.raw`p \to 0,1`} /> sends logit → <InlineMath math={String.raw`\pm\infty`} />.
      </p>
    </SubCard>
  );
}

function BCEPanel() {
  const W = 320;
  const H = 200;
  const PAD = { l: 30, r: 8, t: 10, b: 24 };
  const xMin = -6;
  const xMax = 6;
  const yMin = 0;
  const yMax = 6;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const lossY1 = buildPath(
    sample((z) => bce(z, 1).loss, xMin, xMax, 200),
    sx,
    sy
  );
  const lossY0 = buildPath(
    sample((z) => bce(z, 0).loss, xMin, xMax, 200),
    sx,
    sy
  );

  return (
    <SubCard title="Binary cross-entropy">
      <BlockMath math={String.raw`L = \log(1 + e^{z}) - y\,z`} />
      <BlockMath math={String.raw`\partial L / \partial z = \sigma(z) - y`} />
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[0, 2, 4, 6]} W={W} H={H} PAD={PAD} compact xLabel="z" />
        <path d={lossY1} stroke="#34d399" strokeWidth={2} fill="none" />
        <path d={lossY0} stroke="#f87171" strokeWidth={2} fill="none" />
      </svg>
      <div className="flex items-center gap-3 text-[11px]">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-1 bg-emerald-400 rounded-sm" /> y = 1
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-1 bg-red-400 rounded-sm" /> y = 0
        </span>
      </div>
    </SubCard>
  );
}

/* =============================== family comparison =============================== */

function FamilyPlot() {
  const W = 720;
  const H = 260;
  const PAD = { l: 36, r: 16, t: 12, b: 26 };
  const xMin = -6;
  const xMax = 6;
  const yMin = -1.1;
  const yMax = 1.1;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const sigP = buildPath(sample(sigmoid, xMin, xMax, 200), sx, sy);
  const tanhP = buildPath(sample(tanh, xMin, xMax, 200), sx, sy);
  const hardP = buildPath(sample(hardSigmoid, xMin, xMax, 200), sx, sy);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-3">
      <div className="flex items-center justify-between mb-2 px-1">
        <Legend
          swatches={[
            { color: C_SIGMOID, label: "σ(x)" },
            { color: C_TANH, label: "tanh(x) = 2σ(2x)−1" },
            { color: C_HARD, label: "hard-σ(x) = clip(0.2x + 0.5, 0, 1)" },
          ]}
        />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[-1, -0.5, 0, 0.5, 1]} W={W} H={H} PAD={PAD} />
        <line x1={sx(xMin)} y1={sy(0)} x2={sx(xMax)} y2={sy(0)} stroke={C_AXIS} />
        <path d={sigP} stroke={C_SIGMOID} strokeWidth={2.2} fill="none" />
        <path d={tanhP} stroke={C_TANH} strokeWidth={2} fill="none" />
        <path d={hardP} stroke={C_HARD} strokeWidth={2} fill="none" strokeDasharray="5 4" />
      </svg>
    </section>
  );
}

/* =============================== shared primitives =============================== */

function MathBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-3 flex items-center justify-center">
      {children}
    </div>
  );
}

function SubCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-3.5 flex flex-col gap-2">
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
        {title}
      </div>
      {children}
    </div>
  );
}

function Legend({
  swatches,
}: {
  swatches: { color: string; label: string; dash?: boolean }[];
}) {
  return (
    <div className="flex items-center gap-3 flex-wrap">
      {swatches.map((s) => (
        <span key={s.label} className="flex items-center gap-1.5 text-[11px] font-mono text-ink-200">
          <span
            className="inline-block"
            style={{
              width: 16,
              height: 2,
              background: s.color,
              outline: s.dash ? "none" : undefined,
              borderTop: s.dash ? `2px dashed ${s.color}` : undefined,
            }}
          />
          {s.label}
        </span>
      ))}
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
  xLabel?: string;
}

function Grid({ sx, sy, xMin, xMax, yTicks, xTicks, W, H, PAD, compact, xLabel = "x" }: GridProps) {
  const defaultXTicks = [-8, -6, -4, -2, 0, 2, 4, 6, 8].filter((v) => v >= xMin && v <= xMax);
  const xt = xTicks ?? defaultXTicks;
  const tickFs = compact ? 9 : 10;
  return (
    <g>
      {/* Axes */}
      <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke={C_AXIS} />
      <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke={C_AXIS} />
      {/* Y ticks */}
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
            {formatTick(v)}
          </text>
        </g>
      ))}
      {/* X ticks */}
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
            {formatTick(v)}
          </text>
        </g>
      ))}
      <text
        x={W - PAD.r}
        y={H - PAD.b - 4}
        textAnchor="end"
        fontSize={tickFs}
        fill="#5b6478"
        fontFamily="JetBrains Mono, monospace"
      >
        {xLabel}
      </text>
    </g>
  );
}

/* =============================== utils =============================== */

function buildPath(
  pts: { x: number; y: number }[],
  sx: (v: number) => number,
  sy: (v: number) => number
): string {
  return pts
    .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x).toFixed(2)} ${sy(p.y).toFixed(2)}`)
    .join(" ");
}

function formatTick(v: number): string {
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(2).replace(/\.?0+$/, "");
}
