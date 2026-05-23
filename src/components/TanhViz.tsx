import { useMemo, useRef, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  artanh,
  hardTanh,
  sample,
  scaledTanh,
  tanh,
  tanhDeriv,
} from "../lib/tanh";
import { sigmoid } from "../lib/sigmoid";

const C_TANH = "#22d3ee";
const C_DERIV = "#34d399";
const C_SIG = "#7c5cff";
const C_HARD = "#f472b6";
const C_SCALED = "#fbbf24";
const C_AXIS = "rgba(255,255,255,0.10)";

export default function TanhViz() {
  const [x, setX] = useState(1);
  const tx = tanh(x);
  const txp = tanhDeriv(x);

  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_260px] gap-4">
        <MainPlot x={x} onX={setX} />
        <ValuesPanel x={x} tx={tx} txp={txp} onPick={setX} />
      </div>

      <PropertiesGrid />

      <UsageGrid />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <DerivativePanel />
        <InversePanel />
        <SigmoidRelationPanel />
      </div>

      <FamilyPlot />
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
          The <span className="text-cyan-300">hyperbolic tangent</span> — an{" "}
          <em>odd, S-shaped</em> squashing function{" "}
          <InlineMath math={String.raw`\mathbb{R} \to (-1,\,1)`} />. It is the
          <em> zero-centered, rescaled sigmoid</em>:{" "}
          <InlineMath math={String.raw`\tanh(x) = 2\sigma(2x) - 1`} />, and the ratio
          of the hyperbolic sine to the hyperbolic cosine.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`\tanh(x) \;=\; \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\tanh(x) \;=\; \frac{\sinh(x)}{\cosh(x)}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\tanh'(x) \;=\; 1 - \tanh^{2}(x) \;=\; \mathrm{sech}^{2}(x)`} />
      </MathBox>
    </section>
  );
}

/* =============================== main plot =============================== */

function MainPlot({ x, onX }: { x: number; onX: (v: number) => void }) {
  const W = 720;
  const H = 360;
  const PAD = { l: 36, r: 16, t: 12, b: 28 };
  const xMin = -5;
  const xMax = 5;
  const yMin = -1.15;
  const yMax = 1.15;

  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);
  const xPx = sx(x);
  const ty = tanh(x);
  const dy = tanhDeriv(x);

  const tanhPath = useMemo(() => buildPath(sample(tanh, xMin, xMax, 220), sx, sy), []);
  const derivPath = useMemo(() => buildPath(sample(tanhDeriv, xMin, xMax, 220), sx, sy), []);

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
        <Legend
          swatches={[
            { color: C_TANH, label: "tanh(x)" },
            { color: C_DERIV, label: "tanh′(x)", dash: true },
          ]}
        />
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
        <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[-1, -0.5, 0, 0.5, 1]} W={W} H={H} PAD={PAD} />
        {/* y=0 axis */}
        <line x1={sx(xMin)} y1={sy(0)} x2={sx(xMax)} y2={sy(0)} stroke={C_AXIS} />
        {/* x=0 axis */}
        <line x1={sx(0)} y1={PAD.t} x2={sx(0)} y2={H - PAD.b} stroke={C_AXIS} />
        {/* y = ±1 asymptotes */}
        <line x1={sx(xMin)} y1={sy(1)} x2={sx(xMax)} y2={sy(1)} stroke="rgba(34,211,238,0.18)" strokeDasharray="2 5" />
        <line x1={sx(xMin)} y1={sy(-1)} x2={sx(xMax)} y2={sy(-1)} stroke="rgba(34,211,238,0.18)" strokeDasharray="2 5" />

        <path d={tanhPath} stroke={C_TANH} strokeWidth={2.4} fill="none" />
        <path d={derivPath} stroke={C_DERIV} strokeWidth={2} fill="none" strokeDasharray="6 4" />

        {/* cursor */}
        <line x1={xPx} y1={PAD.t} x2={xPx} y2={H - PAD.b} stroke="#ffffff" strokeOpacity={0.45} strokeWidth={1} />
        <circle cx={xPx} cy={sy(ty)} r={5.5} fill={C_TANH} stroke="#fff" strokeWidth={2} />
        <circle cx={xPx} cy={sy(dy)} r={4} fill={C_DERIV} stroke="#fff" strokeWidth={1.5} />

        <g pointerEvents="none">
          <text x={xPx + 8} y={sy(ty) - 8} fontSize={11} fontFamily="JetBrains Mono, monospace" fill={C_TANH}>
            tanh = {ty.toFixed(4)}
          </text>
          <text x={xPx + 8} y={sy(dy) - 8} fontSize={11} fontFamily="JetBrains Mono, monospace" fill={C_DERIV}>
            tanh′ = {dy.toFixed(4)}
          </text>
        </g>
      </svg>
    </div>
  );
}

/* =============================== values panel =============================== */

function ValuesPanel({
  x,
  tx,
  txp,
  onPick,
}: {
  x: number;
  tx: number;
  txp: number;
  onPick: (v: number) => void;
}) {
  const at = artanh(Math.max(Math.min(tx, 1 - 1e-9), -1 + 1e-9));
  const sig = sigmoid(x);
  const presets = [0, 0.5, 1, 2, -2, 4, -4];
  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-3 flex flex-col gap-2 font-mono text-sm">
      <Row label="x" value={x.toFixed(3)} color="#fff" />
      <Row label="tanh(x)" value={tx.toFixed(5)} color={C_TANH} />
      <Row label="tanh′(x)" value={txp.toFixed(5)} color={C_DERIV} />
      <Row label="1 − tanh²(x)" value={(1 - tx * tx).toFixed(5)} color="#a78bfa" hint="= tanh′" />
      <Row label="artanh(tanh(x))" value={at.toFixed(3)} color="#34d399" hint="≈ x" />
      <Row label="σ(x)" value={sig.toFixed(5)} color={C_SIG} hint="for compare" />
      <hr className="border-ink-800/80 my-1" />
      <div className="flex flex-wrap gap-1">
        {presets.map((p) => (
          <button
            key={p}
            onClick={() => onPick(p)}
            className={`px-2 py-0.5 rounded text-[11px] font-mono border transition-colors ${
              Math.abs(p - x) < 1e-3
                ? "border-cyan-400/60 text-white bg-cyan-500/10"
                : "border-ink-700 text-ink-400 hover:text-white hover:border-cyan-400/50"
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
        <InlineMath math={String.raw`\tanh(x) \in (-1,\,1)`} />
      </PropCard>
      <PropCard title="Odd function">
        <InlineMath math={String.raw`\tanh(-x) = -\tanh(x)`} />
      </PropCard>
      <PropCard title="Fixed point">
        <InlineMath math={String.raw`\tanh(0) = 0,\quad \tanh'(0) = 1`} />
      </PropCard>
      <PropCard title="Asymptotes">
        <InlineMath math={String.raw`\tanh(x) \to \pm 1\;\text{as } x \to \pm\infty`} />
      </PropCard>
      <PropCard title="Inverse (artanh)">
        <InlineMath math={String.raw`\tanh^{-1}(p) \;=\; \tfrac12 \ln\tfrac{1+p}{1-p}`} />
      </PropCard>
      <PropCard title="Sigmoid relation">
        <InlineMath math={String.raw`\tanh(x) = 2\sigma(2x) - 1`} />
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
      <UsageCard title="Where used" accent="text-cyan-300">
        <UsageItem head="RNN hidden state" body="vanilla RNN, GRU candidate h̃ — the default recurrent non-linearity." />
        <UsageItem head="LSTM cell output" body={<>candidate <InlineMath math={String.raw`\tilde{c}_t`} /> and <InlineMath math={String.raw`h_t = o_t \odot \tanh(c_t)`} /></>} />
        <UsageItem head="Continuous-control actions" body="DDPG, SAC squash actions to [−1, 1]." />
        <UsageItem head="Normalizing flows" body="invertible coupling layers use tanh for bounded transforms." />
        <UsageItem head="GAN generator output" body="image pixels in [−1, 1] before de-normalization." />
        <UsageItem head="Bounded regression heads" body="any output that must live in a fixed interval." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-emerald-300">
        <UsageItem
          head="Zero-centered output"
          body={
            <>
              <InlineMath math={String.raw`\mathbb{E}[\tanh(z)] \approx 0`} /> for
              zero-mean input ⇒ balanced gradient signs.
            </>
          }
        />
        <UsageItem
          head="Stronger slope at 0"
          body={
            <>
              <InlineMath math={String.raw`\tanh'(0) = 1`} /> vs.{" "}
              <InlineMath math={String.raw`\sigma'(0) = \tfrac14`} /> — 4× the signal.
            </>
          }
        />
        <UsageItem head="Bounded ⇒ stable" body="activations cannot explode; pairs well with recurrent dynamics." />
        <UsageItem head="Smooth & odd" body="differentiable everywhere; antisymmetric simplifies analysis." />
        <UsageItem
          head="Cheap inverse"
          body={
            <InlineMath math={String.raw`\mathrm{artanh}(p) = \tfrac12 \log\tfrac{1+p}{1-p}`} />
          }
        />
        <UsageItem
          head="Affine ↔ sigmoid"
          body={
            <InlineMath math={String.raw`\tanh(x) = 2\sigma(2x)-1`} />
          }
        />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem
          head="Vanishing gradient"
          body={
            <>
              <InlineMath math={String.raw`\tanh'(x) \le 1`} />, and{" "}
              <InlineMath math={String.raw`\to 0`} /> for{" "}
              <InlineMath math={String.raw`|x| > 3`} />.
            </>
          }
        />
        <UsageItem head="Saturation" body="gradient ≈ 0 once a neuron pins to ±1 — slow recovery." />
        <UsageItem head="exp(·) cost" body="more expensive than ReLU; matters at large activation widths." />
        <UsageItem head="Hidden layers ⇒ avoid in deep nets" body="ReLU / GELU / SiLU train faster and deeper." />
        <UsageItem head="Still better than σ for hidden" body="if you must use a sigmoid-family activation in a hidden layer, prefer tanh." />
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

/* =============================== sub-charts =============================== */

function DerivativePanel() {
  const W = 320;
  const H = 200;
  const PAD = { l: 30, r: 8, t: 10, b: 24 };
  const xMin = -5;
  const xMax = 5;
  const yMin = 0;
  const yMax = 1.15;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);
  const path = buildPath(sample(tanhDeriv, xMin, xMax, 200), sx, sy);
  const sigDerivPath = buildPath(
    sample((x) => sigmoid(x) * (1 - sigmoid(x)), xMin, xMax, 200),
    sx,
    sy
  );

  return (
    <SubCard title="Derivative">
      <BlockMath math={String.raw`\tanh'(x) = 1 - \tanh^{2}(x)`} />
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[0, 0.25, 0.5, 1]} xTicks={[-4, -2, 0, 2, 4]} W={W} H={H} PAD={PAD} compact />
        <line x1={sx(xMin)} y1={sy(1)} x2={sx(xMax)} y2={sy(1)} stroke="rgba(52,211,153,0.4)" strokeDasharray="2 4" />
        <text x={W - PAD.r - 4} y={sy(1) - 4} fontSize={9} textAnchor="end" fill={C_DERIV} fontFamily="JetBrains Mono, monospace">
          max = 1
        </text>
        <line x1={sx(xMin)} y1={sy(0.25)} x2={sx(xMax)} y2={sy(0.25)} stroke="rgba(124,92,255,0.25)" strokeDasharray="2 4" />
        <text x={W - PAD.r - 4} y={sy(0.25) - 4} fontSize={9} textAnchor="end" fill={C_SIG} fontFamily="JetBrains Mono, monospace">
          σ′ max = 1/4
        </text>
        <path d={sigDerivPath} stroke={C_SIG} strokeWidth={1.6} fill="none" strokeDasharray="4 3" />
        <path d={path} stroke={C_DERIV} strokeWidth={2} fill="none" />
        <circle cx={sx(0)} cy={sy(1)} r={3.5} fill={C_DERIV} />
      </svg>
      <p className="text-[11px] text-ink-400 leading-snug">
        Peak <InlineMath math={String.raw`\tanh'(0)=1`} /> — 4× the sigmoid peak.
        Falls below <InlineMath math={String.raw`0.01`} /> by{" "}
        <InlineMath math={String.raw`|x| \approx 3`} />.
      </p>
    </SubCard>
  );
}

function InversePanel() {
  const W = 320;
  const H = 200;
  const PAD = { l: 36, r: 8, t: 10, b: 24 };
  const pMin = -0.98;
  const pMax = 0.98;
  const yMin = -3;
  const yMax = 3;
  const sx = (v: number) => PAD.l + ((v - pMin) / (pMax - pMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);
  const path = buildPath(sample(artanh, pMin, pMax, 200), sx, sy);

  return (
    <SubCard title="Inverse · artanh">
      <BlockMath math={String.raw`\mathrm{artanh}(p) = \tfrac12 \log\tfrac{1+p}{1-p}`} />
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <Grid sx={sx} sy={sy} xMin={pMin} xMax={pMax} yTicks={[-2, 0, 2]} xTicks={[-0.9, -0.5, 0, 0.5, 0.9]} W={W} H={H} PAD={PAD} compact />
        <line x1={sx(0)} y1={PAD.t} x2={sx(0)} y2={H - PAD.b} stroke="rgba(255,255,255,0.08)" strokeDasharray="2 4" />
        <line x1={sx(pMin)} y1={sy(0)} x2={sx(pMax)} y2={sy(0)} stroke={C_AXIS} />
        <path d={path} stroke="#34d399" strokeWidth={2} fill="none" />
        <circle cx={sx(0)} cy={sy(0)} r={3.5} fill="#34d399" />
      </svg>
      <p className="text-[11px] text-ink-400 leading-snug">
        Maps <InlineMath math={String.raw`(-1,1)`} /> back to{" "}
        <InlineMath math={String.raw`\mathbb{R}`} />. Diverges at the boundary.
      </p>
    </SubCard>
  );
}

function SigmoidRelationPanel() {
  const W = 320;
  const H = 200;
  const PAD = { l: 30, r: 8, t: 10, b: 24 };
  const xMin = -4;
  const xMax = 4;
  const yMin = -1.15;
  const yMax = 1.15;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const tanhP = buildPath(sample(tanh, xMin, xMax, 200), sx, sy);
  const sigShiftedP = buildPath(
    sample((x) => 2 * sigmoid(2 * x) - 1, xMin, xMax, 200),
    sx,
    sy
  );

  return (
    <SubCard title="tanh ≡ 2σ(2x) − 1">
      <BlockMath math={String.raw`\tanh(x) = 2\sigma(2x) - 1`} />
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[-1, 0, 1]} xTicks={[-4, -2, 0, 2, 4]} W={W} H={H} PAD={PAD} compact />
        <line x1={sx(xMin)} y1={sy(0)} x2={sx(xMax)} y2={sy(0)} stroke={C_AXIS} />
        <path d={sigShiftedP} stroke={C_SIG} strokeWidth={3.5} strokeOpacity={0.55} fill="none" />
        <path d={tanhP} stroke={C_TANH} strokeWidth={2} fill="none" strokeDasharray="4 3" />
      </svg>
      <p className="text-[11px] text-ink-400 leading-snug">
        The two curves coincide exactly. tanh is a rescaled, recentered sigmoid —
        nothing more.
      </p>
    </SubCard>
  );
}

/* =============================== family comparison =============================== */

function FamilyPlot() {
  const W = 720;
  const H = 280;
  const PAD = { l: 36, r: 16, t: 12, b: 26 };
  const xMin = -5;
  const xMax = 5;
  const yMin = -1.3;
  const yMax = 1.3;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const tanhP = buildPath(sample(tanh, xMin, xMax, 220), sx, sy);
  const hardP = buildPath(sample(hardTanh, xMin, xMax, 220), sx, sy);
  const scaledP = buildPath(sample(scaledTanh, xMin, xMax, 220), sx, sy);
  const sigP = buildPath(sample((x) => 2 * sigmoid(x) - 1, xMin, xMax, 220), sx, sy);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-3">
      <div className="flex items-center justify-between mb-2 px-1">
        <Legend
          swatches={[
            { color: C_TANH, label: "tanh(x)" },
            { color: C_HARD, label: "hard-tanh = clip(x, −1, 1)" },
            { color: C_SCALED, label: "1.7159 · tanh(⅔ x)  (LeCun)" },
            { color: C_SIG, label: "2σ(x) − 1  (≠ tanh)" },
          ]}
        />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[-1, -0.5, 0, 0.5, 1]} W={W} H={H} PAD={PAD} />
        <line x1={sx(xMin)} y1={sy(0)} x2={sx(xMax)} y2={sy(0)} stroke={C_AXIS} />
        <path d={hardP} stroke={C_HARD} strokeWidth={2} fill="none" strokeDasharray="5 4" />
        <path d={scaledP} stroke={C_SCALED} strokeWidth={2} fill="none" />
        <path d={sigP} stroke={C_SIG} strokeWidth={2} fill="none" strokeDasharray="3 3" />
        <path d={tanhP} stroke={C_TANH} strokeWidth={2.4} fill="none" />
      </svg>
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

function SubCard({ title, children }: { title: string; children: React.ReactNode }) {
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
}

function Grid({ sx, sy, xMin, xMax, yTicks, xTicks, W, H, PAD, compact }: GridProps) {
  const defaultXTicks = [-5, -3, -1, 0, 1, 3, 5].filter((v) => v >= xMin && v <= xMax);
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
