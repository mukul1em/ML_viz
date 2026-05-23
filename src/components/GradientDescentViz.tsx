import { Suspense, lazy, useMemo, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  LOSSES,
  OPTIMIZER_LIST,
  runOptimizer,
  type LossId,
  type LossSpec,
  type OptimizerId,
  type OptimizerSpec,
  type Trajectory,
  type Vec2,
} from "../lib/gradientDescent";

const Surface3D = lazy(() => import("./gd/Surface3D"));

const C_AXIS = "rgba(255,255,255,0.10)";
const DEFAULT_ON: OptimizerId[] = ["sgd", "momentum", "adam"];

export default function GradientDescentViz() {
  const [lossId, setLossId] = useState<LossId>("ravine");
  const [active, setActive] = useState<Set<OptimizerId>>(new Set(DEFAULT_ON));
  const loss = LOSSES[lossId];

  const [lr, setLr] = useState(loss.defaultLr);
  const [start, setStart] = useState<Vec2>([loss.start[0], loss.start[1]]);
  const [steps, setSteps] = useState(80);
  const [beta1, setBeta1] = useState(0.9);
  const [beta2, setBeta2] = useState(0.999);
  const [wd, setWd] = useState(0);

  // When loss changes, reset start + lr to that surface's defaults.
  const handleLossChange = (id: LossId) => {
    setLossId(id);
    setLr(LOSSES[id].defaultLr);
    setStart([LOSSES[id].start[0], LOSSES[id].start[1]]);
  };

  const visible: OptimizerSpec[] = useMemo(
    () => OPTIMIZER_LIST.filter((o) => active.has(o.id)),
    [active]
  );

  const trajectories: Trajectory[] = useMemo(
    () =>
      visible.map((opt) =>
        runOptimizer(opt, {
          loss,
          start,
          lr,
          steps,
          beta1,
          beta2,
          eps: 1e-8,
          wd,
        })
      ),
    [visible, loss, start, lr, steps, beta1, beta2, wd]
  );

  const toggle = (id: OptimizerId) => {
    const next = new Set(active);
    next.has(id) ? next.delete(id) : next.add(id);
    if (next.size === 0) next.add("sgd");
    setActive(next);
  };

  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <LossPicker lossId={lossId} onChange={handleLossChange} />

      <OptimizerToggleBar active={active} onToggle={toggle} />

      <Suspense
        fallback={
          <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 h-[560px] flex items-center justify-center">
            <span className="text-ink-400 font-mono text-sm animate-pulse">
              loading 3D scene…
            </span>
          </div>
        }
      >
        <Surface3D
          loss={loss}
          start={start}
          trajectories={trajectories}
          onSetStart={setStart}
        />
      </Suspense>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_1fr] gap-4">
        <ContourPlot loss={loss} start={start} trajectories={trajectories} onSetStart={setStart} />
        <LossCurves trajectories={trajectories} />
      </div>

      <HyperParamPanel
        loss={loss}
        lr={lr}
        steps={steps}
        beta1={beta1}
        beta2={beta2}
        wd={wd}
        onLr={setLr}
        onSteps={setSteps}
        onBeta1={setBeta1}
        onBeta2={setBeta2}
        onWd={setWd}
      />

      <DecoupledWDCallout />

      <PropertiesGrid />
      <UsageGrid />
      <AlgorithmCheatsheet />
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
          <span className="text-emerald-300">Gradient descent</span> follows the
          steepest-descent direction <InlineMath math={String.raw`-\nabla L(\theta)`} />{" "}
          with a step size <InlineMath math={String.raw`\eta`} />. Variants add{" "}
          <em>momentum</em> (exponential averaging of past gradients) and{" "}
          <em>adaptive per-coordinate scaling</em> (squared-gradient averages) to
          handle ill-conditioning, noise, and saddles.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`\theta_{t+1} \;=\; \theta_t - \eta\, \nabla L(\theta_t)`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`v_{t} \;=\; \beta_1 v_{t-1} + g_t,\quad \theta_{t+1} = \theta_t - \eta v_t`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\theta_{t+1} \;=\; \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}`} />
      </MathBox>
    </section>
  );
}

/* =============================== loss picker =============================== */

function LossPicker({ lossId, onChange }: { lossId: LossId; onChange: (id: LossId) => void }) {
  const ids: LossId[] = ["bowl", "ravine", "rosenbrock", "saddle"];
  const loss = LOSSES[lossId];
  return (
    <section className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold pr-1">
          Loss surface
        </span>
        {ids.map((id) => (
          <button
            key={id}
            onClick={() => onChange(id)}
            className={`px-2.5 py-1 rounded text-[11.5px] font-mono border transition-colors ${
              lossId === id
                ? "border-emerald-400/60 text-white bg-emerald-500/15"
                : "border-ink-700 text-ink-400 hover:text-white"
            }`}
          >
            {LOSSES[id].name}
          </button>
        ))}
      </div>
      <div className="flex flex-col sm:flex-row sm:items-baseline gap-2">
        <BlockMath math={loss.formula} />
        <span className="text-[11.5px] text-ink-400 leading-snug">{loss.description}</span>
      </div>
    </section>
  );
}

/* =============================== optimizer toggle =============================== */

function OptimizerToggleBar({
  active,
  onToggle,
}: {
  active: Set<OptimizerId>;
  onToggle: (id: OptimizerId) => void;
}) {
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-2.5 flex flex-wrap items-center gap-1.5">
      <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold pr-1">
        Optimizers
      </span>
      {OPTIMIZER_LIST.map((o) => {
        const on = active.has(o.id);
        return (
          <button
            key={o.id}
            onClick={() => onToggle(o.id)}
            className="px-2.5 py-1 rounded text-[11.5px] font-mono border transition-colors flex items-center gap-1.5"
            style={{
              borderColor: on ? o.color : "rgba(255,255,255,0.10)",
              color: on ? "#fff" : "#7a8094",
              background: on ? o.color + "1f" : "transparent",
            }}
          >
            <span
              className="inline-block w-2.5 h-2.5 rounded-sm"
              style={{ background: on ? o.color : "transparent", outline: `1px solid ${o.color}` }}
            />
            {o.name}
          </button>
        );
      })}
    </div>
  );
}

/* =============================== contour plot =============================== */

function ContourPlot({
  loss,
  start,
  trajectories,
  onSetStart,
}: {
  loss: LossSpec;
  start: Vec2;
  trajectories: Trajectory[];
  onSetStart: (p: Vec2) => void;
}) {
  const W = 540;
  const H = 480;
  const PAD = { l: 30, r: 12, t: 12, b: 26 };
  const { xMin, xMax, yMin, yMax } = loss.domain;
  const sx = (x: number) => PAD.l + ((x - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (y: number) => H - PAD.b - ((y - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  // Heatmap of log-loss, memoized per loss surface.
  const N = 60;
  const cells = useMemo(() => {
    const fGrid: number[] = [];
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const x = xMin + ((xMax - xMin) * j) / (N - 1);
        const y = yMin + ((yMax - yMin) * i) / (N - 1);
        const v = loss.f([x, y]);
        const lv = Math.log1p(Math.abs(v)) * Math.sign(v);
        fGrid.push(lv);
        if (lv < lo) lo = lv;
        if (lv > hi) hi = lv;
      }
    }
    return { fGrid, lo, hi };
  }, [loss.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const cw = (W - PAD.l - PAD.r) / (N - 1);
  const ch = (H - PAD.t - PAD.b) / (N - 1);

  const colorFor = (v: number) => {
    const t = (v - cells.lo) / Math.max(cells.hi - cells.lo, 1e-9);
    // Dark navy → violet → warm. Higher loss = warmer.
    const r = Math.round(20 + 200 * t);
    const g = Math.round(15 + 40 * t);
    const b = Math.round(40 + 80 * (1 - t));
    return `rgb(${r},${g},${b})`;
  };

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const py = ((e.clientY - rect.top) / rect.height) * H;
    const x = xMin + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (xMax - xMin);
    const y = yMin + ((H - PAD.b - py) / (H - PAD.t - PAD.b)) * (yMax - yMin);
    if (x < xMin || x > xMax || y < yMin || y > yMax) return;
    onSetStart([+x.toFixed(3), +y.toFixed(3)]);
  };

  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-3 overflow-hidden">
      <div className="flex items-center justify-between text-[11px] mb-2 px-1">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Top-down · click to move start
        </span>
        <span className="text-ink-500 font-mono">heatmap of log(1+|f|)</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block cursor-crosshair" onClick={handleClick}>
        {/* heatmap */}
        <g>
          {cells.fGrid.map((v, idx) => {
            const j = idx % N;
            const i = Math.floor(idx / N);
            return (
              <rect
                key={idx}
                x={PAD.l + j * cw - cw / 2}
                y={H - PAD.b - i * ch - ch / 2}
                width={cw + 0.5}
                height={ch + 0.5}
                fill={colorFor(v)}
              />
            );
          })}
        </g>
        {/* axes */}
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke={C_AXIS} />
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke={C_AXIS} />
        {/* origin crosshair */}
        <line x1={sx(0)} y1={PAD.t} x2={sx(0)} y2={H - PAD.b} stroke="rgba(255,255,255,0.10)" strokeDasharray="2 4" />
        <line x1={PAD.l} y1={sy(0)} x2={W - PAD.r} y2={sy(0)} stroke="rgba(255,255,255,0.10)" strokeDasharray="2 4" />
        {/* tick labels */}
        {tickList(xMin, xMax).map((v) => (
          <text key={`x-${v}`} x={sx(v)} y={H - PAD.b + 14} textAnchor="middle" fontSize={9} fill="#9ca3b3" fontFamily="JetBrains Mono, monospace">
            {fmt(v)}
          </text>
        ))}
        {tickList(yMin, yMax).map((v) => (
          <text key={`y-${v}`} x={PAD.l - 4} y={sy(v) + 3} textAnchor="end" fontSize={9} fill="#9ca3b3" fontFamily="JetBrains Mono, monospace">
            {fmt(v)}
          </text>
        ))}
        {/* minima */}
        {loss.minima.map((m, idx) => (
          <g key={`min-${idx}`}>
            <circle cx={sx(m[0])} cy={sy(m[1])} r={5} fill="#fff" />
            <circle cx={sx(m[0])} cy={sy(m[1])} r={9} fill="none" stroke="#fff" strokeOpacity={0.4} />
          </g>
        ))}
        {/* trajectories */}
        {trajectories.map((traj) => {
          const d = traj.points
            .map((p, i) => {
              const cx = clamp(sx(p[0]), PAD.l, W - PAD.r);
              const cy = clamp(sy(p[1]), PAD.t, H - PAD.b);
              return `${i === 0 ? "M" : "L"} ${cx.toFixed(2)} ${cy.toFixed(2)}`;
            })
            .join(" ");
          return (
            <g key={traj.optId}>
              <path d={d} stroke={traj.color} strokeWidth={2} fill="none" strokeOpacity={0.95} />
              {traj.points.map((p, i) => {
                if (i % 4 !== 0 && i !== traj.points.length - 1) return null;
                return (
                  <circle
                    key={i}
                    cx={clamp(sx(p[0]), PAD.l, W - PAD.r)}
                    cy={clamp(sy(p[1]), PAD.t, H - PAD.b)}
                    r={i === traj.points.length - 1 ? 4 : 2}
                    fill={traj.color}
                    stroke={i === traj.points.length - 1 ? "#fff" : undefined}
                    strokeWidth={i === traj.points.length - 1 ? 1.5 : 0}
                  />
                );
              })}
            </g>
          );
        })}
        {/* start marker */}
        <g>
          <circle cx={sx(start[0])} cy={sy(start[1])} r={6} fill="none" stroke="#fff" strokeWidth={2} />
          <circle cx={sx(start[0])} cy={sy(start[1])} r={2.5} fill="#fff" />
        </g>
      </svg>
    </div>
  );
}

/* =============================== loss curves =============================== */

function LossCurves({ trajectories }: { trajectories: Trajectory[] }) {
  const W = 540;
  const H = 480;
  const PAD = { l: 44, r: 12, t: 12, b: 26 };
  const xMin = 0;
  const xMax = Math.max(1, ...trajectories.map((t) => t.losses.length - 1));
  const yMax = Math.max(
    1e-3,
    ...trajectories.flatMap((t) => t.losses.filter(Number.isFinite))
  );
  // log y for better dynamic range; clamp to avoid log(0)
  const yMinL = Math.log10(1e-3);
  const yMaxL = Math.log10(Math.max(yMax, 1e-3));
  const sx = (x: number) => PAD.l + ((x - xMin) / Math.max(xMax - xMin, 1)) * (W - PAD.l - PAD.r);
  const sy = (l: number) =>
    H - PAD.b - ((Math.log10(Math.max(l, 1e-3)) - yMinL) / (yMaxL - yMinL)) * (H - PAD.t - PAD.b);

  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-3 overflow-hidden">
      <div className="flex items-center justify-between text-[11px] mb-2 px-1">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Loss vs. step · log₁₀ scale
        </span>
        <span className="text-ink-500 font-mono">lower = better</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        {/* grid */}
        {[1, 0, -1, -2, -3].map((p) => (
          <g key={p}>
            <line x1={PAD.l} y1={sy(Math.pow(10, p))} x2={W - PAD.r} y2={sy(Math.pow(10, p))} stroke="rgba(255,255,255,0.05)" />
            <text x={PAD.l - 4} y={sy(Math.pow(10, p)) + 3} textAnchor="end" fontSize={9} fill="#9ca3b3" fontFamily="JetBrains Mono, monospace">
              10^{p}
            </text>
          </g>
        ))}
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke={C_AXIS} />
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke={C_AXIS} />
        {/* trajectories */}
        {trajectories.map((traj) => {
          const d = traj.losses
            .map((l, i) => `${i === 0 ? "M" : "L"} ${sx(i).toFixed(2)} ${sy(l).toFixed(2)}`)
            .join(" ");
          const final = traj.losses[traj.losses.length - 1];
          return (
            <g key={traj.optId}>
              <path d={d} stroke={traj.color} strokeWidth={2} fill="none" />
              <text
                x={W - PAD.r - 6}
                y={Math.max(PAD.t + 10, sy(final) - 4)}
                textAnchor="end"
                fontSize={10}
                fill={traj.color}
                fontFamily="JetBrains Mono, monospace"
              >
                {traj.name} · {Number(final).toExponential(2)}
              </text>
            </g>
          );
        })}
        {/* x label */}
        <text x={W - PAD.r} y={H - PAD.b - 4} textAnchor="end" fontSize={9} fill="#5b6478" fontFamily="JetBrains Mono, monospace">
          step
        </text>
      </svg>
    </div>
  );
}

/* =============================== hyperparams =============================== */

function HyperParamPanel({
  loss,
  lr,
  steps,
  beta1,
  beta2,
  wd,
  onLr,
  onSteps,
  onBeta1,
  onBeta2,
  onWd,
}: {
  loss: LossSpec;
  lr: number;
  steps: number;
  beta1: number;
  beta2: number;
  wd: number;
  onLr: (v: number) => void;
  onSteps: (v: number) => void;
  onBeta1: (v: number) => void;
  onBeta2: (v: number) => void;
  onWd: (v: number) => void;
}) {
  const lrMax = loss.id === "rosenbrock" ? 0.01 : 1.5;
  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
      <ParamSlider
        label="learning rate η"
        value={lr}
        min={1e-4}
        max={lrMax}
        step={lrMax / 200}
        onChange={onLr}
        format={(v) => v.toFixed(v < 0.01 ? 4 : 3)}
      />
      <ParamSlider
        label="steps"
        value={steps}
        min={10}
        max={300}
        step={1}
        onChange={onSteps}
        format={(v) => v.toFixed(0)}
      />
      <ParamSlider
        label="β₁ (momentum)"
        value={beta1}
        min={0}
        max={0.99}
        step={0.01}
        onChange={onBeta1}
        format={(v) => v.toFixed(2)}
      />
      <ParamSlider
        label="β₂ (RMS / Adam)"
        value={beta2}
        min={0.5}
        max={0.9999}
        step={0.001}
        onChange={onBeta2}
        format={(v) => v.toFixed(4)}
      />
      <ParamSlider
        label="weight decay λ (AdamW · Lion)"
        value={wd}
        min={0}
        max={0.2}
        step={0.001}
        onChange={onWd}
        format={(v) => v.toFixed(3)}
      />
    </section>
  );
}

function DecoupledWDCallout() {
  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4">
      <div className="flex flex-col sm:flex-row sm:items-baseline gap-2 sm:gap-3">
        <span className="text-[10px] uppercase tracking-wider text-violet-300 font-bold shrink-0">
          AdamW · Lion · decoupled weight decay
        </span>
        <p className="text-ink-100 text-sm leading-relaxed">
          Vanilla Adam absorbs an{" "}
          <InlineMath math={String.raw`L_2`} /> penalty into the gradient —
          which is then <em>rescaled</em> by{" "}
          <InlineMath math={String.raw`1/\sqrt{\hat v}`} />, so per-parameter
          effective decay varies wildly.{" "}
          <span className="text-violet-300">AdamW</span> (Loshchilov &amp;
          Hutter, 2019) decouples the two:{" "}
          <InlineMath math={String.raw`\theta \leftarrow \theta - \eta\!\left(\tfrac{\hat m}{\sqrt{\hat v}+\varepsilon} + \lambda\theta\right)`} />
          {" "}— every weight shrinks at the same uniform rate.{" "}
          <span className="text-rose-300">Lion</span> (Chen et al., 2023) goes
          further:{" "}
          <InlineMath math={String.raw`\theta \leftarrow \theta - \eta(\operatorname{sign}(c) + \lambda\theta)`} />
          {" "}— a single bit per coordinate, the same decoupled WD, and 3–5×
          the memory savings of Adam. Pull the λ slider above to watch both
          trajectories curve back toward the origin.
        </p>
      </div>
    </section>
  );
}

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          {label}
        </span>
        <span className="font-mono text-[12px] tabular-nums text-white">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="accent-emerald-400"
      />
    </div>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="First-order">
        <InlineMath math={String.raw`\nabla L\;\text{only — no Hessian.}`} />
      </PropCard>
      <PropCard title="Convergence (convex)">
        <InlineMath math={String.raw`\mathcal{O}(1/T)\;\text{rate; with strong-convexity }\;\mathcal{O}(e^{-T})`} />
      </PropCard>
      <PropCard title="Step-size rule">
        <InlineMath math={String.raw`\eta < \tfrac{2}{L}\;(L = \text{Lipschitz of } \nabla L)`} />
      </PropCard>
      <PropCard title="SGD variance">
        <InlineMath math={String.raw`\mathbb{E}[g_t] = \nabla L,\quad \mathrm{Var}(g_t) > 0`} />
      </PropCard>
      <PropCard title="Momentum effective LR">
        <InlineMath math={String.raw`\eta_{\text{eff}} \approx \eta / (1 - \beta_1)`} />
      </PropCard>
      <PropCard title="Adam ≈ RMSProp + Momentum">
        <InlineMath math={String.raw`\hat m_t = \tfrac{m_t}{1-\beta_1^{t}},\; \hat v_t = \tfrac{v_t}{1-\beta_2^{t}}`} />
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
      <UsageCard title="Where used" accent="text-emerald-300">
        <UsageItem head="Every deep-net trainer" body="SGD, SGD+momentum (CNN classics), Adam / AdamW (transformers)." />
        <UsageItem head="Logistic / linear models" body="closed-form rarely scales; GD/SGD does." />
        <UsageItem head="Matrix factorization" body="SGD on observed entries (collaborative filtering)." />
        <UsageItem head="RL policy gradients" body="Adam on actor / critic objectives." />
        <UsageItem head="Diffusion training" body="AdamW with cosine LR + warmup, on the noise-prediction loss." />
        <UsageItem head="GAN training" body="Adam with β₁ ≈ 0.5 (lower momentum for stability)." />
        <UsageItem head="Bayesian sampling" body="SGLD = SGD + Gaussian noise — samples from a posterior." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-cyan-300">
        <UsageItem
          head="Scalable"
          body={
            <>
              Only requires one gradient per step —{" "}
              <InlineMath math={String.raw`\mathcal{O}(d)`} /> memory, no Hessian.
            </>
          }
        />
        <UsageItem
          head="Mini-batching"
          body={
            <>
              SGD estimates <InlineMath math={String.raw`\nabla L`} /> with{" "}
              <InlineMath math={String.raw`B \ll N`} /> samples → cheap per step,
              noise regularizes.
            </>
          }
        />
        <UsageItem
          head="Momentum smooths"
          body="EMA of past gradients cancels oscillation in ill-conditioned directions."
        />
        <UsageItem
          head="Adaptive per-coordinate"
          body="AdaGrad / RMSProp / Adam rescale each parameter — robust to feature scales."
        />
        <UsageItem
          head="Adam = bias-corrected + adaptive + momentum"
          body="works out-of-the-box on most modern architectures."
        />
        <UsageItem
          head="Implicit regularization"
          body="SGD on overparameterized nets prefers flat minima → better generalization."
        />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem
          head="Step size sensitivity"
          body="too small → slow; too large → divergence. Often the dominant hyperparameter."
        />
        <UsageItem
          head="No global guarantee"
          body="non-convex landscapes ⇒ local minima, saddle points, sharp minima."
        />
        <UsageItem
          head="Ill-conditioning"
          body={
            <>
              Curvature ratio <InlineMath math={String.raw`\lambda_{\max}/\lambda_{\min}`} />{" "}
              dominates SGD convergence rate.
            </>
          }
        />
        <UsageItem
          head="Adam ≠ best generalizer"
          body="SGD+momentum often beats Adam on CNN classification (Wilson et al., 2017)."
        />
        <UsageItem
          head="Weight decay subtlety"
          body="Adam's L2 ≠ weight decay; AdamW decouples them and fixes it."
        />
        <UsageItem
          head="Warmup needed"
          body="adaptive variance estimates are unreliable for small t — warmup + LR schedule."
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

/* =============================== algorithm cheatsheet =============================== */

function AlgorithmCheatsheet() {
  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 overflow-hidden">
      <div className="px-4 py-2 border-b border-ink-800/80 text-[10px] uppercase tracking-wider text-ink-400 font-bold">
        Optimizer cheatsheet
      </div>
      <div className="grid grid-cols-[auto_1fr] text-sm">
        <div className="contents text-[10px] uppercase tracking-wider text-ink-500 font-bold">
          <div className="px-4 py-2 border-b border-ink-800/60">Optimizer</div>
          <div className="px-4 py-2 border-b border-ink-800/60">Update rule</div>
        </div>
        {OPTIMIZER_LIST.map((o) => (
          <div key={o.id} className="contents">
            <div className="px-4 py-3 border-b border-ink-800/40 flex items-center gap-2 font-mono">
              <span
                className="inline-block w-2.5 h-2.5 rounded-sm"
                style={{ background: o.color }}
              />
              <span className="text-ink-100 text-[12px]">{o.name}</span>
            </div>
            <div className="px-4 py-3 border-b border-ink-800/40">
              <BlockMath math={o.formula} />
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

function tickList(min: number, max: number): number[] {
  const span = max - min;
  const step = span <= 2 ? 0.5 : span <= 4 ? 1 : 2;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(3));
  return out;
}

function fmt(v: number): string {
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(2).replace(/\.?0+$/, "");
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
