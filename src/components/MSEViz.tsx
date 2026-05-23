import { useMemo, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  l1Fit,
  mae,
  makeScatter,
  mse,
  olsFit,
  rmse,
  squaredError,
  squaredErrorDeriv,
  type ScatterPoint,
} from "../lib/mse";

const C_AXIS = "rgba(255,255,255,0.10)";
const C_LOSS = "#f87171";
const C_GRAD = "#fbbf24";
const C_MSE = "#7c5cff";
const C_MAE = "#22d3ee";
const C_TARGET = "#22d3ee";
const C_PRED = "#7c5cff";

export default function MSEViz() {
  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <InteractiveResidualSection />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <LossVsResidualPanel />
        <GradientPanel />
      </div>

      <OutlierLab />

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
          <span className="text-rose-300">Mean Squared Error</span> is the average
          of squared residuals{" "}
          <InlineMath math={String.raw`r_i = \hat y_i - y_i`} /> — the maximum-likelihood
          estimator under <em>Gaussian noise</em>. The square makes large errors
          dominate, gives a single global minimum, and yields the clean linear
          gradient <InlineMath math={String.raw`\partial L/\partial \hat y = 2(\hat y - y)/N`} />.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`\mathrm{MSE} \;=\; \frac{1}{N}\sum_{i=1}^{N}(\hat y_i - y_i)^{2}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\mathrm{RMSE} \;=\; \sqrt{\mathrm{MSE}}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\frac{\partial L}{\partial \hat y_i} \;=\; \frac{2}{N}\,(\hat y_i - y_i)`} />
      </MathBox>
    </section>
  );
}

/* =============================== interactive prediction =============================== */

function InteractiveResidualSection() {
  const [y, setY] = useState(2.0);
  const [yHat, setYHat] = useState(1.0);

  const r = yHat - y;
  const loss = squaredError(r);
  const grad = squaredErrorDeriv(r);

  const W = 540;
  const H = 220;
  const PAD = { l: 36, r: 14, t: 16, b: 28 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;
  const axisMin = -5;
  const axisMax = 5;
  const sx = (v: number) => PAD.l + ((v - axisMin) / (axisMax - axisMin)) * innerW;
  const sy = (v: number) => H - PAD.b - ((v - axisMin) / (axisMax - axisMin)) * innerH;

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Single prediction · drag y and ŷ
        </span>
        <BlockMath math={String.raw`L \;=\; (\hat y - y)^{2}`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-5">
        {/* number-line viz */}
        <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
          <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold px-1 pb-1">
            Residual r = ŷ − y
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
            {/* axes */}
            <line x1={PAD.l} y1={sy(0)} x2={W - PAD.r} y2={sy(0)} stroke={C_AXIS} />
            <line x1={sx(0)} y1={PAD.t} x2={sx(0)} y2={H - PAD.b} stroke={C_AXIS} />
            {[-4, -2, 0, 2, 4].map((v) => (
              <g key={v}>
                <text
                  x={sx(v)}
                  y={sy(0) + 14}
                  textAnchor="middle"
                  fontSize={9}
                  fill="#7a8094"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {v}
                </text>
              </g>
            ))}
            {/* residual segment */}
            <line
              x1={sx(y)}
              y1={sy(0)}
              x2={sx(yHat)}
              y2={sy(0)}
              stroke={C_LOSS}
              strokeWidth={3}
              strokeOpacity={0.6}
            />
            {/* squared-error visualization: literal square hovering above */}
            <SquareViz y={y} yHat={yHat} sx={sx} sy={sy} />
            {/* y marker */}
            <g>
              <circle cx={sx(y)} cy={sy(0)} r={6} fill={C_TARGET} />
              <text
                x={sx(y)}
                y={sy(0) - 10}
                textAnchor="middle"
                fontSize={11}
                fill={C_TARGET}
                fontFamily="JetBrains Mono, monospace"
                fontWeight={700}
              >
                y = {y.toFixed(2)}
              </text>
            </g>
            {/* ŷ marker */}
            <g>
              <circle cx={sx(yHat)} cy={sy(0)} r={6} fill={C_PRED} />
              <text
                x={sx(yHat)}
                y={sy(0) + 30}
                textAnchor="middle"
                fontSize={11}
                fill={C_PRED}
                fontFamily="JetBrains Mono, monospace"
                fontWeight={700}
              >
                ŷ = {yHat.toFixed(2)}
              </text>
            </g>
          </svg>
        </div>

        <div className="flex flex-col gap-3">
          <SliderRow
            label="y (target)"
            value={y}
            onChange={setY}
            min={-4}
            max={4}
            step={0.05}
            color={C_TARGET}
          />
          <SliderRow
            label="ŷ (prediction)"
            value={yHat}
            onChange={setYHat}
            min={-4}
            max={4}
            step={0.05}
            color={C_PRED}
          />

          <div className="grid grid-cols-3 gap-2 text-[12px] font-mono pt-1">
            <Stat label="r = ŷ − y" value={r.toFixed(3)} color="#fff" />
            <Stat label="L = r²" value={loss.toFixed(3)} color={C_LOSS} />
            <Stat label="∂L/∂ŷ = 2r" value={grad.toFixed(3)} color={C_GRAD} />
          </div>

          <p className="text-[11px] text-ink-400 leading-snug">
            The shaded square has area{" "}
            <InlineMath math={String.raw`r^{2}`} /> — literally the loss.
            Doubling the residual quadruples the loss; the slope grows
            linearly, so MSE pulls hard from far away and gently when close.
          </p>
        </div>
      </div>
    </section>
  );
}

function SquareViz({
  y,
  yHat,
  sx,
  sy,
}: {
  y: number;
  yHat: number;
  sx: (v: number) => number;
  sy: (v: number) => number;
}) {
  const x1 = sx(Math.min(y, yHat));
  const x2 = sx(Math.max(y, yHat));
  const w = Math.abs(x2 - x1);
  if (w < 1) return null;
  return (
    <g>
      <rect
        x={x1}
        y={sy(0) - w}
        width={w}
        height={w}
        fill={C_LOSS}
        fillOpacity={0.18}
        stroke={C_LOSS}
        strokeOpacity={0.7}
        strokeWidth={1.2}
        strokeDasharray="3 3"
      />
    </g>
  );
}

/* =============================== loss vs residual curve =============================== */

function LossVsResidualPanel() {
  const W = 380;
  const H = 240;
  const PAD = { l: 36, r: 14, t: 14, b: 26 };
  const xMin = -3;
  const xMax = 3;
  const yMin = 0;
  const yMax = 9;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const msePath = useMemo(
    () => samplePath((r) => r * r, xMin, xMax, sx, sy, 220),
    []
  );
  const maePath = useMemo(
    () => samplePath((r) => Math.abs(r) * 1.5, xMin, xMax, sx, sy, 220),
    []
  );

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          L(r) · MSE vs MAE
        </span>
        <span className="text-[10px] font-mono text-ink-500">
          MSE curves up; MAE stays linear
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <Grid
          sx={sx}
          sy={sy}
          xMin={xMin}
          xMax={xMax}
          yTicks={[0, 2, 4, 6, 8]}
          xTicks={[-3, -2, -1, 0, 1, 2, 3]}
          W={W}
          H={H}
          PAD={PAD}
          compact
          xLabel="r"
        />
        <path d={maePath} stroke={C_MAE} strokeWidth={2} fill="none" strokeDasharray="4 4" />
        <path d={msePath} stroke={C_MSE} strokeWidth={2.4} fill="none" />
        {/* legend */}
        <g transform={`translate(${PAD.l + 8}, ${PAD.t + 6})`}>
          <rect width="84" height="32" rx="6" fill="rgba(0,0,0,0.4)" />
          <line x1={8} y1={10} x2={22} y2={10} stroke={C_MSE} strokeWidth={2.4} />
          <text x={26} y={13} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
            MSE r²
          </text>
          <line x1={8} y1={24} x2={22} y2={24} stroke={C_MAE} strokeWidth={2} strokeDasharray="4 4" />
          <text x={26} y={27} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
            1.5·|r|
          </text>
        </g>
      </svg>

      <p className="text-[11px] text-ink-400 leading-snug">
        Past <InlineMath math={String.raw`|r| \approx 1.5`} /> MSE overtakes the
        linear MAE and grows without bound — that is the source of both its{" "}
        <span className="text-emerald-300">strong gradient pull</span> and its{" "}
        <span className="text-amber-300">outlier sensitivity</span>.
      </p>
    </section>
  );
}

/* =============================== gradient panel =============================== */

function GradientPanel() {
  const W = 380;
  const H = 240;
  const PAD = { l: 36, r: 14, t: 14, b: 26 };
  const xMin = -3;
  const xMax = 3;
  const yMin = -6;
  const yMax = 6;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const gradPath = useMemo(
    () => samplePath((r) => 2 * r, xMin, xMax, sx, sy, 220),
    []
  );
  const maeGrad = useMemo(
    () =>
      samplePath((r) => 1.5 * Math.sign(r || 0.0001), xMin, xMax, sx, sy, 220),
    []
  );

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          ∂L/∂ŷ · gradient
        </span>
        <span className="text-[10px] font-mono text-ink-500">
          MSE: linear · MAE: step
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <Grid
          sx={sx}
          sy={sy}
          xMin={xMin}
          xMax={xMax}
          yTicks={[-6, -3, 0, 3, 6]}
          xTicks={[-3, -2, -1, 0, 1, 2, 3]}
          W={W}
          H={H}
          PAD={PAD}
          compact
          xLabel="r"
        />
        <line x1={sx(xMin)} y1={sy(0)} x2={sx(xMax)} y2={sy(0)} stroke="rgba(255,255,255,0.10)" />
        <path d={maeGrad} stroke={C_MAE} strokeWidth={2} fill="none" strokeDasharray="4 4" />
        <path d={gradPath} stroke={C_GRAD} strokeWidth={2.4} fill="none" />
      </svg>

      <p className="text-[11px] text-ink-400 leading-snug">
        MSE's gradient is <InlineMath math={String.raw`2r`} /> — proportional to
        the residual, so optimizers naturally <em>slow down</em> as the prediction
        improves. MAE's gradient is the constant{" "}
        <InlineMath math={String.raw`\pm 1`} /> — no slow-down near zero,
        producing the characteristic L1 "kink".
      </p>
    </section>
  );
}

/* =============================== outlier lab =============================== */

function OutlierLab() {
  const [outliers, setOutliers] = useState(0);
  const [outlierMag, setOutlierMag] = useState(6);

  const data = useMemo<ScatterPoint[]>(
    () =>
      makeScatter({
        n: 24,
        m: 1.0,
        b: 0.5,
        noise: 0.5,
        outliers,
        outlierMag,
        xMin: -3,
        xMax: 3,
        seed: 7,
      }),
    [outliers, outlierMag]
  );

  const xs = useMemo(() => data.map((p) => p.x), [data]);
  const ys = useMemo(() => data.map((p) => p.y), [data]);

  const fitMse = useMemo(() => olsFit(xs, ys), [xs, ys]);
  const fitMae = useMemo(() => l1Fit(xs, ys), [xs, ys]);

  const W = 560;
  const H = 340;
  const PAD = { l: 38, r: 14, t: 14, b: 28 };
  const xMin = -3.5;
  const xMax = 3.5;
  // dynamic y range to accommodate outliers
  const yPad = Math.max(4, outlierMag + 1);
  const yMin = -yPad;
  const yMax = yPad;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const lineMse = lineFromFit(fitMse, xMin, xMax);
  const lineMae = lineFromFit(fitMae, xMin, xMax);

  const yMsePred = useMemo(
    () => xs.map((x) => fitMse.m * x + fitMse.b),
    [xs, fitMse]
  );
  const yMaePred = useMemo(
    () => xs.map((x) => fitMae.m * x + fitMae.b),
    [xs, fitMae]
  );

  const mseMse = mse(ys, yMsePred);
  const maeMae = mae(ys, yMaePred);
  const rmseMse = rmse(ys, yMsePred);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Outlier lab · OLS (MSE) vs L1 (MAE)
        </span>
        <span className="text-[10px] font-mono text-ink-500">
          true line · y = x + 0.5
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-5">
        <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
            <Grid
              sx={sx}
              sy={sy}
              xMin={xMin}
              xMax={xMax}
              yTicks={[Math.floor(yMin), 0, Math.ceil(yMax)]}
              xTicks={[-3, -2, -1, 0, 1, 2, 3]}
              W={W}
              H={H}
              PAD={PAD}
              compact
            />
            <line x1={sx(xMin)} y1={sy(0)} x2={sx(xMax)} y2={sy(0)} stroke={C_AXIS} />
            <line x1={sx(0)} y1={PAD.t} x2={sx(0)} y2={H - PAD.b} stroke={C_AXIS} />
            {/* true line */}
            <line
              x1={sx(xMin)}
              y1={sy(xMin + 0.5)}
              x2={sx(xMax)}
              y2={sy(xMax + 0.5)}
              stroke="rgba(255,255,255,0.35)"
              strokeDasharray="4 4"
            />
            {/* MSE fit */}
            <line
              x1={sx(xMin)}
              y1={sy(lineMse.y1)}
              x2={sx(xMax)}
              y2={sy(lineMse.y2)}
              stroke={C_MSE}
              strokeWidth={2.4}
            />
            {/* MAE fit */}
            <line
              x1={sx(xMin)}
              y1={sy(lineMae.y1)}
              x2={sx(xMax)}
              y2={sy(lineMae.y2)}
              stroke={C_MAE}
              strokeWidth={2.4}
              strokeDasharray="6 4"
            />
            {/* data points */}
            {data.map((p, i) => {
              const isOutlier = Math.abs(p.y - (p.x + 0.5)) > 2.5;
              return (
                <circle
                  key={i}
                  cx={sx(p.x)}
                  cy={sy(p.y)}
                  r={4}
                  fill={isOutlier ? C_LOSS : "#e2e8f0"}
                  stroke={isOutlier ? "#fff" : "none"}
                  strokeWidth={1}
                />
              );
            })}
            {/* legend */}
            <g transform={`translate(${PAD.l + 8}, ${PAD.t + 6})`}>
              <rect width="160" height="46" rx="6" fill="rgba(0,0,0,0.4)" />
              <line x1={8} y1={12} x2={26} y2={12} stroke="rgba(255,255,255,0.35)" strokeDasharray="4 4" />
              <text x={32} y={15} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                true line
              </text>
              <line x1={8} y1={26} x2={26} y2={26} stroke={C_MSE} strokeWidth={2.4} />
              <text x={32} y={29} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                OLS (min MSE)
              </text>
              <line x1={8} y1={40} x2={26} y2={40} stroke={C_MAE} strokeWidth={2.4} strokeDasharray="6 4" />
              <text x={32} y={43} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                L1 (min MAE)
              </text>
            </g>
          </svg>
        </div>

        <div className="flex flex-col gap-3">
          <SliderRow
            label="# outliers"
            value={outliers}
            onChange={(v) => setOutliers(Math.round(v))}
            min={0}
            max={6}
            step={1}
            color={C_LOSS}
            format={(v) => v.toFixed(0)}
          />
          <SliderRow
            label="outlier magnitude"
            value={outlierMag}
            onChange={setOutlierMag}
            min={2}
            max={12}
            step={0.5}
            color={C_LOSS}
          />

          <hr className="border-ink-800/80 my-1" />

          <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
            OLS fit (minimizes MSE)
          </div>
          <div className="grid grid-cols-2 gap-2 text-[12px] font-mono">
            <Stat label="slope m" value={fitMse.m.toFixed(3)} color={C_MSE} />
            <Stat label="intercept b" value={fitMse.b.toFixed(3)} color={C_MSE} />
            <Stat label="MSE" value={mseMse.toFixed(3)} color={C_LOSS} />
            <Stat label="RMSE" value={rmseMse.toFixed(3)} color={C_LOSS} />
          </div>

          <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold pt-1">
            L1 fit (minimizes MAE)
          </div>
          <div className="grid grid-cols-2 gap-2 text-[12px] font-mono">
            <Stat label="slope m" value={fitMae.m.toFixed(3)} color={C_MAE} />
            <Stat label="intercept b" value={fitMae.b.toFixed(3)} color={C_MAE} />
            <Stat
              label="MAE"
              value={maeMae.toFixed(3)}
              color="#a7f3d0"
              hint="(of L1 fit)"
            />
          </div>

          <p className="text-[11px] text-ink-400 leading-snug pt-1">
            Crank up the outliers — the violet OLS line tilts toward them
            because each squared residual{" "}
            <InlineMath math={String.raw`r^{2}`} /> dominates the sum. The cyan
            L1 line, which optimizes <InlineMath math={String.raw`|r|`} />,
            barely budges.
          </p>
        </div>
      </div>
    </section>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="Range">
        <InlineMath math={String.raw`\mathrm{MSE} \in [0,\,\infty)`} />
      </PropCard>
      <PropCard title="Convex">
        <InlineMath math={String.raw`\nabla^{2}_{\!\hat y}\, L = 2\,I \succ 0`} />
      </PropCard>
      <PropCard title="Minimum">
        <InlineMath math={String.raw`\hat y^{*} = \mathbb{E}[y \mid x]`} />
      </PropCard>
      <PropCard title="Gaussian MLE">
        <InlineMath math={String.raw`y = \hat y + \mathcal N(0, \sigma^{2}) \;\Rightarrow\; \text{MSE} \propto -\!\log p`} />
      </PropCard>
      <PropCard title="Linear gradient">
        <InlineMath math={String.raw`\partial L / \partial \hat y_i = 2\,(\hat y_i - y_i)/N`} />
      </PropCard>
      <PropCard title="Scale">
        <InlineMath math={String.raw`y \to \alpha y \;\Rightarrow\; L \to \alpha^{2} L`} />
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
      <UsageCard title="Where used" accent="text-rose-300">
        <UsageItem head="Linear / ridge regression" body="closed-form normal equations minimize MSE." />
        <UsageItem head="Neural-net regression" body="house prices, age estimation, depth maps." />
        <UsageItem head="Diffusion training" body="ε-prediction MSE between predicted and true noise." />
        <UsageItem head="Score matching" body="Σ ‖ s_θ(x) − ∇ log p(x) ‖²." />
        <UsageItem head="Audio / image reconstruction" body="autoencoders, super-resolution, denoising." />
        <UsageItem head="RL value functions" body="TD-target MSE for V(s) and Q(s,a)." />
        <UsageItem head="PSNR / SSIM" body="PSNR = 10 log₁₀(MAX² / MSE)." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-emerald-300">
        <UsageItem
          head="Gaussian MLE"
          body={
            <>
              Maximizing likelihood under{" "}
              <InlineMath math={String.raw`\mathcal N(\hat y, \sigma^{2})`} /> ≡
              minimizing MSE.
            </>
          }
        />
        <UsageItem
          head="Smooth, convex (in ŷ)"
          body="strictly convex quadratic — one global minimum, second-order methods love it."
        />
        <UsageItem
          head="Linear gradient"
          body={
            <>
              <InlineMath math={String.raw`2(\hat y - y)`} /> — proportional to
              residual, naturally tapers near zero.
            </>
          }
        />
        <UsageItem
          head="Mean is optimal"
          body={
            <>
              <InlineMath math={String.raw`\arg\min_c \mathbb{E}[(y-c)^{2}] = \mathbb{E}[y]`} />.
            </>
          }
        />
        <UsageItem
          head="Decomposable"
          body="E[MSE] = bias² + variance + irreducible noise (Geman 1992)."
        />
        <UsageItem
          head="Works with linear algebra"
          body="OLS, ridge, kernel ridge, Kalman — all derived from quadratic loss."
        />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem
          head="Outlier-fragile"
          body={
            <>
              A single point with residual{" "}
              <InlineMath math={String.raw`r`} /> contributes{" "}
              <InlineMath math={String.raw`r^{2}`} /> — large errors dominate
              the sum.
            </>
          }
        />
        <UsageItem
          head="Scale-dependent"
          body="changes if you rescale y; compare with R², MAPE, or normalized variants."
        />
        <UsageItem
          head="Symmetric"
          body="over- and under-predictions cost equally — bad for asymmetric domains (e.g. inventory)."
        />
        <UsageItem
          head="Penalizes variance"
          body="optimal predictor is conditional mean → blurry images, smoothed audio."
        />
        <UsageItem
          head="Heavy-tailed noise"
          body="Gaussian assumption breaks → Huber / quantile / negative-log-Student are better."
        />
        <UsageItem
          head="Not a probability"
          body="raw MSE doesn't carry calibration meaning; use proper scoring rules for distributions."
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
      <span
        className="text-base font-semibold tabular-nums"
        style={color ? { color } : undefined}
      >
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
  format,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  color?: string;
  format?: (v: number) => string;
}) {
  return (
    <div className="flex items-center gap-2 font-mono text-[12px]">
      <span className="w-32 shrink-0 text-ink-400" style={color ? { color } : undefined}>
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
      <span className="w-14 text-right tabular-nums">
        {format ? format(value) : value.toFixed(2)}
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

function Grid({ sx, sy, yTicks, xTicks, W, H, PAD, compact, xLabel }: GridProps) {
  const xt = xTicks ?? [];
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
      {xLabel && (
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
      )}
    </g>
  );
}

function samplePath(
  fn: (x: number) => number,
  xMin: number,
  xMax: number,
  sx: (v: number) => number,
  sy: (v: number) => number,
  n = 220
): string {
  let d = "";
  for (let i = 0; i < n; i++) {
    const x = xMin + ((xMax - xMin) * i) / (n - 1);
    const y = fn(x);
    d += `${i === 0 ? "M" : "L"} ${sx(x).toFixed(2)} ${sy(y).toFixed(2)} `;
  }
  return d;
}

function lineFromFit(
  fit: { m: number; b: number },
  xMin: number,
  xMax: number
): { y1: number; y2: number } {
  return { y1: fit.m * xMin + fit.b, y2: fit.m * xMax + fit.b };
}

function fmt(v: number): string {
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(2).replace(/\.?0+$/, "");
}
