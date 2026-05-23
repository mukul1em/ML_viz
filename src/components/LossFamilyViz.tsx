import { useMemo, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  CLASSIFICATION_VARIANTS,
  FOCAL_VARIANTS,
  REGRESSION_VARIANTS,
  focalLoss,
  huberLoss,
  outlierSweep,
  sample,
  type LossVariant,
} from "../lib/lossFamily";

const C_AXIS = "rgba(255,255,255,0.10)";

const REG_DEFAULT = ["mse", "mae", "huber"];
const CLS_DEFAULT = ["hinge", "logistic"];

export default function LossFamilyViz() {
  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <RegressionFamilySection />
      <ClassificationFamilySection />
      <FocalSection />
      <HuberDeltaSection />
      <OutlierRobustnessSection />

      <UsageGrid />
      <CheatTable />
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
          The <span className="text-violet-300">"losses & friends"</span> family —{" "}
          <em>MAE</em>, <em>Huber</em>, <em>Smooth-L1</em>, <em>Log-cosh</em>,{" "}
          <em>Hinge</em>, <em>Squared-hinge</em>, <em>Logistic</em>,{" "}
          <em>Exponential</em>, and <em>Focal</em> — the robust and
          margin-based alternatives to MSE / cross-entropy. Each trades a piece of
          MSE-style "punish-far-errors-hard" behavior for outlier robustness,
          calibration, sparsity, or hard-example focus.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`\text{Huber}_{\delta}(r)=\begin{cases}\tfrac12 r^{2} & |r|\le\delta\\\delta(|r|-\tfrac{\delta}{2}) & |r|>\delta\end{cases}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\text{Hinge}(m) \;=\; \max(0,\,1-m)`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\text{Focal}(p) \;=\; -\alpha\,(1-p)^{\gamma}\log p`} />
      </MathBox>
    </section>
  );
}

/* =============================== regression family =============================== */

function RegressionFamilySection() {
  const [active, setActive] = useState<Set<string>>(new Set(REG_DEFAULT));
  const visible = REGRESSION_VARIANTS.filter((v) => active.has(v.id));

  const toggle = (id: string) => {
    const next = new Set(active);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    if (next.size === 0) next.add("mse");
    setActive(next);
  };

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Regression losses · L(r) where r = ŷ − y
        </span>
        <span className="text-[10px] font-mono text-ink-500">
          Used in regression / score-matching / diffusion training
        </span>
      </div>

      <ToggleBar variants={REGRESSION_VARIANTS} active={active} onToggle={toggle} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <FamilyPlot
          visible={visible}
          xMin={-3}
          xMax={3}
          yMin={0}
          yMax={5}
          xLabel="r"
          mode="loss"
          yTicks={[0, 1, 2, 3, 4, 5]}
        />
        <FamilyPlot
          visible={visible}
          xMin={-3}
          xMax={3}
          yMin={-3}
          yMax={3}
          xLabel="r"
          mode="deriv"
          yTicks={[-3, -1.5, 0, 1.5, 3]}
          subtitle="∂L/∂r"
        />
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        MSE blows up like <InlineMath math={String.raw`r^{2}`} />; MAE stays
        linear; Huber and Log-cosh interpolate — quadratic near zero (smooth
        gradient like MSE) but linear in the tails (robust like MAE).
      </p>
    </section>
  );
}

/* =============================== classification family =============================== */

function ClassificationFamilySection() {
  const [active, setActive] = useState<Set<string>>(new Set(CLS_DEFAULT));
  const visible = CLASSIFICATION_VARIANTS.filter((v) => active.has(v.id));

  const toggle = (id: string) => {
    const next = new Set(active);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    if (next.size === 0) next.add("hinge");
    setActive(next);
  };

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Margin losses · L(m) where m = y · z, y ∈ {"{−1,+1}"}
        </span>
        <span className="text-[10px] font-mono text-ink-500">
          Used in SVMs · perceptrons · LR (margin form)
        </span>
      </div>

      <ToggleBar variants={CLASSIFICATION_VARIANTS} active={active} onToggle={toggle} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <FamilyPlot
          visible={visible}
          xMin={-2}
          xMax={3}
          yMin={0}
          yMax={5}
          xLabel="m"
          mode="loss"
          yTicks={[0, 1, 2, 3, 4, 5]}
          xTicks={[-2, -1, 0, 1, 2, 3]}
          verticalRefs={[{ x: 1, label: "m=1 boundary" }]}
        />
        <FamilyPlot
          visible={visible}
          xMin={-2}
          xMax={3}
          yMin={-3}
          yMax={1}
          xLabel="m"
          mode="deriv"
          yTicks={[-3, -2, -1, 0, 1]}
          subtitle="∂L/∂m"
          xTicks={[-2, -1, 0, 1, 2, 3]}
        />
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        Hinge is zero past <InlineMath math={String.raw`m=1`} /> — once the
        margin is satisfied, the example contributes no gradient (the source of
        SVM sparsity). Logistic decays smoothly to zero; Exponential punishes
        miscategorization the hardest.
      </p>
    </section>
  );
}

/* =============================== focal =============================== */

function FocalSection() {
  const [alpha, setAlpha] = useState(1);

  const W = 720;
  const H = 320;
  const PAD = { l: 36, r: 14, t: 16, b: 30 };
  const xMin = 0.01;
  const xMax = 1;
  const yMin = 0;
  const yMax = 5;
  const sx = (v: number) =>
    PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) =>
    H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const paths = useMemo(
    () =>
      FOCAL_VARIANTS.map((v) => ({
        v,
        d: buildPath(
          sample((p) => Math.min(focalLoss(p, v.gamma, alpha), yMax), xMin, xMax, 240),
          sx,
          sy
        ),
      })),
    [alpha]
  );

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Focal loss · −α (1−p)<sup>γ</sup> log p
        </span>
        <BlockMath math={String.raw`L_{\text{focal}}(p) \;=\; -\alpha\,(1-p)^{\gamma}\,\log p`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_240px] gap-4">
        <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
            <Grid
              sx={sx}
              sy={sy}
              xMin={xMin}
              xMax={xMax}
              yTicks={[0, 1, 2, 3, 4, 5]}
              xTicks={[0.1, 0.3, 0.5, 0.7, 0.9]}
              W={W}
              H={H}
              PAD={PAD}
              xLabel="p (predicted prob. of true class)"
            />
            {paths.map(({ v, d }) => (
              <path key={v.id} d={d} stroke={v.color} strokeWidth={2.2} fill="none" />
            ))}
            <g transform={`translate(${W - PAD.r - 138}, ${PAD.t + 6})`}>
              <rect width="132" height={FOCAL_VARIANTS.length * 14 + 8} rx={6} fill="rgba(0,0,0,0.45)" />
              {FOCAL_VARIANTS.map((v, i) => (
                <g key={v.id} transform={`translate(8, ${12 + i * 14})`}>
                  <line x1={0} y1={0} x2={16} y2={0} stroke={v.color} strokeWidth={2.4} />
                  <text x={22} y={3} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                    {v.name}
                  </text>
                </g>
              ))}
            </g>
          </svg>
        </div>

        <div className="flex flex-col gap-3">
          <SliderRow
            label="α (weight)"
            value={alpha}
            onChange={setAlpha}
            min={0.1}
            max={4}
            step={0.05}
            color="#fbbf24"
          />
          <div className="text-[11px] text-ink-400 leading-snug">
            <p>
              When <InlineMath math={String.raw`\gamma = 0`} /> focal collapses
              to cross-entropy. As <InlineMath math={String.raw`\gamma`} />{" "}
              grows, well-classified examples (high{" "}
              <InlineMath math={String.raw`p`} />) get exponentially{" "}
              <em>down-weighted</em> — the loss focuses on the hard cases.
            </p>
            <p className="mt-2">
              <span className="text-emerald-300">Used in:</span> dense object
              detection (RetinaNet), highly imbalanced classification, hard-mining.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

/* =============================== Huber δ =============================== */

function HuberDeltaSection() {
  const [delta, setDelta] = useState(1);

  const W = 540;
  const H = 280;
  const PAD = { l: 36, r: 14, t: 14, b: 28 };
  const xMin = -3;
  const xMax = 3;
  const yMin = 0;
  const yMax = 5;
  const sx = (v: number) =>
    PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) =>
    H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const mseD = useMemo(() => buildPath(sample((r) => Math.min(r * r, yMax), xMin, xMax, 220), sx, sy), []);
  const maeD = useMemo(() => buildPath(sample((r) => Math.abs(r), xMin, xMax, 220), sx, sy), []);
  const huberD = useMemo(
    () => buildPath(sample((r) => Math.min(huberLoss(r, delta), yMax), xMin, xMax, 220), sx, sy),
    [delta]
  );

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Huber δ → MSE ⇔ MAE knob
        </span>
        <BlockMath math={String.raw`\delta \to 0 \Rightarrow |r|,\quad \delta \to \infty \Rightarrow \tfrac{1}{2}r^{2}`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_220px] gap-4">
        <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
            <Grid
              sx={sx}
              sy={sy}
              xMin={xMin}
              xMax={xMax}
              yTicks={[0, 1, 2, 3, 4, 5]}
              xTicks={[-3, -2, -1, 0, 1, 2, 3]}
              W={W}
              H={H}
              PAD={PAD}
              xLabel="r"
            />
            <path d={mseD} stroke="#7c5cff" strokeWidth={1.6} strokeDasharray="4 4" fill="none" />
            <path d={maeD} stroke="#22d3ee" strokeWidth={1.6} strokeDasharray="4 4" fill="none" />
            <path d={huberD} stroke="#34d399" strokeWidth={2.6} fill="none" />
            {/* δ markers */}
            <line x1={sx(delta)} y1={PAD.t} x2={sx(delta)} y2={H - PAD.b} stroke="rgba(52,211,153,0.4)" strokeDasharray="2 4" />
            <line x1={sx(-delta)} y1={PAD.t} x2={sx(-delta)} y2={H - PAD.b} stroke="rgba(52,211,153,0.4)" strokeDasharray="2 4" />
            <text x={sx(delta)} y={PAD.t + 10} fontSize={10} fill="#6ee7b7" textAnchor="middle" fontFamily="JetBrains Mono, monospace">
              +δ
            </text>
            <text x={sx(-delta)} y={PAD.t + 10} fontSize={10} fill="#6ee7b7" textAnchor="middle" fontFamily="JetBrains Mono, monospace">
              −δ
            </text>
            {/* legend */}
            <g transform={`translate(${PAD.l + 8}, ${PAD.t + 6})`}>
              <rect width="118" height="46" rx={6} fill="rgba(0,0,0,0.4)" />
              <line x1={8} y1={12} x2={26} y2={12} stroke="#7c5cff" strokeDasharray="4 4" />
              <text x={32} y={15} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                MSE (r²)
              </text>
              <line x1={8} y1={26} x2={26} y2={26} stroke="#22d3ee" strokeDasharray="4 4" />
              <text x={32} y={29} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                MAE (|r|)
              </text>
              <line x1={8} y1={40} x2={26} y2={40} stroke="#34d399" strokeWidth={2} />
              <text x={32} y={43} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                Huber
              </text>
            </g>
          </svg>
        </div>

        <div className="flex flex-col gap-3">
          <SliderRow
            label="δ (breakpoint)"
            value={delta}
            onChange={setDelta}
            min={0.05}
            max={2.5}
            step={0.05}
            color="#34d399"
          />
          <p className="text-[11px] text-ink-400 leading-snug">
            Inside <InlineMath math={String.raw`|r|\le\delta`} /> Huber is
            quadratic — smooth gradient, MSE-like pull. Outside, it grows
            linearly — outlier-robust like MAE.
          </p>
        </div>
      </div>
    </section>
  );
}

/* =============================== outlier robustness =============================== */

function OutlierRobustnessSection() {
  // 24 "clean" residuals from a small Gaussian-ish sample, then sweep an outlier.
  const baseResiduals = useMemo(() => {
    const r: number[] = [];
    for (let i = 0; i < 24; i++) {
      const t = (i / 23 - 0.5) * 2;
      r.push(0.4 * Math.sin(i * 0.9) + 0.15 * Math.cos(i * 0.3) + t * 0.05);
    }
    return r;
  }, []);

  const sweepRange = useMemo(() => {
    const out: number[] = [];
    for (let i = 0; i < 50; i++) out.push(i * 0.4);
    return out;
  }, []);

  const families = useMemo(
    () => REGRESSION_VARIANTS.filter((v) => v.id !== "smoothl1"),
    []
  );
  const data = useMemo(
    () => outlierSweep(families, baseResiduals, sweepRange),
    [families, baseResiduals, sweepRange]
  );

  const W = 720;
  const H = 320;
  const PAD = { l: 44, r: 14, t: 14, b: 30 };
  const xMin = 0;
  const xMax = Math.max(...sweepRange);
  const yMax = Math.max(
    1e-3,
    ...data.flatMap((d) => d.curve.map((p) => Math.min(p.y, 200)))
  );
  const yMin = 0;
  const sx = (v: number) =>
    PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) =>
    H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Robustness sweep · how a single outlier inflates total loss
        </span>
        <span className="text-[10px] font-mono text-ink-500">
          x = outlier residual magnitude, y = Σ L
        </span>
      </div>

      <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
          <Grid
            sx={sx}
            sy={sy}
            xMin={xMin}
            xMax={xMax}
            yTicks={niceYTicks(yMin, yMax)}
            xTicks={[0, 4, 8, 12, 16, 20]}
            W={W}
            H={H}
            PAD={PAD}
            xLabel="|r_outlier|"
          />
          {data.map(({ variant, curve }) => {
            const d = curve
              .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x).toFixed(2)} ${sy(Math.min(p.y, yMax)).toFixed(2)}`)
              .join(" ");
            return <path key={variant.id} d={d} stroke={variant.color} strokeWidth={2.2} fill="none" />;
          })}
          <g transform={`translate(${PAD.l + 8}, ${PAD.t + 6})`}>
            <rect width="118" height={data.length * 14 + 8} rx={6} fill="rgba(0,0,0,0.45)" />
            {data.map(({ variant }, i) => (
              <g key={variant.id} transform={`translate(8, ${12 + i * 14})`}>
                <line x1={0} y1={0} x2={16} y2={0} stroke={variant.color} strokeWidth={2.4} />
                <text x={22} y={3} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                  {variant.name}
                </text>
              </g>
            ))}
          </g>
        </svg>
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        MSE's parabola explodes; MAE and Huber stay bounded — that's the
        textbook robust-statistics intuition in a single picture. Picking your
        loss <em>is</em> a model-of-noise assumption.
      </p>
    </section>
  );
}

/* =============================== usage =============================== */

function UsageGrid() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <UsageCard title="Where used" accent="text-violet-300">
        <UsageItem head="MAE / L1" body="median regression, recommendation, image-to-image tasks." />
        <UsageItem head="Huber / Smooth-L1" body="bounding-box regression in Fast/Faster R-CNN, robust Kalman." />
        <UsageItem head="Log-cosh" body="financial time-series, twice-differentiable MAE substitute." />
        <UsageItem head="Hinge" body="classical SVMs, max-margin classifiers." />
        <UsageItem head="Squared hinge" body="L2-SVMs, gives smooth gradient at the kink." />
        <UsageItem head="Focal" body="RetinaNet, dense detection, severe class imbalance." />
        <UsageItem head="Exponential" body="AdaBoost, exponential family probabilistic boosting." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-cyan-300">
        <UsageItem
          head="Outlier robustness"
          body={
            <>
              MAE / Huber have bounded derivative —{" "}
              <InlineMath math={String.raw`|\partial L/\partial r| \le 1`} /> or{" "}
              <InlineMath math={String.raw`\delta`} />.
            </>
          }
        />
        <UsageItem
          head="Margin = generalization"
          body={
            <>
              Hinge forces a non-zero gap{" "}
              <InlineMath math={String.raw`m \ge 1`} /> → bounded generalization
              error.
            </>
          }
        />
        <UsageItem
          head="Sparse support"
          body="hinge gives zero gradient past the margin — only support vectors matter."
        />
        <UsageItem
          head="Hard-example focus"
          body={
            <>
              Focal down-weights easy ones by{" "}
              <InlineMath math={String.raw`(1-p)^{\gamma}`} /> — easy negatives
              don't drown the rare positives.
            </>
          }
        />
        <UsageItem
          head="Smooth optimization"
          body="Log-cosh, smooth-L1 give twice-differentiable surfaces — second-order methods work."
        />
        <UsageItem
          head="Probabilistic interpretations"
          body="MAE ↔ Laplace MLE, Huber ↔ contaminated normal, hinge ↔ no-prob (non-proper)."
        />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem
          head="MAE non-smooth"
          body="kink at 0 → constant slope kills near-zero convergence; subgradient required."
        />
        <UsageItem
          head="Huber δ to tune"
          body="picks a scale; wrong δ → behaves like MSE or MAE in disguise."
        />
        <UsageItem
          head="Hinge ≠ probabilistic"
          body="not a proper scoring rule; can't directly extract calibrated class probabilities."
        />
        <UsageItem
          head="Focal hyperparameters"
          body={<>γ and α interact; common γ=2, α=0.25.</>}
        />
        <UsageItem
          head="Exponential explodes"
          body="extremely sensitive to label noise — AdaBoost famously overfits noisy labels."
        />
        <UsageItem
          head="Multi-class hinge clunky"
          body="needs Crammer–Singer or one-vs-rest formulations; CE is usually simpler."
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

/* =============================== cheatsheet =============================== */

function CheatTable() {
  const rows = [
    ...REGRESSION_VARIANTS.map((v) => ({ ...v, axis: "r" })),
    ...CLASSIFICATION_VARIANTS.map((v) => ({ ...v, axis: "m" })),
    {
      id: "focal",
      name: "Focal",
      color: "#fbbf24",
      formula: String.raw`-\alpha (1-p)^{\gamma}\log p`,
      derivFormula: String.raw`\alpha\big[\gamma(1-p)^{\gamma-1}\log p - (1-p)^{\gamma}/p\big]`,
      axis: "p",
    },
  ];
  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 overflow-hidden">
      <div className="px-4 py-2 border-b border-ink-800/80 text-[10px] uppercase tracking-wider text-ink-400 font-bold">
        Loss cheatsheet
      </div>
      <div className="grid grid-cols-[auto_60px_1fr_1fr] text-sm">
        <div className="contents text-[10px] uppercase tracking-wider text-ink-500 font-bold">
          <div className="px-4 py-2 border-b border-ink-800/60">Name</div>
          <div className="px-4 py-2 border-b border-ink-800/60">arg</div>
          <div className="px-4 py-2 border-b border-ink-800/60">L</div>
          <div className="px-4 py-2 border-b border-ink-800/60">dL</div>
        </div>
        {rows.map((v) => (
          <div key={v.id} className="contents">
            <div className="px-4 py-3 border-b border-ink-800/40 flex items-center gap-2 font-mono">
              <span
                className="inline-block w-2.5 h-2.5 rounded-sm"
                style={{ background: v.color }}
              />
              <span className="text-ink-100 text-[12px]">{v.name}</span>
            </div>
            <div className="px-4 py-3 border-b border-ink-800/40 font-mono text-[12px] text-ink-300">
              {v.axis}
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

function ToggleBar({
  variants,
  active,
  onToggle,
}: {
  variants: LossVariant[];
  active: Set<string>;
  onToggle: (id: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {variants.map((v) => {
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
              style={{
                background: on ? v.color : "transparent",
                outline: `1px solid ${v.color}`,
              }}
            />
            {v.name}
          </button>
        );
      })}
    </div>
  );
}

function FamilyPlot({
  visible,
  xMin,
  xMax,
  yMin,
  yMax,
  yTicks,
  xTicks,
  xLabel,
  mode,
  subtitle,
  verticalRefs,
}: {
  visible: LossVariant[];
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  yTicks: number[];
  xTicks?: number[];
  xLabel: string;
  mode: "loss" | "deriv";
  subtitle?: string;
  verticalRefs?: { x: number; label: string }[];
}) {
  const W = 380;
  const H = 260;
  const PAD = { l: 36, r: 14, t: 16, b: 28 };
  const sx = (v: number) =>
    PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) =>
    H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const paths = useMemo(
    () =>
      visible.map((v) => ({
        v,
        d: buildPath(
          sample(mode === "loss" ? v.fn : v.deriv, xMin, xMax, 240),
          sx,
          sy
        ),
      })),
    [visible, mode]
  );

  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2 flex flex-col gap-1">
      <div className="flex items-center justify-between px-1">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          {subtitle ?? "L"}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <Grid
          sx={sx}
          sy={sy}
          xMin={xMin}
          xMax={xMax}
          yTicks={yTicks}
          xTicks={xTicks ?? defaultXTicks(xMin, xMax)}
          W={W}
          H={H}
          PAD={PAD}
          xLabel={xLabel}
          compact
        />
        <line x1={sx(xMin)} y1={sy(0)} x2={sx(xMax)} y2={sy(0)} stroke={C_AXIS} />
        {verticalRefs?.map((ref) => (
          <g key={ref.x}>
            <line
              x1={sx(ref.x)}
              y1={PAD.t}
              x2={sx(ref.x)}
              y2={H - PAD.b}
              stroke="rgba(255,255,255,0.18)"
              strokeDasharray="2 4"
            />
            <text
              x={sx(ref.x) + 4}
              y={PAD.t + 10}
              fontSize={9}
              fill="#cbd5e1"
              fontFamily="JetBrains Mono, monospace"
            >
              {ref.label}
            </text>
          </g>
        ))}
        {paths.map(({ v, d }) => (
          <path key={v.id} d={d} stroke={v.color} strokeWidth={2.2} fill="none" />
        ))}
      </svg>
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
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  color?: string;
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
      <span className="w-12 text-right tabular-nums">{value.toFixed(2)}</span>
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

function Grid({ sx, sy, xMin, xMax, yTicks, xTicks, W, H, PAD, compact, xLabel }: GridProps) {
  const xt = xTicks ?? defaultXTicks(xMin, xMax);
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

function defaultXTicks(xMin: number, xMax: number): number[] {
  const span = xMax - xMin;
  const step = span <= 2 ? 0.5 : span <= 6 ? 1 : 2;
  const out: number[] = [];
  for (let v = Math.ceil(xMin / step) * step; v <= xMax + 1e-9; v += step) {
    out.push(+v.toFixed(3));
  }
  return out;
}

function niceYTicks(min: number, max: number): number[] {
  const span = max - min;
  const step = span <= 5 ? 1 : span <= 20 ? 5 : span <= 100 ? 25 : 100;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) {
    out.push(+v.toFixed(3));
  }
  return out;
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
