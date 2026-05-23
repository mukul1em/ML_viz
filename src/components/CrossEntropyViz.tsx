import { useMemo, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import { softmaxProbs } from "../lib/softmax";
import {
  bce,
  crossEntropy,
  entropy,
  klDivergence,
  normalize,
  oneHot,
} from "../lib/crossEntropy";

const C_TARGET = "#22d3ee";
const C_PRED = "#7c5cff";
const C_LOSS = "#f87171";
const C_GRAD = "#fbbf24";
const C_AXIS = "rgba(255,255,255,0.10)";
const CLASS_LABELS = ["A", "B", "C", "D"];

export default function CrossEntropyViz() {
  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <CategoricalCESection />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <BinaryCESection />
        <LossVsConfidencePanel />
      </div>

      <PropertiesGrid />
      <UsageGrid />

      <KLDecompositionSection />
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
          <span className="text-rose-300">Cross-entropy</span>{" "}
          <InlineMath math={String.raw`H(p,q)`} /> is the expected{" "}
          <em>information content</em>{" "}
          <InlineMath math={String.raw`-\log q(x)`} /> of predictions{" "}
          <InlineMath math={String.raw`q`} /> evaluated under the true distribution{" "}
          <InlineMath math={String.raw`p`} /> — i.e. the average bits / nats needed to
          encode samples from <InlineMath math={String.raw`p`} /> using a code optimal
          for <InlineMath math={String.raw`q`} />. With a one-hot label it collapses to
          the <em>negative log-likelihood</em>{" "}
          <InlineMath math={String.raw`-\log q_{y}`} />.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`H(p) \;=\; -\sum_{i} p_i \log p_i`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`H(p,q) \;=\; -\sum_{i} p_i \log q_i`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`H(p,q) \;=\; H(p) + D_{\mathrm{KL}}(p \,\|\, q)`} />
      </MathBox>
    </section>
  );
}

/* =============================== categorical CE =============================== */

function CategoricalCESection() {
  const [logits, setLogits] = useState<number[]>([2.0, 0.5, -0.5, -1.0]);
  const [target, setTarget] = useState<number>(0);

  const q = useMemo(() => softmaxProbs(logits), [logits]);
  const y = useMemo(() => oneHot(logits.length, target), [logits.length, target]);
  const loss = useMemo(() => -Math.log(Math.max(q[target], 1e-12)), [q, target]);
  const grad = useMemo(() => q.map((qi, i) => qi - y[i]), [q, y]);
  const Hq = useMemo(() => entropy(q), [q]);
  const Hpq = useMemo(() => crossEntropy(y, q), [y, q]);
  const kl = useMemo(() => klDivergence(y, q), [y, q]);

  const setLogit = (i: number, v: number) =>
    setLogits(logits.map((l, j) => (j === i ? v : l)));

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Categorical CE · softmax + CE
        </div>
        <BlockMath math={String.raw`L \;=\; -\log q_{y} \;=\; -\log\!\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right)`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-5">
        {/* Logit sliders + target picker */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] uppercase tracking-wider text-ink-400 font-bold">
              Target class y
            </span>
            <div className="flex gap-1">
              {CLASS_LABELS.map((lab, i) => (
                <button
                  key={i}
                  onClick={() => setTarget(i)}
                  className={`px-2.5 py-0.5 rounded text-[11.5px] font-mono border transition-colors ${
                    target === i
                      ? "border-cyan-400/60 text-white bg-cyan-500/15"
                      : "border-ink-700 text-ink-400 hover:text-white"
                  }`}
                >
                  {lab}
                </button>
              ))}
            </div>
          </div>

          {logits.map((v, i) => (
            <div key={i} className="flex items-center gap-2 font-mono text-[12px]">
              <span
                className="w-6 text-right shrink-0"
                style={{ color: i === target ? C_TARGET : "#7a8094" }}
              >
                z<sub>{CLASS_LABELS[i]}</sub>
              </span>
              <input
                type="range"
                min={-4}
                max={4}
                step={0.1}
                value={v}
                onChange={(e) => setLogit(i, parseFloat(e.target.value))}
                className="flex-1 accent-violet-400"
              />
              <span className="w-12 text-right tabular-nums">{v.toFixed(2)}</span>
            </div>
          ))}

          <hr className="border-ink-800/80 my-1" />
          <div className="grid grid-cols-3 gap-2 text-[12px] font-mono">
            <Stat label="H(p,q) = L" value={Hpq.toFixed(4)} color={C_LOSS} />
            <Stat label="H(p)" value="0.0000" hint="one-hot ⇒ 0" />
            <Stat label="D_KL" value={kl.toFixed(4)} color="#a78bfa" />
            <Stat label="H(q)" value={Hq.toFixed(4)} hint="model entropy" />
            <Stat label="q_y" value={q[target].toFixed(4)} color={C_TARGET} />
            <Stat label="−log q_y" value={loss.toFixed(4)} color={C_LOSS} />
          </div>
        </div>

        {/* Bars: predicted vs target + gradient */}
        <div className="flex flex-col gap-3">
          <ProbBars q={q} y={y} target={target} />
          <GradBars grad={grad} target={target} />
        </div>
      </div>
    </section>
  );
}

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

function ProbBars({
  q,
  y,
  target,
}: {
  q: number[];
  y: number[];
  target: number;
}) {
  const W = 360;
  const H = 200;
  const PAD = { l: 32, r: 12, t: 24, b: 28 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;
  const barW = innerW / q.length / 2.4;
  const groupW = innerW / q.length;

  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider px-1 pb-1">
        <span className="text-ink-400 font-bold">Probabilities</span>
        <span className="flex items-center gap-2 text-[10px] font-mono">
          <span className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: C_PRED }} />
            <span className="text-ink-300">q (model)</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: C_TARGET }} />
            <span className="text-ink-300">y (target)</span>
          </span>
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke={C_AXIS} />
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke={C_AXIS} />
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line
              x1={PAD.l}
              y1={PAD.t + (1 - g) * innerH}
              x2={W - PAD.r}
              y2={PAD.t + (1 - g) * innerH}
              stroke="rgba(255,255,255,0.04)"
            />
            <text
              x={PAD.l - 4}
              y={PAD.t + (1 - g) * innerH + 3}
              textAnchor="end"
              fontSize={9}
              fill="#7a8094"
              fontFamily="JetBrains Mono, monospace"
            >
              {g}
            </text>
          </g>
        ))}
        {q.map((qi, i) => {
          const cx = PAD.l + i * groupW + groupW / 2;
          const hq = qi * innerH;
          const hy = y[i] * innerH;
          return (
            <g key={i}>
              <rect
                x={cx - barW - 1}
                y={H - PAD.b - hq}
                width={barW}
                height={hq}
                fill={C_PRED}
                opacity={i === target ? 1 : 0.7}
              />
              <rect
                x={cx + 1}
                y={H - PAD.b - hy}
                width={barW}
                height={hy}
                fill={C_TARGET}
              />
              <text
                x={cx}
                y={H - PAD.b - hq - 4}
                textAnchor="middle"
                fontSize={9}
                fill="#cbd5e1"
                fontFamily="JetBrains Mono, monospace"
              >
                {qi.toFixed(2)}
              </text>
              <text
                x={cx}
                y={H - PAD.b + 14}
                textAnchor="middle"
                fontSize={10}
                fill={i === target ? C_TARGET : "#7a8094"}
                fontFamily="JetBrains Mono, monospace"
                fontWeight={i === target ? 700 : 400}
              >
                {CLASS_LABELS[i]}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function GradBars({ grad, target }: { grad: number[]; target: number }) {
  const W = 360;
  const H = 130;
  const PAD = { l: 32, r: 12, t: 18, b: 22 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;
  const groupW = innerW / grad.length;
  const barW = groupW * 0.55;
  const yScale = 1; // |grad| ≤ 1
  const sy = (v: number) => PAD.t + innerH / 2 - (v / yScale) * (innerH / 2);

  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider px-1 pb-1">
        <span className="text-ink-400 font-bold">Gradient · q − y</span>
        <span className="text-[10px] font-mono text-amber-300">
          identical to the softmax∘CE gradient
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        {/* zero line */}
        <line x1={PAD.l} y1={sy(0)} x2={W - PAD.r} y2={sy(0)} stroke={C_AXIS} />
        {[-0.5, 0.5].map((g) => (
          <line
            key={g}
            x1={PAD.l}
            y1={sy(g)}
            x2={W - PAD.r}
            y2={sy(g)}
            stroke="rgba(255,255,255,0.04)"
          />
        ))}
        {[-0.5, 0, 0.5].map((g) => (
          <text
            key={g}
            x={PAD.l - 4}
            y={sy(g) + 3}
            textAnchor="end"
            fontSize={9}
            fill="#7a8094"
            fontFamily="JetBrains Mono, monospace"
          >
            {g}
          </text>
        ))}
        {grad.map((g, i) => {
          const cx = PAD.l + i * groupW + groupW / 2;
          const y0 = sy(0);
          const y1 = sy(g);
          const top = Math.min(y0, y1);
          const h = Math.abs(y1 - y0);
          return (
            <g key={i}>
              <rect
                x={cx - barW / 2}
                y={top}
                width={barW}
                height={h}
                fill={g >= 0 ? C_GRAD : "#34d399"}
                opacity={i === target ? 1 : 0.85}
              />
              <text
                x={cx}
                y={g >= 0 ? top - 2 : top + h + 9}
                textAnchor="middle"
                fontSize={9}
                fill="#cbd5e1"
                fontFamily="JetBrains Mono, monospace"
              >
                {g >= 0 ? "+" : ""}
                {g.toFixed(2)}
              </text>
              <text
                x={cx}
                y={H - 6}
                textAnchor="middle"
                fontSize={10}
                fill={i === target ? C_TARGET : "#7a8094"}
                fontFamily="JetBrains Mono, monospace"
              >
                {CLASS_LABELS[i]}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* =============================== binary CE =============================== */

function BinaryCESection() {
  const [z, setZ] = useState(1);
  const [y, setY] = useState<0 | 1>(1);
  const { loss, grad, prob } = bce(z, y);

  const W = 380;
  const H = 240;
  const PAD = { l: 36, r: 14, t: 14, b: 26 };
  const xMin = -6;
  const xMax = 6;
  const yMin = 0;
  const yMax = 6;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const loss1 = useMemo(() => samplePath((zi) => bce(zi, 1).loss, xMin, xMax, sx, sy), []);
  const loss0 = useMemo(() => samplePath((zi) => bce(zi, 0).loss, xMin, xMax, sx, sy), []);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Binary CE · sigmoid + BCE
        </span>
        <div className="flex gap-1">
          {[0, 1].map((v) => (
            <button
              key={v}
              onClick={() => setY(v as 0 | 1)}
              className={`px-2.5 py-0.5 rounded text-[11.5px] font-mono border transition-colors ${
                y === v
                  ? "border-cyan-400/60 text-white bg-cyan-500/15"
                  : "border-ink-700 text-ink-400 hover:text-white"
              }`}
            >
              y = {v}
            </button>
          ))}
        </div>
      </div>

      <BlockMath math={String.raw`L \;=\; \log(1 + e^{z}) - y\,z, \quad \partial L / \partial z = \sigma(z) - y`} />

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[0, 2, 4, 6]} W={W} H={H} PAD={PAD} compact xLabel="z" />
        <path d={loss0} stroke={y === 0 ? C_LOSS : "rgba(248,113,113,0.35)"} strokeWidth={y === 0 ? 2.4 : 1.5} fill="none" />
        <path d={loss1} stroke={y === 1 ? "#34d399" : "rgba(52,211,153,0.35)"} strokeWidth={y === 1 ? 2.4 : 1.5} fill="none" />
        {/* cursor */}
        <line x1={sx(z)} y1={PAD.t} x2={sx(z)} y2={H - PAD.b} stroke="#fff" strokeOpacity={0.4} />
        <circle cx={sx(z)} cy={sy(Math.min(loss, yMax - 0.05))} r={4} fill={y === 1 ? "#34d399" : C_LOSS} stroke="#fff" strokeWidth={1.5} />
      </svg>

      <div className="flex items-center gap-2 font-mono text-[12px]">
        <span className="text-ink-400 w-8">z</span>
        <input
          type="range"
          min={-6}
          max={6}
          step={0.05}
          value={z}
          onChange={(e) => setZ(parseFloat(e.target.value))}
          className="flex-1 accent-cyan-400"
        />
        <span className="w-12 text-right tabular-nums">{z.toFixed(2)}</span>
      </div>

      <div className="grid grid-cols-3 gap-2 text-[12px] font-mono pt-1">
        <Stat label="σ(z)" value={prob.toFixed(4)} color={C_TARGET} />
        <Stat label="L" value={loss.toFixed(4)} color={C_LOSS} />
        <Stat label="∂L/∂z" value={grad.toFixed(4)} color={C_GRAD} />
      </div>
    </section>
  );
}

/* =============================== loss vs confidence curve =============================== */

function LossVsConfidencePanel() {
  const [qy, setQy] = useState(0.7);
  const W = 380;
  const H = 240;
  const PAD = { l: 36, r: 14, t: 14, b: 26 };
  const xMin = 0.01;
  const xMax = 1;
  const yMin = 0;
  const yMax = 5;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const path = useMemo(
    () => samplePath((p) => -Math.log(p), xMin, xMax, sx, sy, 200),
    []
  );
  const loss = -Math.log(qy);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Loss · −log(q<sub>y</sub>)
        </span>
        <span className="text-[10px] font-mono text-ink-500">
          per-sample NLL
        </span>
      </div>

      <BlockMath math={String.raw`L \;=\; -\log q_{y}`} />

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <Grid
          sx={sx}
          sy={sy}
          xMin={xMin}
          xMax={xMax}
          yTicks={[0, 1, 2, 3, 4, 5]}
          xTicks={[0.1, 0.3, 0.5, 0.7, 0.9, 1]}
          W={W}
          H={H}
          PAD={PAD}
          compact
          xLabel="q_y"
        />
        {[0.5, 1.0, 2.3, 4.6].map((y) => (
          <line
            key={y}
            x1={sx(xMin)}
            y1={sy(y)}
            x2={sx(xMax)}
            y2={sy(y)}
            stroke="rgba(255,255,255,0.04)"
            strokeDasharray="2 4"
          />
        ))}
        <path d={path} stroke={C_LOSS} strokeWidth={2.2} fill="none" />
        <line x1={sx(qy)} y1={PAD.t} x2={sx(qy)} y2={H - PAD.b} stroke="#fff" strokeOpacity={0.4} />
        <circle cx={sx(qy)} cy={sy(Math.min(loss, yMax - 0.05))} r={4} fill={C_LOSS} stroke="#fff" strokeWidth={1.5} />
      </svg>

      <div className="flex items-center gap-2 font-mono text-[12px]">
        <span className="text-ink-400 w-12">q_y</span>
        <input
          type="range"
          min={0.01}
          max={1}
          step={0.005}
          value={qy}
          onChange={(e) => setQy(parseFloat(e.target.value))}
          className="flex-1 accent-rose-400"
        />
        <span className="w-12 text-right tabular-nums">{qy.toFixed(3)}</span>
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        Linear in confidence near <InlineMath math={String.raw`q_y = 1`} />, but
        <span className="text-rose-300"> diverges</span> as{" "}
        <InlineMath math={String.raw`q_y \to 0`} /> — the model is heavily punished
        for confidently wrong predictions.
      </p>
    </section>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="NLL view">
        <InlineMath math={String.raw`L = -\tfrac{1}{N}\sum_n \log q_{y_n}(x_n)`} />
      </PropCard>
      <PropCard title="Non-negative">
        <InlineMath math={String.raw`H(p,q) \;\ge\; H(p) \;\ge\; 0`} />
      </PropCard>
      <PropCard title="Equality">
        <InlineMath math={String.raw`H(p,q) = H(p) \iff q = p`} />
      </PropCard>
      <PropCard title="Softmax + CE gradient">
        <InlineMath math={String.raw`\nabla_{\!z}\, L \;=\; q - y`} />
      </PropCard>
      <PropCard title="Sigmoid + BCE gradient">
        <InlineMath math={String.raw`\nabla_{\!z}\, L \;=\; \sigma(z) - y`} />
      </PropCard>
      <PropCard title="Proper scoring">
        <InlineMath math={String.raw`\arg\min_q \mathbb{E}_{p}[-\log q] \;=\; p`} />
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
        <UsageItem head="Classification" body="image / text / audio — categorical CE on softmax logits." />
        <UsageItem head="Language modeling" body="next-token prediction — sum of per-token CEs." />
        <UsageItem head="Binary / multi-label" body="per-class sigmoid + BCE (BCEWithLogits)." />
        <UsageItem head="Distillation" body="soft-target CE between teacher q and student p." />
        <UsageItem head="GAN training" body="discriminator objective is two BCEs (real / fake)." />
        <UsageItem head="VAE ELBO" body="reconstruction term is CE + KL regularizer." />
        <UsageItem head="Diffusion VLB" body="variational bound includes CE-like terms." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-emerald-300">
        <UsageItem
          head="MLE"
          body={
            <>
              Minimizing CE <InlineMath math={String.raw`=`} /> maximizing
              log-likelihood under categorical / Bernoulli model.
            </>
          }
        />
        <UsageItem
          head="Clean gradient"
          body={
            <>
              softmax + CE → <InlineMath math={String.raw`q - y`} />; sigmoid + BCE →{" "}
              <InlineMath math={String.raw`\sigma(z) - y`} />. No vanishing
              softmax-derivative.
            </>
          }
        />
        <UsageItem
          head="Proper scoring rule"
          body="optimum is achieved iff q matches the true distribution p."
        />
        <UsageItem
          head="Information-theoretic"
          body="counts bits / nats; directly comparable across models, datasets."
        />
        <UsageItem
          head="Punishes overconfidence"
          body={
            <>
              <InlineMath math={String.raw`-\log q_y \to \infty`} /> as{" "}
              <InlineMath math={String.raw`q_y \to 0`} />.
            </>
          }
        />
        <UsageItem
          head="Calibration"
          body="trained-to-CE classifiers tend to be (approximately) well-calibrated."
        />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem
          head="Unbounded loss"
          body="confidently wrong example can dominate the gradient — outliers hurt."
        />
        <UsageItem
          head="Label-noise fragile"
          body={
            <>
              Mislabeled <InlineMath math={String.raw`y`} /> still demands{" "}
              <InlineMath math={String.raw`q_y \to 1`} /> — use label smoothing or
              symmetric losses.
            </>
          }
        />
        <UsageItem
          head="Class imbalance"
          body="rare classes get drowned — weight, focal loss, or balanced sampling."
        />
        <UsageItem
          head="Needs stable log/exp"
          body={
            <>
              Compute via <InlineMath math={String.raw`\mathrm{logsumexp}`} /> /{" "}
              <InlineMath math={String.raw`\mathrm{softplus}`} />, never raw{" "}
              <InlineMath math={String.raw`\log(\mathrm{softmax}(\cdot))`} />.
            </>
          }
        />
        <UsageItem
          head="Modern miscalibration"
          body="deep nets over-fit CE → overconfident; temperature scaling restores calibration."
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

/* =============================== KL decomposition =============================== */

function KLDecompositionSection() {
  const [pRaw, setPRaw] = useState<number[]>([4, 3, 2, 1]);
  const [qRaw, setQRaw] = useState<number[]>([3, 3, 3, 1]);

  const p = useMemo(() => normalize(pRaw), [pRaw]);
  const q = useMemo(() => normalize(qRaw), [qRaw]);

  const Hp = useMemo(() => entropy(p), [p]);
  const Hpq = useMemo(() => crossEntropy(p, q), [p, q]);
  const kl = useMemo(() => klDivergence(p, q), [p, q]);
  const klRev = useMemo(() => klDivergence(q, p), [p, q]);

  const setPi = (i: number, v: number) => setPRaw(pRaw.map((x, j) => (j === i ? v : x)));
  const setQi = (i: number, v: number) => setQRaw(qRaw.map((x, j) => (j === i ? v : x)));

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          KL decomposition · H(p, q) = H(p) + D_KL(p ‖ q)
        </span>
        <BlockMath math={String.raw`D_{\mathrm{KL}}(p \,\|\, q) \;=\; \sum_i p_i \log \tfrac{p_i}{q_i}`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-5">
        <div className="flex flex-col gap-3">
          <DistributionEditor title="p (true)" color={C_TARGET} values={pRaw} onChange={setPi} probs={p} />
          <DistributionEditor title="q (model)" color={C_PRED} values={qRaw} onChange={setQi} probs={q} />
        </div>

        <div className="flex flex-col gap-3">
          <StackedBar Hp={Hp} kl={kl} Hpq={Hpq} />
          <div className="grid grid-cols-2 gap-2 text-[12px] font-mono">
            <Stat label="H(p)" value={Hp.toFixed(4)} color={C_TARGET} />
            <Stat label="H(p, q)" value={Hpq.toFixed(4)} color={C_LOSS} />
            <Stat label="D_KL(p ‖ q)" value={kl.toFixed(4)} color="#a78bfa" />
            <Stat label="D_KL(q ‖ p)" value={klRev.toFixed(4)} color="#a78bfa" hint="≠ D_KL(p‖q)" />
          </div>
          <p className="text-[11px] text-ink-400 leading-snug">
            Cross-entropy decomposes into the irreducible entropy of{" "}
            <InlineMath math={String.raw`p`} /> plus the KL gap. Training pushes{" "}
            <InlineMath math={String.raw`q \to p`} />, driving the KL term to zero.
            Note: <InlineMath math={String.raw`D_{\mathrm{KL}}`} /> is{" "}
            <span className="text-amber-300">not symmetric</span>.
          </p>
        </div>
      </div>
    </section>
  );
}

function DistributionEditor({
  title,
  color,
  values,
  onChange,
  probs,
}: {
  title: string;
  color: string;
  values: number[];
  onChange: (i: number, v: number) => void;
  probs: number[];
}) {
  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider font-bold" style={{ color }}>
          {title}
        </span>
        <span className="text-[10px] font-mono text-ink-500">Σ = 1</span>
      </div>
      {values.map((v, i) => (
        <div key={i} className="flex items-center gap-2 font-mono text-[11.5px]">
          <span className="w-4 text-right text-ink-400">{CLASS_LABELS[i]}</span>
          <input
            type="range"
            min={0}
            max={10}
            step={0.1}
            value={v}
            onChange={(e) => onChange(i, parseFloat(e.target.value))}
            className="flex-1"
            style={{ accentColor: color }}
          />
          <span className="w-14 text-right tabular-nums" style={{ color }}>
            {probs[i].toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}

function StackedBar({ Hp, kl, Hpq }: { Hp: number; kl: number; Hpq: number }) {
  const W = 360;
  const H = 110;
  const PAD = { l: 36, r: 12, t: 24, b: 22 };
  const innerW = W - PAD.l - PAD.r;
  const total = Math.max(Hpq, 0.001);
  const wHp = (Hp / total) * innerW;
  const wKl = (kl / total) * innerW;
  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2 flex flex-col gap-1">
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold px-1">
        H(p, q) = H(p) + D<sub>KL</sub>(p ‖ q)
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        {/* axis labels */}
        <text x={PAD.l} y={PAD.t - 6} fontSize={10} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
          0
        </text>
        <text x={W - PAD.r} y={PAD.t - 6} fontSize={10} fill="#7a8094" textAnchor="end" fontFamily="JetBrains Mono, monospace">
          {Hpq.toFixed(3)}
        </text>
        {/* H(p) segment */}
        <rect x={PAD.l} y={PAD.t} width={wHp} height={H - PAD.t - PAD.b} fill={C_TARGET} />
        <text
          x={PAD.l + wHp / 2}
          y={PAD.t + (H - PAD.t - PAD.b) / 2 + 4}
          textAnchor="middle"
          fontSize={11}
          fill="#0b0d12"
          fontFamily="JetBrains Mono, monospace"
          fontWeight={700}
        >
          H(p) = {Hp.toFixed(3)}
        </text>
        {/* KL segment */}
        <rect x={PAD.l + wHp} y={PAD.t} width={wKl} height={H - PAD.t - PAD.b} fill="#a78bfa" />
        {wKl > 60 && (
          <text
            x={PAD.l + wHp + wKl / 2}
            y={PAD.t + (H - PAD.t - PAD.b) / 2 + 4}
            textAnchor="middle"
            fontSize={11}
            fill="#0b0d12"
            fontFamily="JetBrains Mono, monospace"
            fontWeight={700}
          >
            KL = {kl.toFixed(3)}
          </text>
        )}
        {/* total bar */}
        <line x1={PAD.l} y1={H - PAD.b + 6} x2={W - PAD.r} y2={H - PAD.b + 6} stroke="#7a8094" />
        <text
          x={W / 2}
          y={H - 4}
          textAnchor="middle"
          fontSize={10}
          fill="#cbd5e1"
          fontFamily="JetBrains Mono, monospace"
        >
          H(p, q) = {Hpq.toFixed(3)}
        </text>
      </svg>
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
  const defaultXTicks = niceTicks(xMin, xMax);
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

function niceTicks(min: number, max: number): number[] {
  const span = max - min;
  const step = span <= 2 ? 0.5 : span <= 6 ? 1 : 2;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(3));
  return out;
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

function fmt(v: number): string {
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(2).replace(/\.?0+$/, "");
}
