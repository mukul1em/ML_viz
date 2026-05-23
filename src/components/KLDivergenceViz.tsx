import { useMemo, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  crossEntropy,
  entropy,
  klDivergence,
  normalize,
} from "../lib/crossEntropy";
import {
  gaussianPdf,
  jsDivergence,
  klContinuous,
  mixturePdf,
  modeSeekingFit,
  momentMatchingFit,
  type GMM2,
} from "../lib/klDivergence";

const C_AXIS = "rgba(255,255,255,0.10)";
const C_P = "#22d3ee";
const C_Q = "#a78bfa";
const C_FORWARD = "#34d399";
const C_REVERSE = "#fbbf24";
const C_JS = "#f472b6";
const CLASS_LABELS = ["A", "B", "C", "D"];

export default function KLDivergenceViz() {
  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <DiscreteSection />
      <ModeMeanSection />
      <AsymmetrySection />

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
          <span className="text-violet-300">Kullback–Leibler divergence</span>{" "}
          measures the extra bits/nats needed to encode samples from{" "}
          <InlineMath math={String.raw`p`} /> when the code is optimized for{" "}
          <InlineMath math={String.raw`q`} />. It's the <em>asymmetric</em>{" "}
          gap between two distributions —{" "}
          <InlineMath math={String.raw`D_{\mathrm{KL}}(p \,\|\, q) \neq D_{\mathrm{KL}}(q \,\|\, p)`} />
          {" "}— and the same quantity that drives VAE ELBOs, RLHF
          policy-regularization, and knowledge distillation.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`D_{\mathrm{KL}}(p \,\|\, q) \;=\; \sum_i p_i \log \tfrac{p_i}{q_i}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`H(p,q) \;=\; H(p) + D_{\mathrm{KL}}(p \,\|\, q)`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`D_{\mathrm{JS}}(p,q) \;=\; \tfrac12 D_{\mathrm{KL}}(p\|m) + \tfrac12 D_{\mathrm{KL}}(q\|m),\; m = \tfrac{p+q}{2}`} />
      </MathBox>
    </section>
  );
}

/* =============================== discrete =============================== */

function DiscreteSection() {
  const [pRaw, setPRaw] = useState<number[]>([4, 3, 2, 1]);
  const [qRaw, setQRaw] = useState<number[]>([1, 2, 3, 4]);

  const p = useMemo(() => normalize(pRaw), [pRaw]);
  const q = useMemo(() => normalize(qRaw), [qRaw]);

  const Hp = useMemo(() => entropy(p), [p]);
  const Hq = useMemo(() => entropy(q), [q]);
  const Hpq = useMemo(() => crossEntropy(p, q), [p, q]);
  const klFwd = useMemo(() => klDivergence(p, q), [p, q]);
  const klRev = useMemo(() => klDivergence(q, p), [p, q]);
  const js = useMemo(() => {
    const m = p.map((pi, i) => 0.5 * (pi + q[i]));
    return 0.5 * klDivergence(p, m) + 0.5 * klDivergence(q, m);
  }, [p, q]);

  const setPi = (i: number, v: number) =>
    setPRaw(pRaw.map((x, j) => (j === i ? v : x)));
  const setQi = (i: number, v: number) =>
    setQRaw(qRaw.map((x, j) => (j === i ? v : x)));

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Discrete · edit p and q, watch every divergence
        </span>
        <BlockMath math={String.raw`D_{\mathrm{KL}}(p \,\|\, q) \;=\; \sum_i p_i \log \tfrac{p_i}{q_i}`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-5">
        <div className="flex flex-col gap-3">
          <DistEditor title="p" color={C_P} values={pRaw} onChange={setPi} probs={p} />
          <DistEditor title="q" color={C_Q} values={qRaw} onChange={setQi} probs={q} />
        </div>

        <div className="flex flex-col gap-3">
          <OverlayBars p={p} q={q} />
          <div className="grid grid-cols-2 gap-2 text-[12px] font-mono">
            <Stat label="H(p)" value={Hp.toFixed(4)} color={C_P} />
            <Stat label="H(q)" value={Hq.toFixed(4)} color={C_Q} />
            <Stat label="H(p,q)" value={Hpq.toFixed(4)} color="#f87171" />
            <Stat label="D_KL(p‖q)" value={klFwd.toFixed(4)} color={C_FORWARD} hint="forward · mean-seeking" />
            <Stat label="D_KL(q‖p)" value={klRev.toFixed(4)} color={C_REVERSE} hint="reverse · mode-seeking" />
            <Stat label="D_JS(p,q)" value={js.toFixed(4)} color={C_JS} hint="symmetric · bounded" />
          </div>
        </div>
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        Forward and reverse KL move in opposite directions when{" "}
        <InlineMath math={String.raw`p \neq q`} /> — try a "spike" p with a
        flat q to see KL forward shoot up. JS is bounded by{" "}
        <InlineMath math={String.raw`\log 2 \approx 0.693`} /> and stays
        symmetric.
      </p>
    </section>
  );
}

function DistEditor({
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
        <span className="text-[11px] uppercase tracking-wider font-bold" style={{ color }}>
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

function OverlayBars({ p, q }: { p: number[]; q: number[] }) {
  const W = 360;
  const H = 200;
  const PAD = { l: 32, r: 12, t: 24, b: 28 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;
  const groupW = innerW / p.length;
  const barW = groupW / 2.4;
  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider px-1 pb-1">
        <span className="text-ink-400 font-bold">Distributions</span>
        <span className="flex items-center gap-2 text-[10px] font-mono">
          <span className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: C_P }} />
            <span className="text-ink-300">p</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: C_Q }} />
            <span className="text-ink-300">q</span>
          </span>
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke={C_AXIS} />
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke={C_AXIS} />
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line x1={PAD.l} y1={PAD.t + (1 - g) * innerH} x2={W - PAD.r} y2={PAD.t + (1 - g) * innerH} stroke="rgba(255,255,255,0.04)" />
            <text x={PAD.l - 4} y={PAD.t + (1 - g) * innerH + 3} textAnchor="end" fontSize={9} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
              {g}
            </text>
          </g>
        ))}
        {p.map((pi, i) => {
          const cx = PAD.l + i * groupW + groupW / 2;
          const hP = pi * innerH;
          const hQ = q[i] * innerH;
          return (
            <g key={i}>
              <rect x={cx - barW - 1} y={H - PAD.b - hP} width={barW} height={hP} fill={C_P} />
              <rect x={cx + 1} y={H - PAD.b - hQ} width={barW} height={hQ} fill={C_Q} />
              <text x={cx} y={H - PAD.b + 14} textAnchor="middle" fontSize={10} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
                {CLASS_LABELS[i]}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* =============================== mode vs mean =============================== */

function ModeMeanSection() {
  const [sep, setSep] = useState(3);
  const [pi, setPi] = useState(0.5);

  const target: GMM2 = useMemo(
    () => ({
      pi,
      mu1: -sep / 2,
      sigma1: 0.7,
      mu2: sep / 2,
      sigma2: 0.7,
    }),
    [sep, pi]
  );

  const xMin = -6;
  const xMax = 6;

  const mm = useMemo(() => momentMatchingFit(target), [target]);
  const ms = useMemo(() => modeSeekingFit(target, xMin, xMax), [target]);

  const pFn = (x: number) => mixturePdf(x, target);
  const qFwd = (x: number) => gaussianPdf(x, mm.mu, mm.sigma);
  const qRev = (x: number) => gaussianPdf(x, ms.mu, ms.sigma);

  const klFwd = useMemo(() => klContinuous(pFn, qFwd, xMin, xMax, 600), [target, mm]);
  const klRev = useMemo(() => klContinuous(qRev, pFn, xMin, xMax, 600), [target, ms]);
  const js = useMemo(() => jsDivergence(pFn, qFwd, xMin, xMax, 600), [target, mm]);

  const W = 720;
  const H = 320;
  const PAD = { l: 36, r: 14, t: 14, b: 28 };
  const yMax = 0.65;
  const sx = (v: number) =>
    PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) =>
    H - PAD.b - ((v - 0) / (yMax - 0)) * (H - PAD.t - PAD.b);

  const pPath = useMemo(() => buildPath(pFn, xMin, xMax, sx, sy), [target]);
  const qFwdPath = useMemo(() => buildPath(qFwd, xMin, xMax, sx, sy), [mm]);
  const qRevPath = useMemo(() => buildPath(qRev, xMin, xMax, sx, sy), [ms]);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Forward (mean-seeking) vs reverse (mode-seeking) KL
        </span>
        <BlockMath math={String.raw`q^{*}_{\text{fwd}} = \arg\min_q D_{\mathrm{KL}}(p \| q)\quad q^{*}_{\text{rev}} = \arg\min_q D_{\mathrm{KL}}(q \| p)`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_240px] gap-4">
        <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
            <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[0, 0.2, 0.4, 0.6]} xTicks={[-6, -4, -2, 0, 2, 4, 6]} W={W} H={H} PAD={PAD} xLabel="x" />
            {/* p fill */}
            <path d={pPath + ` L ${sx(xMax).toFixed(2)} ${sy(0).toFixed(2)} L ${sx(xMin).toFixed(2)} ${sy(0).toFixed(2)} Z`} fill={C_P} fillOpacity={0.12} stroke="none" />
            <path d={pPath} stroke={C_P} strokeWidth={2.4} fill="none" />
            <path d={qFwdPath} stroke={C_FORWARD} strokeWidth={2.2} fill="none" strokeDasharray="6 3" />
            <path d={qRevPath} stroke={C_REVERSE} strokeWidth={2.2} fill="none" strokeDasharray="2 3" />
            {/* legend */}
            <g transform={`translate(${PAD.l + 8}, ${PAD.t + 6})`}>
              <rect width="200" height="60" rx={6} fill="rgba(0,0,0,0.45)" />
              <line x1={8} y1={14} x2={26} y2={14} stroke={C_P} strokeWidth={2.4} />
              <text x={32} y={17} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                p (target mixture)
              </text>
              <line x1={8} y1={30} x2={26} y2={30} stroke={C_FORWARD} strokeWidth={2.2} strokeDasharray="6 3" />
              <text x={32} y={33} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                q · forward-KL (mean)
              </text>
              <line x1={8} y1={46} x2={26} y2={46} stroke={C_REVERSE} strokeWidth={2.2} strokeDasharray="2 3" />
              <text x={32} y={49} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                q · reverse-KL (mode)
              </text>
            </g>
          </svg>
        </div>

        <div className="flex flex-col gap-3">
          <SliderRow label="mode separation" value={sep} onChange={setSep} min={0.5} max={5.5} step={0.05} color={C_P} />
          <SliderRow label="mixing π (mode A)" value={pi} onChange={setPi} min={0.1} max={0.9} step={0.01} color={C_P} />

          <hr className="border-ink-800/80 my-1" />
          <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
            Best-fit single Gaussian
          </div>
          <div className="grid grid-cols-2 gap-2 text-[12px] font-mono">
            <Stat label="μ_fwd" value={mm.mu.toFixed(2)} color={C_FORWARD} />
            <Stat label="σ_fwd" value={mm.sigma.toFixed(2)} color={C_FORWARD} />
            <Stat label="μ_rev" value={ms.mu.toFixed(2)} color={C_REVERSE} />
            <Stat label="σ_rev" value={ms.sigma.toFixed(2)} color={C_REVERSE} />
            <Stat label="D_KL(p‖q_f)" value={klFwd.toFixed(3)} color={C_FORWARD} />
            <Stat label="D_KL(q_r‖p)" value={klRev.toFixed(3)} color={C_REVERSE} />
            <Stat label="D_JS" value={js.toFixed(3)} color={C_JS} />
          </div>

          <p className="text-[11px] text-ink-400 leading-snug pt-1">
            <span className="text-emerald-300">Forward KL</span> places{" "}
            <InlineMath math={String.raw`q`} /> over both modes (covers all of{" "}
            <InlineMath math={String.raw`p`} />, gets the mean right).{" "}
            <span className="text-amber-300">Reverse KL</span> collapses onto
            <em> one</em> mode — anywhere{" "}
            <InlineMath math={String.raw`p \approx 0`} /> is forbidden because{" "}
            <InlineMath math={String.raw`q \log(q/p)`} /> blows up.
          </p>
        </div>
      </div>
    </section>
  );
}

/* =============================== asymmetry: spike vs flat =============================== */

function AsymmetrySection() {
  const [center, setCenter] = useState(0);
  const [spike, setSpike] = useState(0.3);

  const xMin = -4;
  const xMax = 4;

  const pFn = (x: number) => gaussianPdf(x, center, spike);
  const qFn = (x: number) => gaussianPdf(x, 0, 1.5);

  const klPQ = useMemo(() => klContinuous(pFn, qFn, xMin, xMax, 800), [center, spike]);
  const klQP = useMemo(() => klContinuous(qFn, pFn, xMin, xMax, 800), [center, spike]);
  const js = useMemo(() => jsDivergence(pFn, qFn, xMin, xMax, 800), [center, spike]);

  const W = 540;
  const H = 240;
  const PAD = { l: 36, r: 14, t: 14, b: 28 };
  const yMax = Math.max(1.3, gaussianPdf(0, 0, spike));
  const sx = (v: number) =>
    PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) =>
    H - PAD.b - ((v - 0) / (yMax - 0)) * (H - PAD.t - PAD.b);

  const pPath = buildPath(pFn, xMin, xMax, sx, sy);
  const qPath = buildPath(qFn, xMin, xMax, sx, sy);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Asymmetry · KL(p‖q) ≠ KL(q‖p)
        </span>
        <BlockMath math={String.raw`D_{\mathrm{KL}} \;\;\text{is not a metric}`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_240px] gap-4">
        <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
            <Grid sx={sx} sy={sy} xMin={xMin} xMax={xMax} yTicks={[0, 0.5, 1, 1.5].filter((t) => t <= yMax)} xTicks={[-4, -2, 0, 2, 4]} W={W} H={H} PAD={PAD} xLabel="x" />
            <path d={qPath} stroke={C_Q} strokeWidth={2.4} fill="none" />
            <path d={pPath} stroke={C_P} strokeWidth={2.4} fill="none" />
            <g transform={`translate(${PAD.l + 8}, ${PAD.t + 6})`}>
              <rect width="160" height="34" rx={6} fill="rgba(0,0,0,0.45)" />
              <line x1={8} y1={12} x2={26} y2={12} stroke={C_P} strokeWidth={2.4} />
              <text x={32} y={15} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                p · spike at {center.toFixed(2)}
              </text>
              <line x1={8} y1={26} x2={26} y2={26} stroke={C_Q} strokeWidth={2.4} />
              <text x={32} y={29} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                q · wide N(0, 1.5²)
              </text>
            </g>
          </svg>
        </div>

        <div className="flex flex-col gap-3">
          <SliderRow label="spike center μ_p" value={center} onChange={setCenter} min={-3} max={3} step={0.05} color={C_P} />
          <SliderRow label="spike width σ_p" value={spike} onChange={setSpike} min={0.1} max={1.5} step={0.01} color={C_P} />

          <hr className="border-ink-800/80 my-1" />
          <div className="grid grid-cols-1 gap-2 text-[12px] font-mono">
            <Stat label="D_KL(p ‖ q)" value={klPQ.toFixed(3)} color={C_FORWARD} hint="cost of using q for p" />
            <Stat label="D_KL(q ‖ p)" value={klQP.toFixed(3)} color={C_REVERSE} hint="cost of using p for q" />
            <Stat label="D_JS(p, q)" value={js.toFixed(3)} color={C_JS} hint="symmetric average" />
          </div>

          <p className="text-[11px] text-ink-400 leading-snug pt-1">
            Move the spike off-center: KL(p‖q) stays small (one delta inside a
            wide q is cheap), but KL(q‖p) explodes (huge swaths of q are placed
            where p ≈ 0). That's why VAEs prefer reverse KL and EP / moment-matching
            prefers forward KL.
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
      <PropCard title="Non-negative">
        <InlineMath math={String.raw`D_{\mathrm{KL}}(p\,\|\,q) \;\ge\; 0`} />
      </PropCard>
      <PropCard title="Zero iff equal">
        <InlineMath math={String.raw`D_{\mathrm{KL}}(p\,\|\,q) = 0 \iff p = q`} />
      </PropCard>
      <PropCard title="Asymmetric">
        <InlineMath math={String.raw`D_{\mathrm{KL}}(p\,\|\,q) \neq D_{\mathrm{KL}}(q\,\|\,p)`} />
      </PropCard>
      <PropCard title="Not a metric">
        <InlineMath math={String.raw`\text{no triangle inequality, not symmetric}`} />
      </PropCard>
      <PropCard title="Information-theoretic">
        <InlineMath math={String.raw`H(p, q) - H(p) \;=\; D_{\mathrm{KL}}(p\,\|\,q)`} />
      </PropCard>
      <PropCard title="Convex (in both args)">
        <InlineMath math={String.raw`(p_1, q_1), (p_2, q_2) \to \text{jointly convex}`} />
      </PropCard>
      <PropCard title="JS is bounded">
        <InlineMath math={String.raw`0 \le D_{\mathrm{JS}}(p,q) \le \log 2`} />
      </PropCard>
      <PropCard title="Gaussian closed form">
        <InlineMath math={String.raw`D_{\mathrm{KL}}(\mathcal{N}_0 \,\|\, \mathcal{N}_1) = \tfrac12\!\left[\log\tfrac{\sigma_1^{2}}{\sigma_0^{2}} + \tfrac{\sigma_0^{2} + (\mu_0-\mu_1)^{2}}{\sigma_1^{2}} - 1\right]`} />
      </PropCard>
      <PropCard title="Reverse-KL ⇒ mode-seeking">
        <InlineMath math={String.raw`q \log(q/p) \to \infty \text{ where } p \approx 0`} />
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
        <UsageItem
          head="VAE ELBO"
          body={
            <>
              <InlineMath math={String.raw`\mathcal{L} = \mathbb{E}_q[\log p(x|z)] - D_{\mathrm{KL}}(q(z|x) \,\|\, p(z))`} />
              {" "}— reverse-KL pulls q toward the prior.
            </>
          }
        />
        <UsageItem
          head="RLHF · PPO"
          body={
            <>
              KL-penalized policy update{" "}
              <InlineMath math={String.raw`\max_\pi \mathbb{E}[r] - \beta\, D_{\mathrm{KL}}(\pi \,\|\, \pi_{\text{ref}})`} />
              .
            </>
          }
        />
        <UsageItem
          head="Knowledge distillation"
          body={<>student minimizes <InlineMath math={String.raw`D_{\mathrm{KL}}(\text{teacher} \,\|\, \text{student})`} /> at temperature T.</>}
        />
        <UsageItem
          head="Diffusion / VLB"
          body="per-step ELBO is a sum of KLs between forward and reverse Gaussians."
        />
        <UsageItem
          head="Variational inference"
          body="VI minimizes reverse KL between approximation q and posterior p(z|x)."
        />
        <UsageItem
          head="Information bottleneck"
          body={<>tradeoff <InlineMath math={String.raw`I(X;T) - \beta I(T;Y)`} /> — both terms are KLs.</>}
        />
        <UsageItem
          head="GAN (alternative formulations)"
          body="JS divergence underlies the original GAN; KL gives f-GAN family."
        />
      </UsageCard>

      <UsageCard title="Why used" accent="text-cyan-300">
        <UsageItem
          head="Information-theoretic"
          body="exact units (bits / nats) of distributional gap; comparable across models."
        />
        <UsageItem
          head="Decomposes cross-entropy"
          body={
            <>
              <InlineMath math={String.raw`H(p,q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q)`} />
              {" "}— minimizing CE = minimizing KL.
            </>
          }
        />
        <UsageItem
          head="Variational bound"
          body={
            <>
              <InlineMath math={String.raw`\log p(x) \;\ge\; \mathbb{E}_q[\log p(x,z)] - \mathbb{E}_q[\log q]`} />
              .
            </>
          }
        />
        <UsageItem
          head="Convex"
          body="convex in both p and q — well-behaved as a regularizer."
        />
        <UsageItem
          head="Closed-form for exponential families"
          body="Gaussian-to-Gaussian KL is differentiable and stable — VAE-friendly."
        />
        <UsageItem
          head="Asymmetry is a feature"
          body="forward = inclusive (cover all modes); reverse = exclusive (snap to one mode)."
        />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem
          head="Infinite when supports mismatch"
          body={
            <>
              <InlineMath math={String.raw`D_{\mathrm{KL}}(p\,\|\,q) = \infty`} />
              {" "}if <InlineMath math={String.raw`q(x) = 0, p(x) > 0`} />.
            </>
          }
        />
        <UsageItem
          head="Not symmetric"
          body="picking forward vs reverse changes the answer; both are legitimate."
        />
        <UsageItem
          head="Not a metric"
          body="no triangle inequality; for geometry use Wasserstein, Hellinger, or JS."
        />
        <UsageItem
          head="Sensitive to tails"
          body="forward KL needs q wide enough; reverse KL collapses tails."
        />
        <UsageItem
          head="Sampling estimators are biased"
          body={
            <>
              Monte-Carlo estimate of{" "}
              <InlineMath math={String.raw`\mathbb{E}_p[\log(p/q)]`} /> can have
              very high variance.
            </>
          }
        />
        <UsageItem
          head="Mode collapse"
          body="reverse-KL objectives in deep generative models famously collapse to a few modes."
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
  xTicks: number[];
  W: number;
  H: number;
  PAD: { l: number; r: number; t: number; b: number };
  xLabel?: string;
}

function Grid({ sx, sy, yTicks, xTicks, W, H, PAD, xLabel }: GridProps) {
  return (
    <g>
      <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke={C_AXIS} />
      <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke={C_AXIS} />
      {yTicks.map((v) => (
        <g key={`y-${v}`}>
          <line x1={PAD.l - 4} y1={sy(v)} x2={W - PAD.r} y2={sy(v)} stroke="rgba(255,255,255,0.04)" />
          <text x={PAD.l - 6} y={sy(v) + 3} textAnchor="end" fontSize={9} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
            {fmt(v)}
          </text>
        </g>
      ))}
      {xTicks.map((v) => (
        <g key={`x-${v}`}>
          <line x1={sx(v)} y1={H - PAD.b} x2={sx(v)} y2={H - PAD.b + 4} stroke={C_AXIS} />
          <text x={sx(v)} y={H - PAD.b + 14} textAnchor="middle" fontSize={9} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
            {fmt(v)}
          </text>
        </g>
      ))}
      {xLabel && (
        <text x={W - PAD.r} y={H - PAD.b - 4} textAnchor="end" fontSize={9} fill="#5b6478" fontFamily="JetBrains Mono, monospace">
          {xLabel}
        </text>
      )}
    </g>
  );
}

function buildPath(
  fn: (x: number) => number,
  xMin: number,
  xMax: number,
  sx: (v: number) => number,
  sy: (v: number) => number,
  n = 240
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
