import { useMemo, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  alibiSlope,
  buildAlibiMatrix,
  buildEmbeddingMatrix,
  ropeAngle,
  ropeRotate2D,
} from "../lib/positionalEncodings";

const C_AXIS = "rgba(255,255,255,0.10)";
const C_SIN = "#22d3ee";
const C_LEARNED = "#a78bfa";
const C_ROPE = "#f472b6";
const C_ALIBI = "#fbbf24";

const METHODS = ["sinusoidal", "learned", "rope", "alibi"] as const;
type Method = (typeof METHODS)[number];

const METHOD_COLOR: Record<Method, string> = {
  sinusoidal: C_SIN,
  learned: C_LEARNED,
  rope: C_ROPE,
  alibi: C_ALIBI,
};

const METHOD_LABEL: Record<Method, string> = {
  sinusoidal: "Sinusoidal",
  learned: "Learned",
  rope: "RoPE",
  alibi: "ALiBi",
};

const METHOD_FORMULA: Record<Method, string> = {
  sinusoidal: String.raw`PE_{pos, 2i} = \sin(pos / B^{2i/d})`,
  learned: String.raw`PE = \text{nn.Embedding}(L_{\max}, d)`,
  rope: String.raw`R_{\theta_i pos}\,\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}`,
  alibi: String.raw`A_{i,j} \mathrel{+}{=}\, -m_h\,|i - j|`,
};

export default function PositionalEncodingsViz() {
  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <HeatmapsSection />
      <RopeRotationSection />
      <AlibiBiasSection />
      <ExtrapolationSection />

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
          Attention is{" "}
          <span className="text-amber-300">permutation-invariant</span> — it
          treats tokens as a set. <em>Positional encodings</em> inject the order
          back in.{" "}
          <span className="text-cyan-300">Sinusoidal</span> &amp;{" "}
          <span className="text-violet-300">learned</span> add a position vector
          to embeddings; <span className="text-pink-300">RoPE</span> rotates
          queries / keys in 2D pairs;{" "}
          <span className="text-amber-300">ALiBi</span> adds a distance-penalty
          to the attention logits.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`PE_{pos, 2i} \;=\; \sin\!\left(\frac{pos}{10000^{2i/d}}\right)`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\mathrm{RoPE}: \begin{bmatrix}q'_{2i}\\q'_{2i+1}\end{bmatrix} = R(\theta_i\,pos)\,\begin{bmatrix}q_{2i}\\q_{2i+1}\end{bmatrix}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\mathrm{ALiBi}: A_{i,j} \mathrel{+}{=}\, -m_h\,(i - j)`} />
      </MathBox>
    </section>
  );
}

/* =============================== heatmaps =============================== */

function HeatmapsSection() {
  const [seqLen, setSeqLen] = useState(64);
  const [dim, setDim] = useState(64);
  const [base, setBase] = useState(10000);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Four methods · same axes
        </span>
        <span className="text-[10px] font-mono text-ink-500">
          y = position · x = dim (or key idx for ALiBi)
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
        <SliderRow label="seq_len" value={seqLen} onChange={(v) => setSeqLen(Math.round(v))} min={16} max={256} step={8} fmt={(v) => v.toFixed(0)} color={C_SIN} />
        <SliderRow label="dim (per head)" value={dim} onChange={(v) => setDim(Math.round(v))} min={16} max={128} step={4} fmt={(v) => v.toFixed(0)} color={C_SIN} />
        <SliderRow label="RoPE base" value={base} onChange={(v) => setBase(Math.round(v))} min={1000} max={100000} step={1000} fmt={(v) => v.toFixed(0)} color={C_ROPE} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {METHODS.map((m) => (
          <HeatmapTile key={m} method={m} seqLen={seqLen} dim={dim} base={base} />
        ))}
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        Sinusoidal shows clean diagonal stripes across frequency bands; learned is
        unstructured noise; RoPE's rotation produces a moiré of bands across
        feature pairs; ALiBi is the only one that lives in the attention-score
        space rather than the embedding space.
      </p>
    </section>
  );
}

function HeatmapTile({
  method,
  seqLen,
  dim,
  base,
}: {
  method: Method;
  seqLen: number;
  dim: number;
  base: number;
}) {
  const W = 220;
  const H = 200;
  const PAD = { l: 20, r: 4, t: 18, b: 14 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;

  const { matrix, vRange, xLabel, yLabel } = useMemo(() => {
    if (method === "alibi") {
      // head 0 of 8
      const m = buildAlibiMatrix(seqLen, 0, 8);
      const finite = m.flat().filter((v) => Number.isFinite(v));
      const vMin = Math.min(...finite);
      const vMax = 0;
      return { matrix: m, vRange: [vMin, vMax] as [number, number], xLabel: "key j", yLabel: "query i" };
    }
    const m = buildEmbeddingMatrix(method, seqLen, dim, base);
    return { matrix: m, vRange: [-1, 1] as [number, number], xLabel: "dim", yLabel: "pos" };
  }, [method, seqLen, dim, base]);

  const cols = matrix[0]?.length ?? 1;
  const cw = innerW / cols;
  const ch = innerH / seqLen;

  const cMin = vRange[0];
  const cMax = vRange[1];
  const colorFor = (v: number) => {
    if (!Number.isFinite(v)) return "rgba(0,0,0,0.85)";
    if (method === "alibi") {
      // 0 = bright, very negative = dark
      const t = Math.max(0, Math.min(1, (v - cMin) / Math.max(cMax - cMin, 1e-9)));
      const r = Math.round(20 + 200 * t);
      const g = Math.round(15 + 110 * t);
      const b = Math.round(40 + 50 * (1 - t));
      return `rgb(${r},${g},${b})`;
    }
    // diverging: -1 → blue, 0 → black, +1 → warm
    const t = Math.max(-1, Math.min(1, v));
    if (t >= 0) {
      const r = Math.round(40 + 200 * t);
      const g = Math.round(20 + 100 * t);
      const b = Math.round(80 - 60 * t);
      return `rgb(${r},${g},${b})`;
    } else {
      const a = -t;
      const r = Math.round(20 + 30 * a);
      const g = Math.round(40 + 80 * a);
      const b = Math.round(80 + 150 * a);
      return `rgb(${r},${g},${b})`;
    }
  };

  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
      <div className="flex items-center justify-between px-1 pb-1">
        <span
          className="text-[11px] uppercase tracking-wider font-bold"
          style={{ color: METHOD_COLOR[method] }}
        >
          {METHOD_LABEL[method]}
        </span>
        <span className="text-[9px] font-mono text-ink-500">
          {method === "alibi" ? "head 0/8" : `${seqLen}×${dim}`}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        {matrix.map((row, i) =>
          row.map((v, j) => (
            <rect
              key={`${i}-${j}`}
              x={PAD.l + j * cw}
              y={PAD.t + i * ch}
              width={cw + 0.5}
              height={ch + 0.5}
              fill={colorFor(v)}
            />
          ))
        )}
        <text x={PAD.l} y={PAD.t - 4} fontSize={8} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
          {yLabel}↓ {xLabel}→
        </text>
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke={C_AXIS} />
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke={C_AXIS} />
      </svg>
      <BlockMath math={METHOD_FORMULA[method]} />
    </div>
  );
}

/* =============================== RoPE rotation =============================== */

function RopeRotationSection() {
  const [pos, setPos] = useState(8);
  const [pair, setPair] = useState(0);
  const [dim, setDim] = useState(64);

  const angleRad = ropeAngle(pair, pos, dim);
  const baseVec: [number, number] = [1, 0];
  const rotated = ropeRotate2D(baseVec, pos, pair, dim);

  const W = 280;
  const H = 280;
  const cx = W / 2;
  const cy = H / 2;
  const R = 110;

  // Show rotated vectors for positions 0..pos
  const trail: { x: number; y: number; pos: number }[] = [];
  for (let p = 0; p <= pos; p++) {
    const [vx, vy] = ropeRotate2D(baseVec, p, pair, dim);
    trail.push({ x: cx + vx * R, y: cy - vy * R, pos: p });
  }

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          RoPE rotation · feature pair (2i, 2i+1)
        </span>
        <BlockMath math={String.raw`\theta_i = 10000^{-2i/d},\quad \text{angle} = \theta_i \cdot pos`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-4 items-start">
        <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2 flex justify-center">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
            <circle cx={cx} cy={cy} r={R} fill="none" stroke={C_AXIS} />
            <line x1={cx - R - 10} y1={cy} x2={cx + R + 10} y2={cy} stroke={C_AXIS} />
            <line x1={cx} y1={cy - R - 10} x2={cx} y2={cy + R + 10} stroke={C_AXIS} />
            {/* trail */}
            {trail.map((t, i) => (
              <g key={i}>
                <line x1={cx} y1={cy} x2={t.x} y2={t.y} stroke={C_ROPE} strokeWidth={1} strokeOpacity={i === trail.length - 1 ? 0.9 : 0.18} />
                <circle cx={t.x} cy={t.y} r={i === trail.length - 1 ? 5 : 2.4} fill={C_ROPE} fillOpacity={i === trail.length - 1 ? 1 : 0.5} />
              </g>
            ))}
            {/* labels */}
            <text x={cx + R + 6} y={cy + 3} fontSize={10} fill="#7a8094" fontFamily="JetBrains Mono, monospace">x</text>
            <text x={cx + 4} y={cy - R - 6} fontSize={10} fill="#7a8094" fontFamily="JetBrains Mono, monospace">y</text>
            <text x={trail[trail.length - 1].x + 6} y={trail[trail.length - 1].y - 4} fontSize={10} fill={C_ROPE} fontFamily="JetBrains Mono, monospace" fontWeight={700}>
              pos = {pos}
            </text>
          </svg>
        </div>

        <div className="flex flex-col gap-3">
          <SliderRow label="position" value={pos} onChange={(v) => setPos(Math.round(v))} min={0} max={32} step={1} fmt={(v) => v.toFixed(0)} color={C_ROPE} />
          <SliderRow label="pair index i" value={pair} onChange={(v) => setPair(Math.round(v))} min={0} max={dim / 2 - 1} step={1} fmt={(v) => v.toFixed(0)} color={C_ROPE} />
          <SliderRow label="head dim d" value={dim} onChange={(v) => setDim(Math.round(v))} min={16} max={128} step={2} fmt={(v) => v.toFixed(0)} color={C_ROPE} />

          <hr className="border-ink-800/80 my-1" />
          <div className="grid grid-cols-2 gap-2 text-[12px] font-mono">
            <Stat label="θ_i" value={Math.pow(10000, (-2 * pair) / dim).toExponential(2)} color={C_ROPE} hint="rotation rate" />
            <Stat label="angle" value={`${angleRad.toFixed(2)} rad`} color={C_ROPE} hint={`${((angleRad * 180) / Math.PI).toFixed(0)}°`} />
            <Stat label="rotated [x', y']" value={`[${rotated[0].toFixed(2)}, ${rotated[1].toFixed(2)}]`} color="#fff" />
          </div>

          <p className="text-[11px] text-ink-400 leading-snug">
            High-index pairs rotate <em>slowly</em> (low θ); low-index pairs
            rotate fast. Two tokens with relative distance Δ produce attention
            logits depending only on Δ — that's the relative-position property
            that makes RoPE generalize beyond training length.
          </p>
        </div>
      </div>
    </section>
  );
}

/* =============================== ALiBi bias matrix =============================== */

function AlibiBiasSection() {
  const [headIdx, setHeadIdx] = useState(0);
  const [nHeads, setNHeads] = useState(8);
  const seqLen = 24;

  const matrix = useMemo(() => buildAlibiMatrix(seqLen, headIdx, nHeads), [headIdx, nHeads]);
  const slope = alibiSlope(headIdx, nHeads);

  const W = 320;
  const H = 320;
  const PAD = { l: 28, r: 4, t: 18, b: 24 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;
  const cw = innerW / seqLen;
  const ch = innerH / seqLen;

  const finite = matrix.flat().filter((v) => Number.isFinite(v));
  const vMin = Math.min(...finite);

  const colorFor = (v: number) => {
    if (!Number.isFinite(v)) return "rgba(0,0,0,0.85)";
    const t = Math.max(0, Math.min(1, (v - vMin) / Math.max(0 - vMin, 1e-9)));
    const r = Math.round(20 + 220 * t);
    const g = Math.round(15 + 120 * t);
    const b = Math.round(40 + 60 * (1 - t));
    return `rgb(${r},${g},${b})`;
  };

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          ALiBi bias matrix · linear distance penalty
        </span>
        <BlockMath math={String.raw`m_h = 2^{-8(h+1)/H},\quad A_{i,j} \mathrel{+}{=}\, -m_h\,(i - j)`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-4 items-start">
        <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
            <text x={PAD.l} y={PAD.t - 4} fontSize={9} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
              query↓  key→
            </text>
            {matrix.map((row, i) =>
              row.map((v, j) => (
                <rect
                  key={`${i}-${j}`}
                  x={PAD.l + j * cw}
                  y={PAD.t + i * ch}
                  width={cw + 0.5}
                  height={ch + 0.5}
                  fill={colorFor(v)}
                />
              ))
            )}
          </svg>
        </div>

        <div className="flex flex-col gap-3">
          <SliderRow label="head index h" value={headIdx} onChange={(v) => setHeadIdx(Math.round(v))} min={0} max={nHeads - 1} step={1} fmt={(v) => v.toFixed(0)} color={C_ALIBI} />
          <SliderRow label="num heads H" value={nHeads} onChange={(v) => setNHeads(Math.round(v))} min={2} max={32} step={1} fmt={(v) => v.toFixed(0)} color={C_ALIBI} />

          <hr className="border-ink-800/80 my-1" />
          <div className="grid grid-cols-2 gap-2 text-[12px] font-mono">
            <Stat label="m_h (slope)" value={slope.toExponential(2)} color={C_ALIBI} hint={`2^${(Math.log2(slope)).toFixed(2)}`} />
            <Stat label="penalty @ Δ=8" value={(slope * 8).toFixed(2)} color={C_ALIBI} hint="logit reduction" />
          </div>

          <p className="text-[11px] text-ink-400 leading-snug">
            Each head gets its own slope <InlineMath math={String.raw`m_h`} />,
            so the early heads look almost local (steep slope, only attend to
            recent tokens) while later heads see almost everything. No learned
            parameters — just a fixed schedule that extrapolates cleanly to
            untrained sequence lengths.
          </p>
        </div>
      </div>
    </section>
  );
}

/* =============================== length extrapolation =============================== */

function ExtrapolationSection() {
  const trainLen = 128;
  const testLen = 512;
  const dim = 64;

  // For each method, plot self-similarity (cosine of PE[pos] with PE[pos % trainLen])
  // as a function of pos ∈ [0, testLen]. Sinusoidal stays high (periodic), learned
  // craters past trainLen (out-of-vocab), RoPE has its own pattern.
  const W = 720;
  const H = 240;
  const PAD = { l: 38, r: 14, t: 16, b: 28 };
  const xMin = 0;
  const xMax = testLen;
  const yMin = -1;
  const yMax = 1;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  type Series = { method: Exclude<Method, "alibi">; color: string; label: string };
  const series: Series[] = [
    { method: "sinusoidal", color: C_SIN, label: "Sinusoidal" },
    { method: "learned", color: C_LEARNED, label: "Learned" },
    { method: "rope", color: C_ROPE, label: "RoPE" },
  ];

  const paths = useMemo(() => {
    return series.map((s) => {
      const pts: { x: number; y: number }[] = [];
      for (let pos = 0; pos < testLen; pos += 2) {
        const sim = embeddingSelfSim(s.method, pos, dim);
        pts.push({ x: pos, y: sim });
      }
      const d = pts
        .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x).toFixed(2)} ${sy(p.y).toFixed(2)}`)
        .join(" ");
      return { ...s, d };
    });
  }, []);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Length extrapolation · train@{trainLen}, eval@{testLen}
        </span>
        <span className="text-[10px] font-mono text-ink-500">
          embedding similarity to position 0
        </span>
      </div>

      <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
          <Grid sx={sx} sy={sy} yTicks={[-1, -0.5, 0, 0.5, 1]} xTicks={[0, 64, 128, 192, 256, 320, 384, 448, 512]} W={W} H={H} PAD={PAD} compact xLabel="pos" />
          {/* train cutoff */}
          <line x1={sx(trainLen)} y1={PAD.t} x2={sx(trainLen)} y2={H - PAD.b} stroke="rgba(255,255,255,0.18)" strokeDasharray="2 4" />
          <text x={sx(trainLen) + 4} y={PAD.t + 12} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
            train cutoff
          </text>
          <line x1={sx(xMin)} y1={sy(0)} x2={sx(xMax)} y2={sy(0)} stroke="rgba(255,255,255,0.06)" />
          {paths.map((p) => (
            <path key={p.method} d={p.d} stroke={p.color} strokeWidth={2.2} fill="none" />
          ))}
          {/* legend */}
          <g transform={`translate(${PAD.l + 8}, ${PAD.t + 6})`}>
            <rect width="118" height="46" rx={6} fill="rgba(0,0,0,0.45)" />
            {paths.map((p, i) => (
              <g key={p.method} transform={`translate(8, ${12 + i * 14})`}>
                <line x1={0} y1={0} x2={16} y2={0} stroke={p.color} strokeWidth={2.2} />
                <text x={22} y={3} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">{p.label}</text>
              </g>
            ))}
          </g>
        </svg>
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        Past the dashed train cutoff, <span className="text-violet-300">learned</span>{" "}
        embeddings have <em>never been trained</em> — they're effectively
        random.{" "}
        <span className="text-cyan-300">Sinusoidal</span> and{" "}
        <span className="text-pink-300">RoPE</span> stay structured because they
        are deterministic functions of position; that is the empirical reason
        modern LLMs (LLaMA, Qwen, Mistral) all use RoPE and ALiBi-flavored
        schemes for long-context generalization.
      </p>
    </section>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="Why needed">
        <InlineMath math={String.raw`\text{Attention}(\pi X) = \pi\,\text{Attention}(X)`} />
      </PropCard>
      <PropCard title="Sinusoidal">
        <InlineMath math={String.raw`\text{fixed, deterministic, periodic}`} />
      </PropCard>
      <PropCard title="Learned">
        <InlineMath math={String.raw`\mathbb{R}^{L_{\max}\times d}\text{ parameters}`} />
      </PropCard>
      <PropCard title="RoPE relative-position">
        <InlineMath math={String.raw`\langle R_m q,\, R_n k\rangle = f(q, k, n-m)`} />
      </PropCard>
      <PropCard title="RoPE base">
        <InlineMath math={String.raw`B=10000 \to \text{period } \approx 2\pi B^{2i/d}`} />
      </PropCard>
      <PropCard title="ALiBi schedule">
        <InlineMath math={String.raw`m_h = 2^{-8(h+1)/H}`} />
      </PropCard>
      <PropCard title="ALiBi parameter-free">
        <InlineMath math={String.raw`\#\text{params}_{\text{ALiBi}} = 0`} />
      </PropCard>
      <PropCard title="Length extrapolation">
        <InlineMath math={String.raw`\text{RoPE+NTK} \approx \text{ALiBi} \gg \text{learned}`} />
      </PropCard>
      <PropCard title="Pre- vs post-attention">
        <InlineMath math={String.raw`\text{sin/learned}: \text{embed};\; \text{RoPE/ALiBi}: \text{attn}`} />
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
        <UsageItem head="Sinusoidal" body="original Transformer, T5 (variant), most pre-2021 encoder-decoders." />
        <UsageItem head="Learned absolute" body="BERT, GPT-2, GPT-3, ViT — bounded by L_max trained on." />
        <UsageItem head="RoPE" body="LLaMA-1/2/3, Qwen-2.5, Mistral, Gemma, Yi, DeepSeek." />
        <UsageItem head="ALiBi" body="BLOOM, MPT, Falcon (variants), strong length extrapolation." />
        <UsageItem head="NoPE" body="recent ablations show decoder-only LMs without PE can still learn position implicitly." />
        <UsageItem head="2D / patch PE" body="ViT, DALL·E, image-token grids." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-emerald-300">
        <UsageItem head="Sinusoidal" body="zero parameters, easy extrapolation, classical inductive bias." />
        <UsageItem head="Learned" body="task-specific, no formula required, can capture odd structure." />
        <UsageItem head="RoPE" body={<>relative-position via{" "}<InlineMath math={String.raw`\langle R_m q, R_n k\rangle`} /> depends only on{" "}<InlineMath math={String.raw`n-m`} />.</>} />
        <UsageItem head="ALiBi" body="local-attention bias for free, scales effortlessly to 32k+ tokens." />
        <UsageItem head="Decoupling embedding from PE" body="RoPE/ALiBi let the embedding stay length-agnostic." />
        <UsageItem head="NTK / YaRN / LongRoPE" body="post-training RoPE-base scaling extends context 4-8× without retraining." />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem head="Sinusoidal beyond train length" body="works, but attention patterns may still degrade." />
        <UsageItem head="Learned beyond L_max" body="undefined — model has no embedding for that position." />
        <UsageItem head="RoPE base tuning" body="fixed B=10000 limits long-context; YaRN / NTK fixes it post-hoc." />
        <UsageItem head="ALiBi attention pattern" body="enforces locality bias — bad for some retrieval-heavy tasks." />
        <UsageItem head="No 2D structure" body="all four assume 1D position; image / video need 2D / 3D extensions." />
        <UsageItem head="Causal mask interaction" body="ALiBi penalty must be combined with causal mask carefully." />
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

/* =============================== similarity helper =============================== */

function embeddingSelfSim(
  method: Exclude<Method, "alibi">,
  pos: number,
  dim: number
): number {
  // Cosine similarity between embedding(pos) and embedding(0)
  const a = buildEmbeddingMatrix(method, pos + 1, dim)[pos];
  const b = buildEmbeddingMatrix(method, 1, dim)[0];
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 1e-12 ? dot / denom : 0;
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
      <span className="text-base font-semibold tabular-nums" style={color ? { color } : undefined}>
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
  fmt,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  color?: string;
  fmt?: (v: number) => string;
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
      <span className="w-12 text-right tabular-nums">
        {fmt ? fmt(value) : value.toFixed(2)}
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
  yTicks: number[];
  xTicks: number[];
  W: number;
  H: number;
  PAD: { l: number; r: number; t: number; b: number };
  compact?: boolean;
  xLabel?: string;
}

function Grid({ sx, sy, yTicks, xTicks, W, H, PAD, compact, xLabel }: GridProps) {
  const tickFs = compact ? 9 : 10;
  return (
    <g>
      <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke={C_AXIS} />
      <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke={C_AXIS} />
      {yTicks.map((v) => (
        <g key={`y-${v}`}>
          <line x1={PAD.l - 4} y1={sy(v)} x2={W - PAD.r} y2={sy(v)} stroke="rgba(255,255,255,0.04)" />
          <text x={PAD.l - 6} y={sy(v) + 3} textAnchor="end" fontSize={tickFs} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
            {fmt(v)}
          </text>
        </g>
      ))}
      {xTicks.map((v) => (
        <g key={`x-${v}`}>
          <line x1={sx(v)} y1={H - PAD.b} x2={sx(v)} y2={H - PAD.b + 4} stroke={C_AXIS} />
          <text x={sx(v)} y={H - PAD.b + 14} textAnchor="middle" fontSize={tickFs} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
            {fmt(v)}
          </text>
        </g>
      ))}
      {xLabel && (
        <text x={W - PAD.r} y={H - PAD.b - 4} textAnchor="end" fontSize={tickFs} fill="#5b6478" fontFamily="JetBrains Mono, monospace">
          {xLabel}
        </text>
      )}
    </g>
  );
}

function fmt(v: number): string {
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(2).replace(/\.?0+$/, "");
}
