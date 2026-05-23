import { useMemo, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  DEFAULT_LOGITS,
  SAMPLE_VOCAB,
  applyRepetitionPenalty,
  argmax,
  beamStep,
  entropyOf,
  lcg,
  minPMask,
  sampleFrom,
  softmaxT,
  topKMask,
  topPMask,
  type Beam,
} from "../lib/sampling";

const C_AXIS = "rgba(255,255,255,0.10)";
const C_FULL = "#7c5cff";
const C_KEEP = "#22d3ee";
const C_DROP = "#3b3f55";
const C_GREEDY = "#fbbf24";
const C_ENTROPY = "#34d399";

export default function SamplingViz() {
  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <SamplingLab />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <EntropyVsTemperaturePanel />
        <DrawSequencePanel />
      </div>

      <BeamSearchPanel />

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
          <span className="text-violet-300">Sampling</span> turns a vector of
          next-token logits into the actual token a language model emits. The
          knobs — <em>temperature</em>, <em>top-k</em>, <em>top-p</em>,{" "}
          <em>min-p</em>, <em>repetition penalty</em> — reshape the categorical
          distribution before a draw. Together they trade off{" "}
          <span className="text-cyan-300">coherence</span> against{" "}
          <span className="text-amber-300">diversity</span>.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`p_i \;=\; \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\text{top-}p\text{: smallest }S \text{ s.t. } \sum_{i\in S} p_i \ge p`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\text{min-}p\text{: keep } i \text{ if } p_i \ge p_{\min} \cdot \max_j p_j`} />
      </MathBox>
    </section>
  );
}

/* =============================== main lab =============================== */

function SamplingLab() {
  const [logits, setLogits] = useState<number[]>([...DEFAULT_LOGITS]);
  const [tau, setTau] = useState(1.0);
  const [k, setK] = useState(0);
  const [topP, setTopP] = useState(1.0);
  const [minP, setMinP] = useState(0);
  const [repPenalty, setRepPenalty] = useState(1.0);
  const [seenTokens, setSeenTokens] = useState<number[]>([]);

  const effectiveLogits = useMemo(
    () => applyRepetitionPenalty(logits, seenTokens, repPenalty),
    [logits, seenTokens, repPenalty]
  );
  const baseProbs = useMemo(() => softmaxT(effectiveLogits, tau), [effectiveLogits, tau]);

  // Pipeline: top-k → top-p → min-p (matches the HuggingFace / vLLM default order).
  const afterK = useMemo(() => (k > 0 ? topKMask(baseProbs, k) : baseProbs), [baseProbs, k]);
  const afterP = useMemo(() => topPMask(afterK, topP), [afterK, topP]);
  const afterMinP = useMemo(() => minPMask(afterP, minP), [afterP, minP]);

  const finalProbs = afterMinP;
  const entropy = useMemo(() => entropyOf(finalProbs), [finalProbs]);
  const greedy = useMemo(() => argmax(baseProbs), [baseProbs]);
  const surviving = finalProbs.filter((v) => v > 1e-9).length;

  const setLogit = (i: number, v: number) =>
    setLogits(logits.map((x, j) => (j === i ? v : x)));

  const resetLogits = () => setLogits([...DEFAULT_LOGITS]);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Sampling lab · live distribution
        </span>
        <button
          onClick={resetLogits}
          className="text-[10px] font-mono text-ink-400 hover:text-white border border-ink-700 px-2 py-0.5 rounded"
        >
          reset logits
        </button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_280px] gap-4">
        <DistributionBars
          tokens={SAMPLE_VOCAB}
          base={baseProbs}
          final={finalProbs}
          greedy={greedy}
        />
        <div className="flex flex-col gap-3">
          <SliderRow
            label={`τ (temperature)`}
            value={tau}
            onChange={setTau}
            min={0.05}
            max={3}
            step={0.01}
            color={C_FULL}
          />
          <SliderRow
            label="top-k  (0 = off)"
            value={k}
            onChange={(v) => setK(Math.round(v))}
            min={0}
            max={SAMPLE_VOCAB.length}
            step={1}
            color={C_KEEP}
            format={(v) => v.toFixed(0)}
          />
          <SliderRow
            label="top-p (nucleus)"
            value={topP}
            onChange={setTopP}
            min={0.05}
            max={1}
            step={0.01}
            color={C_KEEP}
          />
          <SliderRow
            label="min-p"
            value={minP}
            onChange={setMinP}
            min={0}
            max={0.5}
            step={0.005}
            color={C_KEEP}
            format={(v) => v.toFixed(3)}
          />
          <SliderRow
            label="repetition penalty"
            value={repPenalty}
            onChange={setRepPenalty}
            min={1}
            max={2}
            step={0.01}
            color={C_GREEDY}
          />

          <hr className="border-ink-800/80 my-1" />

          <div className="grid grid-cols-2 gap-2 text-[12px] font-mono">
            <Stat label="H(p)" value={entropy.toFixed(3)} color={C_ENTROPY} hint="nats" />
            <Stat label="# surviving" value={surviving.toString()} color={C_KEEP} hint={`of ${SAMPLE_VOCAB.length}`} />
            <Stat label="max p" value={Math.max(...finalProbs).toFixed(3)} color={C_FULL} />
            <Stat label="argmax" value={SAMPLE_VOCAB[argmax(finalProbs)]} color={C_GREEDY} />
          </div>

          <p className="text-[11px] text-ink-400 leading-snug">
            Order applied: <span className="text-cyan-300">top-k → top-p → min-p</span> (the HuggingFace / vLLM default).
            Repetition penalty acts on logits <em>before</em> softmax.
          </p>
        </div>
      </div>

      <LogitEditor logits={logits} onChange={setLogit} seen={seenTokens} onSeen={setSeenTokens} />
    </section>
  );
}

function DistributionBars({
  tokens,
  base,
  final,
  greedy,
}: {
  tokens: string[];
  base: number[];
  final: number[];
  greedy: number;
}) {
  const W = 540;
  const H = 240;
  const PAD = { l: 38, r: 12, t: 18, b: 36 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;
  const groupW = innerW / tokens.length;
  const barW = groupW * 0.7;
  const yMax = Math.max(...base, ...final, 0.05);

  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider px-1 pb-1">
        <span className="text-ink-400 font-bold">p(token | context)</span>
        <span className="flex items-center gap-2 text-[10px] font-mono">
          <span className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: C_DROP }} />
            <span className="text-ink-300">filtered</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: C_KEEP }} />
            <span className="text-ink-300">kept</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: C_GREEDY }} />
            <span className="text-ink-300">argmax</span>
          </span>
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke={C_AXIS} />
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke={C_AXIS} />
        {[0, 0.25, 0.5, 0.75, 1].map((g) => {
          const v = g * yMax;
          const y = PAD.t + (1 - g) * innerH;
          return (
            <g key={g}>
              <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} stroke="rgba(255,255,255,0.04)" />
              <text x={PAD.l - 4} y={y + 3} textAnchor="end" fontSize={9} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
                {v.toFixed(2)}
              </text>
            </g>
          );
        })}
        {tokens.map((tok, i) => {
          const cx = PAD.l + i * groupW + groupW / 2;
          const hBase = (base[i] / yMax) * innerH;
          const hFinal = (final[i] / yMax) * innerH;
          const kept = final[i] > 1e-9;
          return (
            <g key={i}>
              <rect
                x={cx - barW / 2}
                y={H - PAD.b - hBase}
                width={barW}
                height={hBase}
                fill={C_DROP}
                opacity={0.55}
              />
              {kept && (
                <rect
                  x={cx - barW / 2}
                  y={H - PAD.b - hFinal}
                  width={barW}
                  height={hFinal}
                  fill={i === greedy ? C_GREEDY : C_KEEP}
                />
              )}
              <text
                x={cx}
                y={H - PAD.b + 14}
                textAnchor="middle"
                fontSize={10}
                fill={kept ? "#cbd5e1" : "#5b6478"}
                fontFamily="JetBrains Mono, monospace"
                transform={`rotate(-30, ${cx}, ${H - PAD.b + 14})`}
              >
                {tok}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function LogitEditor({
  logits,
  onChange,
  seen,
  onSeen,
}: {
  logits: number[];
  onChange: (i: number, v: number) => void;
  seen: number[];
  onSeen: (s: number[]) => void;
}) {
  const seenSet = new Set(seen);
  const toggleSeen = (i: number) => {
    const next = new Set(seenSet);
    if (next.has(i)) next.delete(i);
    else next.add(i);
    onSeen(Array.from(next));
  };
  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider font-bold">
        <span className="text-ink-400">Logits z (click "seen" to penalize repetition)</span>
        <span className="text-ink-500 font-mono">drag sliders</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-1.5">
        {logits.map((z, i) => (
          <div key={i} className="flex items-center gap-2 font-mono text-[11px]">
            <span className="w-16 shrink-0 text-ink-300">{SAMPLE_VOCAB[i]}</span>
            <input
              type="range"
              min={-4}
              max={4}
              step={0.05}
              value={z}
              onChange={(e) => onChange(i, parseFloat(e.target.value))}
              className="flex-1 accent-violet-400"
            />
            <span className="w-10 text-right tabular-nums">{z.toFixed(2)}</span>
            <button
              onClick={() => toggleSeen(i)}
              className={`text-[10px] px-1.5 py-0.5 rounded border ${
                seenSet.has(i)
                  ? "border-amber-400/60 text-amber-300 bg-amber-500/10"
                  : "border-ink-700 text-ink-500 hover:text-white"
              }`}
            >
              seen
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

/* =============================== entropy vs τ =============================== */

function EntropyVsTemperaturePanel() {
  const W = 380;
  const H = 240;
  const PAD = { l: 36, r: 12, t: 14, b: 28 };
  const xMin = 0.1;
  const xMax = 3;
  const yMin = 0;
  const yMax = Math.log(16) + 0.1;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const points = useMemo(() => {
    const pts: { t: number; h: number }[] = [];
    for (let i = 0; i < 220; i++) {
      const t = xMin + ((xMax - xMin) * i) / 219;
      pts.push({ t, h: entropyOf(softmaxT(DEFAULT_LOGITS, t)) });
    }
    return pts;
  }, []);

  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.t).toFixed(2)} ${sy(p.h).toFixed(2)}`)
    .join(" ");

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Entropy vs temperature
        </span>
        <BlockMath math={String.raw`H(p_\tau) = -\!\sum_i p_i^{(\tau)}\log p_i^{(\tau)}`} />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <Grid sx={sx} sy={sy} yTicks={[0, 1, 2, Math.log(16)]} xTicks={[0.1, 0.5, 1, 1.5, 2, 2.5, 3]} W={W} H={H} PAD={PAD} compact xLabel="τ" />
        <line x1={sx(xMin)} y1={sy(Math.log(16))} x2={sx(xMax)} y2={sy(Math.log(16))} stroke="rgba(255,255,255,0.10)" strokeDasharray="2 4" />
        <text x={W - PAD.r - 6} y={sy(Math.log(16)) - 3} textAnchor="end" fontSize={9} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
          log(V) ≈ max entropy
        </text>
        <path d={path} stroke={C_ENTROPY} strokeWidth={2.4} fill="none" />
      </svg>
      <p className="text-[11px] text-ink-400 leading-snug">
        As <InlineMath math={String.raw`\tau \to 0`} /> the softmax collapses to
        argmax (entropy → 0, greedy). As{" "}
        <InlineMath math={String.raw`\tau \to \infty`} /> it flattens to uniform
        (entropy → <InlineMath math={String.raw`\log V`} />, fully random).
      </p>
    </section>
  );
}

/* =============================== draw sequence =============================== */

function DrawSequencePanel() {
  const [seed, setSeed] = useState(42);
  const [tau, setTau] = useState(0.8);
  const [k, setK] = useState(0);
  const [topP, setTopP] = useState(0.95);
  const [n, setN] = useState(20);

  const draws = useMemo(() => {
    const rng = lcg(seed);
    const out: number[] = [];
    let probs = softmaxT(DEFAULT_LOGITS, tau);
    if (k > 0) probs = topKMask(probs, k);
    if (topP < 1) probs = topPMask(probs, topP);
    for (let i = 0; i < n; i++) out.push(sampleFrom(probs, rng()));
    return out;
  }, [seed, tau, k, topP, n]);

  // Count token frequencies for the bar viz.
  const counts = useMemo(() => {
    const c = new Array(SAMPLE_VOCAB.length).fill(0);
    for (const i of draws) c[i] += 1;
    return c;
  }, [draws]);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Draw N tokens (replay)
        </span>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="text-[10px] font-mono text-ink-400 hover:text-white border border-ink-700 px-2 py-0.5 rounded"
        >
          re-roll seed
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-4 gap-2">
        <MiniSlider label="τ" value={tau} onChange={setTau} min={0.05} max={2} step={0.01} />
        <MiniSlider label="k" value={k} onChange={(v) => setK(Math.round(v))} min={0} max={SAMPLE_VOCAB.length} step={1} fmt={(v) => v.toFixed(0)} />
        <MiniSlider label="top-p" value={topP} onChange={setTopP} min={0.05} max={1} step={0.01} />
        <MiniSlider label="N" value={n} onChange={(v) => setN(Math.round(v))} min={4} max={120} step={1} fmt={(v) => v.toFixed(0)} />
      </div>

      <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2.5 flex flex-wrap gap-1.5 min-h-[48px]">
        {draws.map((i, j) => (
          <span
            key={j}
            className="px-1.5 py-0.5 rounded text-[11px] font-mono"
            style={{ background: "rgba(34,211,238,0.12)", color: "#a5f3fc" }}
          >
            {SAMPLE_VOCAB[i]}
          </span>
        ))}
      </div>

      <DrawHistogram counts={counts} total={n} />
    </section>
  );
}

function DrawHistogram({ counts, total }: { counts: number[]; total: number }) {
  const W = 380;
  const H = 90;
  const PAD = { l: 4, r: 4, t: 4, b: 18 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;
  const groupW = innerW / SAMPLE_VOCAB.length;
  const barW = groupW * 0.7;
  const maxC = Math.max(1, ...counts);
  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-1">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        {counts.map((c, i) => {
          const cx = PAD.l + i * groupW + groupW / 2;
          const h = (c / maxC) * innerH;
          return (
            <g key={i}>
              <rect x={cx - barW / 2} y={H - PAD.b - h} width={barW} height={h} fill={C_KEEP} opacity={0.85} />
              <text x={cx} y={H - 4} textAnchor="middle" fontSize={8} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
                {SAMPLE_VOCAB[i].slice(0, 4)}
              </text>
            </g>
          );
        })}
        <text x={W - 4} y={12} textAnchor="end" fontSize={9} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
          {total} draws
        </text>
      </svg>
    </div>
  );
}

/* =============================== beam search =============================== */

function BeamSearchPanel() {
  const [width, setWidth] = useState(3);
  const [steps, setSteps] = useState(4);

  // Deterministic toy "next-token" log-probs that depend on the last token
  // (so beams differentiate). Top vocab is 8 to keep the tree readable.
  const localVocab = SAMPLE_VOCAB.slice(0, 8);
  const nextLogProbs = (b: Beam) => {
    const last = b.tokens[b.tokens.length - 1] ?? -1;
    const logits = localVocab.map((_, i) => {
      const base = 2 - i * 0.4;
      const bonus = ((i + last * 3) % localVocab.length === 0 ? 0.8 : 0) +
        ((i + last) % 5 === 0 ? 0.3 : 0);
      return base + bonus;
    });
    const probs = softmaxT(logits, 1);
    return probs.map((p) => Math.log(Math.max(p, 1e-9)));
  };

  const beams = useMemo(() => {
    let current: Beam[] = [{ tokens: [0], logProb: 0 }];
    const history: Beam[][] = [current];
    for (let s = 0; s < steps; s++) {
      current = beamStep(current, width, nextLogProbs);
      history.push(current);
    }
    return { history, final: current };
  }, [width, steps]);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Beam search · top-{width} hypothesis tree
        </span>
        <BlockMath math={String.raw`\arg\max_{y_{1:T}} \sum_t \log p(y_t \mid y_{<t})`} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <MiniSlider label="beam width" value={width} onChange={(v) => setWidth(Math.round(v))} min={1} max={6} step={1} fmt={(v) => v.toFixed(0)} />
        <MiniSlider label="steps" value={steps} onChange={(v) => setSteps(Math.round(v))} min={1} max={6} step={1} fmt={(v) => v.toFixed(0)} />
      </div>

      <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
        <BeamTree history={beams.history} vocab={localVocab} />
      </div>

      <div className="flex flex-col gap-1 text-[12px] font-mono">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Final beams
        </span>
        {beams.final.slice(0, 5).map((b, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-ink-500 w-4 text-right">{i + 1}.</span>
            <span className="text-ink-100">
              {b.tokens.map((t) => localVocab[t]).join(" ")}
            </span>
            <span className="text-ink-500 ml-auto tabular-nums">
              logp = {b.logProb.toFixed(2)}
            </span>
          </div>
        ))}
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        Beam search is <em>not</em> sampling — it deterministically keeps the
        top-<InlineMath math={String.raw`W`} /> highest-log-prob hypotheses at
        every step. Great for translation / summarization where a single most-likely
        sequence is wanted; tends to produce <em>repetitive</em> text for open-ended
        generation, which is why open chat models prefer top-p / top-k sampling.
      </p>
    </section>
  );
}

function BeamTree({ history, vocab }: { history: Beam[][]; vocab: string[] }) {
  const W = 720;
  const stepW = W / Math.max(history.length, 1);
  const rowH = 22;
  const H = Math.max(...history.map((bs) => bs.length)) * rowH + 20;

  const positions = history.map((bs, s) =>
    bs.map((_, i) => ({ x: s * stepW + stepW / 2, y: 14 + i * rowH }))
  );

  // Connect each beam at step s to its parent in step s-1 (matched by tokens prefix).
  const links: { x1: number; y1: number; x2: number; y2: number }[] = [];
  for (let s = 1; s < history.length; s++) {
    const prev = history[s - 1];
    const curr = history[s];
    for (let j = 0; j < curr.length; j++) {
      const childTokens = curr[j].tokens;
      const parentIdx = prev.findIndex(
        (p) => p.tokens.length === childTokens.length - 1 && p.tokens.every((t, k) => t === childTokens[k])
      );
      if (parentIdx >= 0) {
        links.push({
          x1: positions[s - 1][parentIdx].x + 16,
          y1: positions[s - 1][parentIdx].y,
          x2: positions[s][j].x - 16,
          y2: positions[s][j].y,
        });
      }
    }
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
      {links.map((l, i) => (
        <line key={i} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} stroke="rgba(124,92,255,0.45)" strokeWidth={1.2} />
      ))}
      {history.map((bs, s) =>
        bs.map((b, j) => {
          const { x, y } = positions[s][j];
          const tok = vocab[b.tokens[b.tokens.length - 1]];
          return (
            <g key={`${s}-${j}`}>
              <rect x={x - 22} y={y - 9} width={44} height={16} rx={4} fill="rgba(124,92,255,0.18)" stroke="rgba(124,92,255,0.5)" />
              <text x={x} y={y + 3} textAnchor="middle" fontSize={10} fill="#e5e7eb" fontFamily="JetBrains Mono, monospace">
                {tok}
              </text>
            </g>
          );
        })
      )}
    </svg>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="Temperature limits">
        <InlineMath math={String.raw`\tau \to 0:\text{greedy};\;\tau \to \infty:\text{uniform}`} />
      </PropCard>
      <PropCard title="Top-k is hard truncation">
        <InlineMath math={String.raw`p_i \gets 0\;\forall i\notin\mathrm{top}_k`} />
      </PropCard>
      <PropCard title="Top-p is adaptive">
        <InlineMath math={String.raw`|S|\;\text{shrinks when } p \text{ is peaky}`} />
      </PropCard>
      <PropCard title="Min-p is calibrated">
        <InlineMath math={String.raw`\text{threshold} = p_{\min}\cdot\max_j p_j`} />
      </PropCard>
      <PropCard title="Rep. penalty">
        <InlineMath math={String.raw`z_i \gets z_i/\rho\text{ if }z_i>0,\;z_i\cdot\rho\text{ else}`} />
      </PropCard>
      <PropCard title="Beam search">
        <InlineMath math={String.raw`\text{keep top-}W\text{ partial sequences by }\sum\log p`} />
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
        <UsageItem head="ChatGPT / Claude / Gemini" body="top-p ≈ 0.9–1.0, τ ≈ 0.7–1.0 by default; tunable per request." />
        <UsageItem head="Code generation" body="low τ (0.1–0.3), low top-p — coherence over diversity." />
        <UsageItem head="Creative writing" body="higher τ (0.8–1.2), top-p ≈ 0.95, repetition penalty ≈ 1.1." />
        <UsageItem head="Speech / TTS / image" body="diffusion samplers use temperature-like noise schedules." />
        <UsageItem head="MT / summarization" body="beam search (W=4–8) for highest single-best output." />
        <UsageItem head="RLHF rollout" body="τ=1, top-p=1 — uncontaminated samples for reward model." />
        <UsageItem head="Constrained decoding" body="mask logits outside grammar/JSON-schema set." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-cyan-300">
        <UsageItem head="Diversity" body="argmax loops; sampling explores the support of p." />
        <UsageItem head="Avoid degeneration" body="top-p / min-p cut the long, low-quality tail (Holtzman 2020)." />
        <UsageItem head="Tunable creativity" body={<>τ alone is enough to slide from <InlineMath math={String.raw`0\to\infty`} /> in spirit.</>} />
        <UsageItem head="Reasonable defaults" body="HF / vLLM combine top-k=50, top-p=0.95, τ=1 as a baseline." />
        <UsageItem head="Beam search optimality" body={<>finds <InlineMath math={String.raw`\arg\max\sum\log p`} /> exactly when W is large enough.</>} />
        <UsageItem head="Calibration-aware (min-p)" body="threshold is relative to the top token — robust to τ scaling." />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem head="Top-k is dataset-blind" body="k=50 is great for English, terrible for a tiny vocab." />
        <UsageItem head="Top-p over-flattens peaks" body="if max p is already 0.99, top-p=0.95 still drops it." />
        <UsageItem head="Beam → repetition" body="open-ended generation with beam search loops or degenerates." />
        <UsageItem head="τ ≠ knob for creativity" body="high τ can produce incoherent token-level garbage." />
        <UsageItem head="Repetition penalty hurts code / lists" body={<>legitimate repetition (e.g. {"`for`"}) suffers.</>} />
        <UsageItem head="No notion of length" body="beam search prefers short sequences; length-normalize or use coverage penalty." />
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
      <span className="w-12 text-right tabular-nums">
        {format ? format(value) : value.toFixed(2)}
      </span>
    </div>
  );
}

function MiniSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  fmt,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  fmt?: (v: number) => string;
}) {
  return (
    <div className="flex flex-col gap-0.5 font-mono text-[11px]">
      <div className="flex items-baseline justify-between">
        <span className="text-ink-400">{label}</span>
        <span className="text-white tabular-nums">{fmt ? fmt(value) : value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="accent-violet-400"
      />
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
