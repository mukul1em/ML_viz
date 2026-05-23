import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Eye,
  KeyRound,
  Library,
  Network,
  Package,
  Search,
  Sparkles,
  ToggleLeft,
  ToggleRight,
  Workflow,
  Zap,
} from "lucide-react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  DEFAULT_TOKENS,
  D_K,
  D_MODEL,
  fullAttentionMatrix,
  traceAttention,
} from "../lib/qkvDemo";

const Q_COLOR = "#f472b6";
const K_COLOR = "#22d3ee";
const V_COLOR = "#34d399";
const TOKEN_COLORS = [
  "#7c5cff",
  "#22d3ee",
  "#f472b6",
  "#fbbf24",
  "#34d399",
  "#f87171",
  "#a78bfa",
  "#60a5fa",
];

export default function QKVViz() {
  const [tokensRaw, setTokensRaw] = useState(DEFAULT_TOKENS.join(" "));
  const tokens = useMemo(
    () =>
      tokensRaw
        .split(/\s+/)
        .map((t) => t.trim())
        .filter(Boolean)
        .slice(0, 8),
    [tokensRaw]
  );
  const [queryIdx, setQueryIdx] = useState(1);
  const [useScaling, setUseScaling] = useState(true);
  const [causal, setCausal] = useState(false);

  const qi = Math.min(queryIdx, Math.max(tokens.length - 1, 0));
  const trace = useMemo(
    () => traceAttention(tokens, qi, { scale: useScaling, causal }),
    [tokens, qi, useScaling, causal]
  );
  const fullMatrix = useMemo(
    () => fullAttentionMatrix(tokens, { scale: useScaling, causal }),
    [tokens, useScaling, causal]
  );

  return (
    <div className="flex flex-col gap-10">
      {/* Controls */}
      <section className="card p-4 lg:p-5">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-4 items-center">
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold w-16">
              Sentence
            </span>
            <input
              value={tokensRaw}
              onChange={(e) => setTokensRaw(e.target.value)}
              className="flex-1 px-3 py-2 rounded-lg bg-ink-950/70 border border-ink-700 focus:border-pink-400/60 focus:outline-none font-mono text-sm"
            />
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <Toggle
              on={useScaling}
              onToggle={() => setUseScaling((s) => !s)}
              label="÷ √d_k"
              hint="scale by sqrt(d_k)"
            />
            <Toggle
              on={causal}
              onToggle={() => setCausal((s) => !s)}
              label="Causal mask"
              hint="prevent attending to future tokens"
            />
          </div>
        </div>
      </section>

      {/* Hero: the trinity */}
      <Hero />

      {/* Library analogy */}
      <LibraryAnalogy />

      {/* Three projections */}
      <ProjectionSection tokens={tokens} trace={trace} />

      {/* Interactive attention pipeline */}
      <PipelineSection
        tokens={tokens}
        queryIdx={qi}
        setQueryIdx={setQueryIdx}
        trace={trace}
        useScaling={useScaling}
        causal={causal}
      />

      {/* Full attention matrix */}
      <FullMatrixSection
        tokens={tokens}
        matrix={fullMatrix}
        causal={causal}
        queryIdx={qi}
      />

      {/* Why three? */}
      <ImportanceSection />
    </div>
  );
}

/* -------------------------------- Sections -------------------------------- */

function Hero() {
  return (
    <section className="flex flex-col gap-5">
      <Eyebrow icon={<KeyRound className="w-3 h-3" />}>The trinity</Eyebrow>
      <h2 className="text-2xl lg:text-3xl font-extrabold tracking-tight">
        Every token plays{" "}
        <span className="bg-gradient-to-r from-pink-400 to-cyan-400 bg-clip-text text-transparent">
          three roles
        </span>{" "}
        at once
      </h2>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Self-attention starts with each token producing three vectors from its
        embedding: a <span className="text-pink-300 font-semibold">Query</span>,
        a <span className="text-cyan-300 font-semibold">Key</span>, and a{" "}
        <span className="text-emerald-300 font-semibold">Value</span>. The
        whole machinery of transformers is built on top of how these three
        cooperate.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
        <RoleCard
          color={Q_COLOR}
          letter="Q"
          name="Query"
          question="What am I looking for?"
          body="The current token asks: what kind of information would help me right now? It probes the rest of the sequence."
          icon={<Search className="w-4 h-4" />}
        />
        <RoleCard
          color={K_COLOR}
          letter="K"
          name="Key"
          question="What do I represent?"
          body="Every token advertises a key: a compact signature describing what kind of queries it can answer. Queries are matched against these keys."
          icon={<KeyRound className="w-4 h-4" />}
        />
        <RoleCard
          color={V_COLOR}
          letter="V"
          name="Value"
          question="What do I contribute?"
          body="If a token is judged relevant, this is the content it actually contributes back into the answer. Decoupled from how it gets matched."
          icon={<Package className="w-4 h-4" />}
        />
      </div>
    </section>
  );
}

function LibraryAnalogy() {
  return (
    <section className="card p-5 lg:p-6">
      <div className="flex items-start gap-4 mb-4">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-pink-500/30 to-cyan-500/30 border border-pink-400/30 flex items-center justify-center shrink-0">
          <Library className="w-5 h-5 text-pink-200" />
        </div>
        <div>
          <h3 className="text-lg font-bold">The library analogy</h3>
          <p className="text-sm text-ink-300 mt-0.5">
            A concrete intuition that always sticks.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <AnalogyCard
          color={Q_COLOR}
          title="You walk in with a Question"
          body={
            <>
              You want to know: <em>"how do octopuses solve mazes?"</em> That
              question is your <strong>Query</strong>. Today you only have one
              question, but you came to find anything that helps answer it.
            </>
          }
        />
        <AnalogyCard
          color={K_COLOR}
          title="Books advertise Keywords"
          body={
            <>
              Every book on the shelf has labels on its spine — animal biology,
              cephalopods, mazes, puzzles. Those labels are the book's{" "}
              <strong>Key</strong>. You don't read every book; you scan
              keywords first.
            </>
          }
        />
        <AnalogyCard
          color={V_COLOR}
          title="The book itself is the Value"
          body={
            <>
              When a key matches your query well, you actually <em>open</em>{" "}
              that book and use its contents. That content is the{" "}
              <strong>Value</strong>. A bad-keyword book stays closed; a
              well-matched book contributes a lot.
            </>
          }
        />
      </div>

      <div className="mt-5 rounded-xl border border-pink-400/30 bg-pink-400/[0.05] p-4 text-sm text-ink-200 leading-relaxed">
        <span className="text-pink-300 font-semibold">Aha →</span>{" "}
        Attention is a soft, differentiable lookup. The query/key dot product
        decides <em>how relevant</em> each item is (a weight), and we return a{" "}
        <em>weighted average</em> of every value — not a single hard pick.
        That's why it's smooth enough to train with gradient descent.
      </div>
    </section>
  );
}

function ProjectionSection({
  tokens,
  trace,
}: {
  tokens: string[];
  trace: ReturnType<typeof traceAttention>;
}) {
  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Workflow className="w-3 h-3" />}>Three projections</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        One input, three learned matrices
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Each token's embedding{" "}
        <InlineMath math={String.raw`x_i \in \mathbb{R}^{d_{model}}`} /> is
        projected three separate times by{" "}
        <InlineMath math="W_Q" />, <InlineMath math="W_K" />,{" "}
        <InlineMath math="W_V" /> to produce its query, key, and value vectors.
        The matrices are <strong>learned</strong>; they make the difference
        between a clueless model and a fluent one.
      </p>

      <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4 overflow-x-auto">
        <BlockMath math={String.raw`Q = X W_Q \quad\quad K = X W_K \quad\quad V = X W_V`} />
        <div className="text-xs text-ink-400 text-center mt-1">
          shapes: X ∈ ℝⁿˣᵈ ·  W ∈ ℝᵈˣᵈₖ  →  Q, K, V ∈ ℝⁿˣᵈₖ
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1.4fr_2fr] gap-5">
        {/* Diagram: x → 3 projections */}
        <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-5">
          <div className="text-[11px] uppercase tracking-wider text-ink-300 font-semibold mb-4">
            One token, three projections
          </div>
          <svg viewBox="0 0 400 240" className="w-full">
            <defs>
              <marker
                id="qkv-arrow"
                markerWidth="10"
                markerHeight="10"
                refX="6"
                refY="3"
                orient="auto"
              >
                <path d="M0,0 L0,6 L7,3 z" fill="#7c5cff" />
              </marker>
            </defs>

            {/* Input x */}
            <rect
              x="20"
              y="100"
              width="100"
              height="40"
              rx="10"
              fill="#7c5cff22"
              stroke="#7c5cff88"
            />
            <text
              x="70"
              y="125"
              textAnchor="middle"
              fill="#fff"
              fontSize={12}
              fontWeight={600}
            >
              xᵢ ∈ ℝᵈ
            </text>

            {/* W matrices */}
            <ProjArrow x1={120} y1={120} x2={210} y2={40} label="× Wq" color={Q_COLOR} />
            <ProjArrow x1={120} y1={120} x2={210} y2={120} label="× Wk" color={K_COLOR} />
            <ProjArrow x1={120} y1={120} x2={210} y2={200} label="× Wv" color={V_COLOR} />

            {/* Output vectors */}
            <OutputBlock x={210} y={20} color={Q_COLOR} letter="qᵢ" full="query" />
            <OutputBlock x={210} y={100} color={K_COLOR} letter="kᵢ" full="key" />
            <OutputBlock x={210} y={180} color={V_COLOR} letter="vᵢ" full="value" />
          </svg>
          <div className="text-[11px] text-ink-400 mt-2">
            The three matrices are independent — nothing forces them to learn
            the same representation.
          </div>
        </div>

        {/* Heatmaps: actual values for the picked query token */}
        <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4">
          <div className="text-[11px] uppercase tracking-wider text-ink-300 font-semibold mb-3">
            All tokens · their Q, K, V vectors
          </div>
          <div className="grid grid-cols-[5.5rem_1fr_1fr_1fr] gap-2 text-[11px] font-mono text-ink-400 mb-1.5">
            <span></span>
            <span className="text-pink-300 text-center">Q</span>
            <span className="text-cyan-300 text-center">K</span>
            <span className="text-emerald-300 text-center">V</span>
          </div>
          <div className="flex flex-col gap-1.5">
            {tokens.map((t, i) => (
              <div
                key={i}
                className="grid grid-cols-[5.5rem_1fr_1fr_1fr] gap-2 items-center"
              >
                <TokenChip token={t} index={i} />
                <HeatRow values={trace.Q[i]} accent={Q_COLOR} />
                <HeatRow values={trace.K[i]} accent={K_COLOR} />
                <HeatRow values={trace.V[i]} accent={V_COLOR} />
              </div>
            ))}
          </div>
          <div className="text-[11px] text-ink-400 mt-3">
            Each row is one token's projection into {D_K}-d space (compressed
            from d_model={D_MODEL} for visualization).
          </div>
        </div>
      </div>
    </section>
  );
}

function PipelineSection({
  tokens,
  queryIdx,
  setQueryIdx,
  trace,
  useScaling,
  causal,
}: {
  tokens: string[];
  queryIdx: number;
  setQueryIdx: (i: number) => void;
  trace: ReturnType<typeof traceAttention>;
  useScaling: boolean;
  causal: boolean;
}) {
  const q = trace.Q[queryIdx] ?? [];
  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Zap className="w-3 h-3" />}>The pipeline</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        Watch one token attend
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Pick a <span className="text-pink-300 font-semibold">query</span>{" "}
        token. We compute its dot product with every{" "}
        <span className="text-cyan-300 font-semibold">key</span>, scale,
        softmax — and use those weights to mix the{" "}
        <span className="text-emerald-300 font-semibold">values</span> into the
        output for that position.
      </p>

      {/* Query picker */}
      <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <div className="text-[11px] uppercase tracking-wider text-ink-300 font-semibold">
            Pick the query token
          </div>
          <div className="text-[11px] text-ink-400 font-mono">
            i = {queryIdx}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {tokens.map((t, i) => (
            <button
              key={i}
              onClick={() => setQueryIdx(i)}
              className={[
                "px-3 py-1.5 rounded-md text-sm font-mono border transition-all",
                i === queryIdx
                  ? "bg-pink-500/25 border-pink-400 text-pink-100 shadow-[0_0_0_2px_rgba(244,114,182,0.25)]"
                  : "bg-ink-900/70 border-ink-700 text-ink-300 hover:border-pink-400/40",
              ].join(" ")}
            >
              {i === queryIdx && <span className="mr-1.5 opacity-80">▶</span>}
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Step 1: the query vector */}
      <Step n={1} title="Take the query vector" color={Q_COLOR}>
        <div className="flex items-center gap-3 flex-wrap">
          <TokenChip token={tokens[queryIdx]} index={queryIdx} large />
          <ArrowRight className="w-4 h-4 text-ink-500" />
          <div className="flex-1 min-w-[200px]">
            <HeatRow values={q} accent={Q_COLOR} height={20} />
            <div className="text-[10px] text-ink-400 font-mono mt-1">
              q<sub>{queryIdx}</sub> ∈ ℝ^{D_K}
            </div>
          </div>
        </div>
      </Step>

      {/* Step 2: dot products with every key */}
      <Step n={2} title="Dot product with every key (relevance score)" color={K_COLOR}>
        <div className="text-[11px] text-ink-400 mb-2">
          Higher dot product = key points in the same direction as the query.
        </div>
        <ScoreList
          tokens={tokens}
          values={trace.scores}
          label="qᵢ · kⱼ"
          color={K_COLOR}
          highlight={queryIdx}
          fmt={(v) => v.toFixed(2)}
        />
      </Step>

      {/* Step 3: scale */}
      <Step n={3} title={`Scale ${useScaling ? "by √d_k" : "(currently OFF)"}`} color="#a78bfa">
        <div className="text-[11px] text-ink-400 mb-2">
          Without scaling, dot products grow with{" "}
          <InlineMath math="d_k" /> and push softmax into saturation — gradients
          die. Dividing by <InlineMath math={String.raw`\sqrt{d_k}`} /> keeps
          things stable.
        </div>
        <ScoreList
          tokens={tokens}
          values={trace.scaled}
          label={useScaling ? "qᵢ·kⱼ / √d_k" : "qᵢ·kⱼ (no scale)"}
          color="#a78bfa"
          highlight={queryIdx}
          fmt={(v) => v.toFixed(2)}
        />
      </Step>

      {/* Step 4: softmax */}
      <Step
        n={4}
        title={`Softmax → attention weights${causal ? "  (causal mask applied)" : ""}`}
        color="#fbbf24"
      >
        <div className="text-[11px] text-ink-400 mb-2">
          Probability distribution over tokens — they sum to 1. Bigger weight
          means more influence on the output.
        </div>
        <ScoreList
          tokens={tokens}
          values={trace.weights}
          label="softmax(...)"
          color="#fbbf24"
          highlight={queryIdx}
          fmt={(v) => `${(v * 100).toFixed(1)}%`}
          asPercent
          dimMasked={causal ? queryIdx : undefined}
        />
      </Step>

      {/* Step 5: weighted sum of V */}
      <Step n={5} title="Weighted sum of values → output" color={V_COLOR}>
        <div className="text-[11px] text-ink-400 mb-3">
          The output is just{" "}
          <InlineMath math={String.raw`\sum_j w_j v_j`} /> — each value vector
          contributes in proportion to its weight.
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex flex-col gap-1 min-w-[260px]">
            {trace.V.map((vRow, i) => {
              const w = trace.weights[i];
              const isMasked = causal && i > queryIdx;
              return (
                <div
                  key={i}
                  className={[
                    "grid grid-cols-[4.5rem_3rem_1fr] gap-2 items-center transition-opacity",
                    isMasked ? "opacity-30" : "",
                  ].join(" ")}
                >
                  <span className="text-[11px] font-mono text-ink-300 truncate">
                    {tokens[i]}
                  </span>
                  <span className="text-[11px] font-mono text-amber-300">
                    {(w * 100).toFixed(1)}%
                  </span>
                  <HeatRow values={vRow} accent={V_COLOR} height={10} />
                </div>
              );
            })}
          </div>
          <ArrowRight className="w-4 h-4 text-ink-500" />
          <div className="flex-1 min-w-[200px]">
            <div className="text-[11px] uppercase tracking-wider text-ink-400 font-semibold mb-1">
              output for token "{tokens[queryIdx]}"
            </div>
            <div className="rounded-lg border border-emerald-400/40 bg-emerald-400/[0.06] p-2">
              <HeatRow values={trace.output} accent="#10b981" height={22} />
            </div>
            <div className="text-[10px] text-ink-400 font-mono mt-1">
              z<sub>{queryIdx}</sub> = Σⱼ wⱼ vⱼ ∈ ℝ^{D_K}
            </div>
          </div>
        </div>
      </Step>

      <div className="rounded-xl border border-ink-800/80 bg-ink-950/40 p-4 text-sm leading-relaxed">
        <div className="font-semibold text-white mb-1.5 flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-pink-300" />
          Putting it together
        </div>
        <BlockMath
          math={String.raw`\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V`}
        />
      </div>
    </section>
  );
}

function FullMatrixSection({
  tokens,
  matrix,
  causal,
  queryIdx,
}: {
  tokens: string[];
  matrix: number[][];
  causal: boolean;
  queryIdx: number;
}) {
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const n = matrix.length;
  const cell = 40;
  const labelW = 70;
  const labelH = 70;
  const size = labelW + n * cell;
  const height = labelH + n * cell;
  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Network className="w-3 h-3" />}>The full picture</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        Every query, every key — the attention matrix
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Doing the pipeline above for <em>every</em> query token at once gives a
        full <InlineMath math="n \times n" /> matrix of attention weights. The
        row you picked above is one row of this matrix.
      </p>
      <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4 overflow-auto">
        <svg width={size} height={height} className="block" style={{ maxWidth: "100%" }}>
          {tokens.map((t, j) => (
            <text
              key={`c-${j}`}
              x={labelW + j * cell + cell / 2}
              y={labelH - 8}
              textAnchor="end"
              transform={`rotate(-40, ${labelW + j * cell + cell / 2}, ${labelH - 8})`}
              fontSize={11}
              fill="#c2c7d4"
              fontFamily="JetBrains Mono, monospace"
            >
              {t}
            </text>
          ))}
          {tokens.map((t, i) => (
            <text
              key={`r-${i}`}
              x={labelW - 8}
              y={labelH + i * cell + cell / 2 + 4}
              textAnchor="end"
              fontSize={11}
              fill={i === queryIdx ? Q_COLOR : "#c2c7d4"}
              fontWeight={i === queryIdx ? 700 : 400}
              fontFamily="JetBrains Mono, monospace"
            >
              {t}
            </text>
          ))}
          {matrix.map((row, i) =>
            row.map((v, j) => {
              const masked = causal && j > i;
              const intensity = masked ? 0 : v;
              const hue = 320 - intensity * 130; // pink → cyan
              const isHover = hover?.i === i && hover?.j === j;
              const isInRow = i === queryIdx;
              return (
                <g key={`${i}-${j}`}>
                  <rect
                    x={labelW + j * cell + 2}
                    y={labelH + i * cell + 2}
                    width={cell - 4}
                    height={cell - 4}
                    rx={6}
                    fill={
                      masked
                        ? "rgba(40,44,56,0.5)"
                        : `hsl(${hue}, 70%, ${18 + intensity * 50}%)`
                    }
                    stroke={
                      isHover
                        ? "#fff"
                        : isInRow
                          ? `${Q_COLOR}aa`
                          : "rgba(255,255,255,0.06)"
                    }
                    strokeWidth={isHover ? 1.5 : isInRow ? 1.2 : 1}
                    onMouseEnter={() => setHover({ i, j })}
                    onMouseLeave={() => setHover(null)}
                  />
                  {!masked && (
                    <text
                      x={labelW + j * cell + cell / 2}
                      y={labelH + i * cell + cell / 2 + 4}
                      textAnchor="middle"
                      fontSize={10}
                      fill={intensity > 0.5 ? "#0a0b10" : "#e6e8ee"}
                      fontFamily="JetBrains Mono, monospace"
                      pointerEvents="none"
                    >
                      {v.toFixed(2)}
                    </text>
                  )}
                  {masked && (
                    <text
                      x={labelW + j * cell + cell / 2}
                      y={labelH + i * cell + cell / 2 + 4}
                      textAnchor="middle"
                      fontSize={11}
                      fill="rgba(255,255,255,0.25)"
                      pointerEvents="none"
                    >
                      ×
                    </text>
                  )}
                </g>
              );
            })
          )}
        </svg>
        {hover && (
          <div className="mt-2 text-xs text-ink-300 font-mono">
            A[{hover.i}, {hover.j}] ={" "}
            {matrix[hover.i][hover.j].toFixed(4)} —{" "}
            <span className="text-pink-300">"{tokens[hover.i]}"</span> attends
            to <span className="text-cyan-300">"{tokens[hover.j]}"</span>
          </div>
        )}
      </div>
      <div className="text-[11px] text-ink-400">
        {causal ? (
          <>
            <Eye className="w-3 h-3 inline-block mr-1" />
            <span className="text-ink-200">Causal mask is ON</span> — upper
            triangle masked out. Used by GPT-style decoders.
          </>
        ) : (
          <>
            <Eye className="w-3 h-3 inline-block mr-1" />
            <span className="text-ink-200">Bidirectional</span> — every cell
            populated. Used by BERT-style encoders.
          </>
        )}
      </div>
    </section>
  );
}

function ImportanceSection() {
  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Sparkles className="w-3 h-3" />}>Why three?</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        The importance of separating Q, K, V
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Naive attention could just take dot products between embeddings
        directly. The three-way split is what gives transformers their
        flexibility, expressiveness, and trainability.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InsightCard
          title="Asymmetric attention"
          color={Q_COLOR}
          body={
            <>
              Without separate <InlineMath math="W_Q" /> and{" "}
              <InlineMath math="W_K" />, "A attending to B" would be the same
              as "B attending to A". With them, the relationship is{" "}
              <strong>directional</strong>: a verb can look back at its
              subject without the subject having to look at the verb the same
              way.
            </>
          }
        />
        <InsightCard
          title="Decoupled representation"
          color={V_COLOR}
          body={
            <>
              A token can be <em>matched</em> by its key while contributing{" "}
              <em>different</em> content through its value. The model can use a
              keyword to find a sentence, then return its meaning — two
              different vectors entirely.
            </>
          }
        />
        <InsightCard
          title="Multi-head learning"
          color="#a78bfa"
          body={
            <>
              The same input can be projected into <strong>many</strong>{" "}
              independent (Q, K, V) sets in parallel. Each "head" learns a
              different relation — syntactic, coreference, positional — and
              they're concatenated to form the final output.
            </>
          }
        />
        <InsightCard
          title="Long-range information routing"
          color={K_COLOR}
          body={
            <>
              Unlike RNNs, attention routes information in <strong>O(1)</strong>{" "}
              path length: the query of any token can be matched against the
              key of any other token, no matter how far apart. The Q/K/V
              decoupling makes that routing learnable end-to-end.
            </>
          }
        />
      </div>

      <div className="rounded-xl border border-ink-800/80 bg-gradient-to-br from-pink-500/[0.05] to-cyan-500/[0.05] p-5 mt-2">
        <div className="text-[11px] uppercase tracking-wider text-ink-300 font-semibold mb-2">
          The big takeaway
        </div>
        <p className="text-ink-100 leading-relaxed">
          QKV isn't just notation. It's a{" "}
          <strong>differentiable, content-based, parallel lookup</strong> —
          and it turns out you can stack a few dozen layers of this and get
          GPT, BERT, T5, ViT, AlphaFold, Whisper, and basically every modern
          foundation model.
        </p>
      </div>
    </section>
  );
}

/* ------------------------- Small reusable primitives ------------------------ */

function Eyebrow({
  children,
  icon,
}: {
  children: React.ReactNode;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-pink-300 font-semibold">
      {icon ?? <span className="w-1.5 h-1.5 rounded-full bg-pink-400" />}
      {children}
    </div>
  );
}

function RoleCard({
  color,
  letter,
  name,
  question,
  body,
  icon,
}: {
  color: string;
  letter: string;
  name: string;
  question: string;
  body: string;
  icon: React.ReactNode;
}) {
  return (
    <div
      className="rounded-2xl border p-4 flex flex-col gap-2 relative overflow-hidden"
      style={{
        background: `linear-gradient(180deg, ${color}1a, transparent)`,
        borderColor: `${color}55`,
      }}
    >
      <div className="flex items-center gap-3">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center font-extrabold text-lg"
          style={{ background: `${color}33`, color }}
        >
          {letter}
        </div>
        <div>
          <div className="text-base font-bold">{name}</div>
          <div className="text-[11px] text-ink-300 flex items-center gap-1">
            {icon}
            {question}
          </div>
        </div>
      </div>
      <p className="text-sm text-ink-300 leading-relaxed">{body}</p>
    </div>
  );
}

function AnalogyCard({
  color,
  title,
  body,
}: {
  color: string;
  title: string;
  body: React.ReactNode;
}) {
  return (
    <div
      className="rounded-xl border p-4 flex flex-col gap-2"
      style={{
        background: `${color}0c`,
        borderColor: `${color}55`,
      }}
    >
      <div className="font-semibold text-white">{title}</div>
      <p className="text-sm text-ink-300 leading-relaxed">{body}</p>
    </div>
  );
}

function Step({
  n,
  title,
  color,
  children,
}: {
  n: number;
  title: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4">
      <div className="flex items-center gap-3 mb-3">
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center font-mono text-sm font-bold"
          style={{ background: `${color}33`, color, border: `1px solid ${color}55` }}
        >
          {n}
        </div>
        <div className="text-sm font-semibold text-white">{title}</div>
      </div>
      {children}
    </div>
  );
}

function ProjArrow({
  x1,
  y1,
  x2,
  y2,
  label,
  color,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  color: string;
}) {
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  return (
    <g>
      <path
        d={`M ${x1} ${y1} Q ${x1 + 30} ${my}, ${x2} ${y2}`}
        stroke={color}
        strokeWidth={1.5}
        fill="none"
      />
      <text
        x={mx + 8}
        y={my - 4}
        fontSize={11}
        fill={color}
        fontFamily="JetBrains Mono, monospace"
        fontWeight={600}
      >
        {label}
      </text>
    </g>
  );
}

function OutputBlock({
  x,
  y,
  color,
  letter,
  full,
}: {
  x: number;
  y: number;
  color: string;
  letter: string;
  full: string;
}) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={170}
        height={40}
        rx={10}
        fill={`${color}22`}
        stroke={`${color}88`}
      />
      <text
        x={x + 16}
        y={y + 26}
        fill={color}
        fontSize={14}
        fontWeight={700}
        fontFamily="JetBrains Mono, monospace"
      >
        {letter}
      </text>
      <text
        x={x + 60}
        y={y + 26}
        fill="#c2c7d4"
        fontSize={12}
      >
        ∈ ℝ^dₖ ({full})
      </text>
    </g>
  );
}

function TokenChip({
  token,
  index,
  large,
}: {
  token: string;
  index: number;
  large?: boolean;
}) {
  const color = TOKEN_COLORS[index % TOKEN_COLORS.length];
  return (
    <span
      className={[
        "px-2.5 py-1 rounded-md font-mono border whitespace-pre",
        large ? "text-base" : "text-sm",
      ].join(" ")}
      style={{
        background: `${color}22`,
        borderColor: `${color}88`,
        color: "#fff",
      }}
    >
      {token}
    </span>
  );
}

function HeatRow({
  values,
  height = 14,
  accent,
}: {
  values: number[];
  height?: number;
  accent?: string;
}) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const norm = (v: number) => (max === min ? 0.5 : (v - min) / (max - min));
  return (
    <div
      className="flex rounded-md overflow-hidden border"
      style={{ height, borderColor: accent ? `${accent}55` : "rgba(255,255,255,0.06)" }}
    >
      {values.map((v, i) => {
        const t = norm(v);
        let bg: string;
        if (accent) {
          const a = Math.round(40 + 200 * t);
          bg = `${accent}${a.toString(16).padStart(2, "0")}`;
        } else {
          const r = Math.round(120 + (255 - 120) * t);
          const g = Math.round(220 * (1 - t) + 100 * t * (1 - t) * 2);
          const b = Math.round(255 * (1 - t) + 80 * t);
          bg = `rgb(${r}, ${g}, ${b})`;
        }
        return (
          <div
            key={i}
            style={{ background: bg, width: `${100 / values.length}%` }}
            title={v.toFixed(3)}
          />
        );
      })}
    </div>
  );
}

function ScoreList({
  tokens,
  values,
  label,
  color,
  highlight,
  fmt,
  asPercent,
  dimMasked,
}: {
  tokens: string[];
  values: number[];
  label: string;
  color: string;
  highlight: number;
  fmt: (v: number) => string;
  asPercent?: boolean;
  dimMasked?: number;
}) {
  const finite = values.filter((v) => Number.isFinite(v));
  const maxAbs = Math.max(1e-6, ...finite.map((v) => Math.abs(v)));
  return (
    <div className="flex flex-col gap-1">
      <div className="grid grid-cols-[5.5rem_3.5rem_1fr] gap-3 text-[10px] text-ink-400 font-mono mb-0.5">
        <span>token j</span>
        <span className="text-right">{label}</span>
        <span></span>
      </div>
      {tokens.map((t, j) => {
        const v = values[j];
        const isFinite = Number.isFinite(v);
        const isMasked =
          dimMasked !== undefined && j > dimMasked;
        const widthPct = isFinite
          ? asPercent
            ? Math.max(0, Math.min(1, v)) * 100
            : (Math.abs(v) / maxAbs) * 100
          : 0;
        const isQuery = j === highlight;
        return (
          <div
            key={j}
            className={[
              "grid grid-cols-[5.5rem_3.5rem_1fr] gap-3 items-center transition-opacity",
              isMasked ? "opacity-30" : "",
            ].join(" ")}
          >
            <span
              className={[
                "text-sm font-mono truncate",
                isQuery ? "text-pink-300 font-semibold" : "text-ink-200",
              ].join(" ")}
            >
              {t}
            </span>
            <span className="text-xs font-mono text-right text-ink-200">
              {isFinite ? fmt(v) : "—∞"}
            </span>
            <div className="h-2.5 rounded-full bg-ink-800 overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{
                  background:
                    isFinite && v < 0 && !asPercent
                      ? `linear-gradient(90deg, ${color}66, ${color}22)`
                      : color,
                }}
                initial={false}
                animate={{ width: `${widthPct}%` }}
                transition={{ type: "spring", stiffness: 220, damping: 24 }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function InsightCard({
  title,
  color,
  body,
}: {
  title: string;
  color: string;
  body: React.ReactNode;
}) {
  return (
    <div
      className="rounded-2xl border p-5"
      style={{
        background: `${color}0a`,
        borderColor: `${color}55`,
      }}
    >
      <div className="font-bold text-white mb-1.5">{title}</div>
      <p className="text-sm text-ink-300 leading-relaxed">{body}</p>
    </div>
  );
}

function Toggle({
  on,
  onToggle,
  label,
  hint,
}: {
  on: boolean;
  onToggle: () => void;
  label: string;
  hint?: string;
}) {
  return (
    <button
      onClick={onToggle}
      title={hint}
      className={[
        "flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all",
        on
          ? "bg-pink-500/15 border-pink-400/50 text-pink-100 shadow-[0_0_0_2px_rgba(244,114,182,0.15)]"
          : "bg-ink-900/70 border-ink-700 text-ink-300 hover:border-pink-400/30",
      ].join(" ")}
    >
      {on ? (
        <ToggleRight className="w-4 h-4 text-pink-300" />
      ) : (
        <ToggleLeft className="w-4 h-4" />
      )}
      <span>{label}</span>
    </button>
  );
}
