import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Tex as InlineMath, TexBlock as BlockMath } from "../Tex";
import { Info } from "lucide-react";
import {
  fakeAttention,
  fakeNextTokenDist,
  fakeTokenize,
  positionalEmbed,
  tokenEmbed,
} from "../../lib/gptDemo";
import { gptComponents, type GPTComponentId } from "../../data/gptComponents";

interface Props {
  selected: GPTComponentId;
  prompt: string;
  numHeads: number;
  dModel: number;
  embedded?: boolean;
}

const TOKEN_COLORS = [
  "#7c5cff",
  "#22d3ee",
  "#f472b6",
  "#fbbf24",
  "#34d399",
  "#f87171",
  "#a78bfa",
  "#60a5fa",
  "#fb7185",
  "#facc15",
];

export default function GPTDetail({
  selected,
  prompt,
  numHeads,
  dModel,
  embedded,
}: Props) {
  const meta = gptComponents[selected];
  const tokens = useMemo(() => fakeTokenize(prompt), [prompt]);

  const containerClass = embedded
    ? "flex flex-col gap-4"
    : "card p-5 lg:p-6 flex flex-col gap-5";

  return (
    <div className={containerClass}>
      <header className="flex flex-col gap-1.5">
        <div className="text-[11px] uppercase tracking-[0.14em] text-accent-soft font-semibold">
          {meta.group}
        </div>
        <h2 className={embedded ? "text-xl font-bold tracking-tight" : "text-2xl font-bold tracking-tight"}>
          {meta.full}
        </h2>
        <p className="text-ink-300 text-sm">{meta.tagline}</p>
      </header>

      <div className="grid gap-4">
        <ComponentVisual
          selected={selected}
          tokens={tokens}
          prompt={prompt}
          numHeads={numHeads}
          dModel={dModel}
        />
      </div>
    </div>
  );
}

function ComponentVisual({
  selected,
  tokens,
  prompt,
  numHeads,
  dModel,
}: {
  selected: GPTComponentId;
  tokens: { token: string; id: number }[];
  prompt: string;
  numHeads: number;
  dModel: number;
}) {
  switch (selected) {
    case "input":
      return <InputVisual prompt={prompt} />;
    case "tokenization":
      return <TokenizationVisual tokens={tokens} />;
    case "tokenEmbedding":
      return <TokenEmbeddingVisual tokens={tokens} d={dModel} />;
    case "positionalEmbedding":
      return <PositionalEmbeddingVisual tokens={tokens} d={dModel} />;
    case "embeddingSum":
      return <EmbeddingSumVisual tokens={tokens} d={dModel} />;
    case "layerNorm1":
    case "layerNorm2":
    case "finalLayerNorm":
      return <LayerNormVisual which={selected} />;
    case "selfAttention":
      return <AttentionVisual tokens={tokens} />;
    case "multiHead":
      return <MultiHeadVisual tokens={tokens} numHeads={numHeads} />;
    case "residual1":
    case "residual2":
      return <ResidualVisual which={selected} />;
    case "ffn":
      return <FFNVisual dModel={dModel} />;
    case "lmHead":
      return <LMHeadVisual dModel={dModel} />;
    case "sampling":
      return <SamplingVisual prompt={prompt} />;
    default:
      return null;
  }
}

/* ------------------------- Reusable presentational ------------------------- */

function Section({
  title,
  children,
  hint,
}: {
  title?: string;
  children: React.ReactNode;
  hint?: string;
}) {
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4">
      {title && (
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs uppercase tracking-wider text-ink-300 font-semibold">
            {title}
          </h4>
          {hint && (
            <span className="text-[11px] text-ink-400 flex items-center gap-1">
              <Info className="w-3 h-3" />
              {hint}
            </span>
          )}
        </div>
      )}
      {children}
    </div>
  );
}

function Definition({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-accent/[0.06] border border-accent/30 p-4 text-sm text-ink-200 leading-relaxed">
      {children}
    </div>
  );
}

function HeatRow({ values, height = 14 }: { values: number[]; height?: number }) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const norm = (v: number) => (max === min ? 0.5 : (v - min) / (max - min));
  return (
    <div
      className="flex rounded-md overflow-hidden border border-ink-800"
      style={{ height }}
    >
      {values.map((v, i) => {
        const t = norm(v);
        // Diverging colormap (cool → warm)
        const r = Math.round(124 + (255 - 124) * t);
        const g = Math.round(92 * (1 - t) + 200 * t * (1 - t) * 2);
        const b = Math.round(255 * (1 - t) + 60 * t);
        return (
          <div
            key={i}
            style={{ background: `rgb(${r}, ${g}, ${b})`, width: `${100 / values.length}%` }}
            title={v.toFixed(3)}
          />
        );
      })}
    </div>
  );
}

function TokenChip({
  token,
  index,
  showId,
  id,
}: {
  token: string;
  index: number;
  showId?: boolean;
  id?: number;
}) {
  const color = TOKEN_COLORS[index % TOKEN_COLORS.length];
  return (
    <div
      className="flex flex-col items-center gap-1"
      style={{ minWidth: "fit-content" }}
    >
      <span
        className="px-2.5 py-1 rounded-md font-mono text-sm border whitespace-pre"
        style={{
          background: `${color}22`,
          borderColor: `${color}55`,
          color: "#fff",
        }}
      >
        {token.replace(/ /g, "\u2423")}
      </span>
      {showId && (
        <span className="text-[10px] font-mono text-ink-400">#{id}</span>
      )}
    </div>
  );
}

/* ---------------------------- Per-component views --------------------------- */

function InputVisual({ prompt }: { prompt: string }) {
  return (
    <>
      <Definition>
        The model's input is just <strong>text</strong> — a prompt that we want
        the language model to continue. Everything that follows is what GPT
        does to turn that string into a probability distribution over the next
        token.
      </Definition>
      <Section title="Current prompt">
        <div className="rounded-lg bg-ink-900/80 border border-ink-700 px-4 py-3 font-mono text-base">
          “{prompt}”
        </div>
        <p className="text-xs text-ink-400 mt-2">
          Try editing the prompt above the diagram to see how every stage
          reacts.
        </p>
      </Section>
    </>
  );
}

function TokenizationVisual({
  tokens,
}: {
  tokens: { token: string; id: number }[];
}) {
  return (
    <>
      <Definition>
        Text is split into <strong>tokens</strong> — subword chunks chosen by a
        learned vocabulary (Byte-Pair Encoding in GPT-2/3). Each token is
        mapped to an integer <em>token ID</em> in <InlineMath math="[0, V)" />,
        where <InlineMath math="V" /> is the vocabulary size (50,257 for
        GPT-2).
      </Definition>
      <Section title="Tokens" hint="Hover for token IDs">
        <div className="flex flex-wrap gap-2.5">
          {tokens.length === 0 ? (
            <span className="text-ink-400 text-sm">No tokens yet — type something above.</span>
          ) : (
            tokens.map((t, i) => (
              <TokenChip key={i} token={t.token} index={i} showId id={t.id} />
            ))
          )}
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
          <Metric label="# tokens" value={tokens.length.toString()} />
          <Metric label="Vocabulary V" value="50,257" />
        </div>
      </Section>
      <Section title="In code">
        <pre className="text-[12px] font-mono text-ink-200 overflow-x-auto">{`tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
ids = tokenizer.encode("The cat sat on the")
# → [464, 3797, 3332, 319, 262]`}</pre>
      </Section>
    </>
  );
}

function TokenEmbeddingVisual({
  tokens,
  d,
}: {
  tokens: { token: string; id: number }[];
  d: number;
}) {
  const rowDim = Math.min(48, d);
  return (
    <>
      <Definition>
        Every token ID indexes into a learned matrix{" "}
        <InlineMath math={String.raw`W_E \in \mathbb{R}^{V \times d_{model}}`} />, returning
        a vector of length <InlineMath math={String.raw`d_{model}`} />. This is the
        token's dense representation.
      </Definition>
      <Section
        title="Embedding lookup"
        hint={`each row = ${rowDim}-d vector`}
      >
        <div className="rounded-xl overflow-hidden border border-ink-800">
          <table className="w-full text-sm">
            <tbody>
              {tokens.map((t, i) => (
                <tr key={i} className="border-t border-ink-800/80 first:border-t-0">
                  <td className="py-2 pl-3 pr-2 w-28 align-middle">
                    <TokenChip token={t.token} index={i} />
                  </td>
                  <td className="py-2 pr-3">
                    <HeatRow values={tokenEmbed(t.id, rowDim)} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <BlockMath math={String.raw`E_t = W_E[\text{id}(t)] \in \mathbb{R}^{d_{model}}`} />
    </>
  );
}

function PositionalEmbeddingVisual({
  tokens,
  d,
}: {
  tokens: { token: string; id: number }[];
  d: number;
}) {
  const rowDim = Math.min(48, d);
  return (
    <>
      <Definition>
        Self-attention is order-invariant by itself, so we explicitly add a{" "}
        <strong>positional embedding</strong>{" "}
        <InlineMath math={String.raw`P_p \in \mathbb{R}^{d_{model}}`} /> that depends on
        the position <InlineMath math="p" />. GPT-2 learns these; the original
        transformer used fixed sinusoids:
      </Definition>
      <Section title="Positional embeddings" hint="sinusoidal, by position">
        <div className="rounded-xl overflow-hidden border border-ink-800">
          <table className="w-full text-sm">
            <tbody>
              {tokens.map((_, p) => (
                <tr key={p} className="border-t border-ink-800/80 first:border-t-0">
                  <td className="py-2 pl-3 pr-2 w-20 font-mono text-xs text-ink-300">
                    pos = {p}
                  </td>
                  <td className="py-2 pr-3">
                    <HeatRow values={positionalEmbed(p, rowDim)} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <BlockMath
        math={String.raw`P_{p, 2k} = \sin\!\left(\tfrac{p}{10000^{2k/d}}\right),\quad P_{p, 2k+1} = \cos\!\left(\tfrac{p}{10000^{2k/d}}\right)`}
      />
    </>
  );
}

function EmbeddingSumVisual({
  tokens,
  d,
}: {
  tokens: { token: string; id: number }[];
  d: number;
}) {
  const rowDim = Math.min(48, d);
  return (
    <>
      <Definition>
        Token and positional vectors are simply <strong>added</strong>{" "}
        element-wise. The result is the actual sequence of vectors{" "}
        <InlineMath math={String.raw`X \in \mathbb{R}^{n \times d_{model}}`} /> that enters
        the first transformer block.
      </Definition>
      <Section title="E + P = X">
        <div className="rounded-xl overflow-hidden border border-ink-800">
          {tokens.map((t, i) => {
            const tEmb = tokenEmbed(t.id, rowDim);
            const pEmb = positionalEmbed(i, rowDim);
            const sum = tEmb.map((v, k) => v + pEmb[k]);
            return (
              <div
                key={i}
                className="grid grid-cols-[5rem_1fr] sm:grid-cols-[6rem_1fr] gap-2 items-center border-t border-ink-800/80 first:border-t-0 py-2 pl-3 pr-3"
              >
                <TokenChip token={t.token} index={i} />
                <div className="flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-ink-400 w-6">E</span>
                    <div className="flex-1"><HeatRow values={tEmb} height={10} /></div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-ink-400 w-6">+ P</span>
                    <div className="flex-1"><HeatRow values={pEmb} height={10} /></div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-accent-soft w-6">= X</span>
                    <div className="flex-1"><HeatRow values={sum} height={12} /></div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </Section>
      <BlockMath math={String.raw`X = W_E[\text{ids}] + W_P[\text{positions}]`} />
    </>
  );
}

function LayerNormVisual({ which }: { which: GPTComponentId }) {
  const raw = useMemo(
    () =>
      Array.from({ length: 24 }, (_, i) =>
        3 * Math.sin(i * 0.7) + 2 * Math.cos(i * 1.3) + (i % 5)
      ),
    []
  );
  const mean = raw.reduce((a, b) => a + b, 0) / raw.length;
  const variance =
    raw.reduce((a, b) => a + (b - mean) ** 2, 0) / raw.length;
  const eps = 1e-5;
  const normed = raw.map((v) => (v - mean) / Math.sqrt(variance + eps));

  return (
    <>
      <Definition>
        LayerNorm normalizes the features of <em>each token independently</em>{" "}
        (across the <InlineMath math={String.raw`d_{model}`} /> dimensions). It stabilizes
        training and is applied <strong>before</strong> attention and FFN in
        the modern "pre-norm" variant used by GPT-2+.
      </Definition>
      <Section title="Single token's features" hint="before vs after">
        <div className="flex flex-col gap-3">
          <div>
            <div className="text-[11px] text-ink-400 mb-1">x (raw)</div>
            <HeatRow values={raw} />
          </div>
          <div>
            <div className="text-[11px] text-ink-400 mb-1">
              y = γ · (x − μ) / √(σ² + ε) + β
            </div>
            <HeatRow values={normed} />
          </div>
          <div className="grid grid-cols-3 gap-2 text-sm">
            <Metric label="μ before" value={mean.toFixed(3)} />
            <Metric label="σ² before" value={variance.toFixed(3)} />
            <Metric label="‖x‖ scale" value="≈ stable" />
          </div>
        </div>
      </Section>
      <BlockMath
        math={String.raw`\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \quad \mu, \sigma^2 \text{ computed across } d_{model}`}
      />
      <div className="text-[11px] text-ink-400">
        ID: <span className="font-mono">{which}</span>
      </div>
    </>
  );
}

function AttentionVisual({
  tokens,
}: {
  tokens: { token: string; id: number }[];
}) {
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const n = Math.max(tokens.length, 2);
  const matrix = useMemo(() => fakeAttention(n, 0), [n]);

  return (
    <>
      <Definition>
        Self-attention lets each token build a context-aware representation by
        looking at <em>other</em> tokens. For every pair (i, j) we compute a
        score from the query of <InlineMath math="i" /> and the key of{" "}
        <InlineMath math="j" />, scale it by{" "}
        <InlineMath math={String.raw`\sqrt{d_k}`} />, mask out the future, and
        softmax. The output is a weighted sum of value vectors.
      </Definition>

      <Section
        title="Attention matrix"
        hint="row = query token, col = key token"
      >
        <AttentionHeatmap
          tokens={tokens.map((t) => t.token)}
          matrix={matrix}
          onHover={setHover}
          hover={hover}
        />
        <p className="text-xs text-ink-400 mt-3 leading-relaxed">
          The <span className="text-ink-200">upper triangle is masked</span>{" "}
          (causal): token <InlineMath math="i" /> may not look at future
          tokens. Each row sums to 1 after softmax.
        </p>
      </Section>

      <Section title="Formulation">
        <BlockMath
          math={String.raw`\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V`}
        />
        <div className="text-xs text-ink-300 leading-relaxed">
          <InlineMath math={String.raw`Q = X W_Q,\; K = X W_K,\; V = X W_V`} />.
          The causal mask <InlineMath math={String.raw`M_{ij} = -\infty`} /> for{" "}
          <InlineMath math="j > i" /> guarantees autoregressive behavior.
        </div>
      </Section>
    </>
  );
}

function AttentionHeatmap({
  tokens,
  matrix,
  onHover,
  hover,
}: {
  tokens: string[];
  matrix: number[][];
  onHover?: (h: { i: number; j: number } | null) => void;
  hover?: { i: number; j: number } | null;
}) {
  const n = matrix.length;
  const cell = 36;
  const labelW = 60;
  const labelH = 60;
  const width = labelW + n * cell;
  const height = labelH + n * cell;

  return (
    <div className="overflow-auto">
      <svg
        width={width}
        height={height}
        className="block"
        style={{ maxWidth: "100%" }}
      >
        {/* Column labels */}
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
            {t.trim() || "·"}
          </text>
        ))}
        {/* Row labels */}
        {tokens.map((t, i) => (
          <text
            key={`r-${i}`}
            x={labelW - 8}
            y={labelH + i * cell + cell / 2 + 4}
            textAnchor="end"
            fontSize={11}
            fill="#c2c7d4"
            fontFamily="JetBrains Mono, monospace"
          >
            {t.trim() || "·"}
          </text>
        ))}
        {/* Cells */}
        {matrix.map((row, i) =>
          row.map((v, j) => {
            const masked = j > i;
            const intensity = masked ? 0 : v;
            const hue = 250 - intensity * 90;
            const sat = masked ? 0 : 70;
            const light = masked ? 14 : 18 + intensity * 50;
            const isHover = hover?.i === i && hover?.j === j;
            return (
              <g key={`${i}-${j}`}>
                <rect
                  x={labelW + j * cell + 2}
                  y={labelH + i * cell + 2}
                  width={cell - 4}
                  height={cell - 4}
                  rx={5}
                  fill={
                    masked
                      ? "rgba(40, 44, 56, 0.5)"
                      : `hsl(${hue}, ${sat}%, ${light}%)`
                  }
                  stroke={
                    isHover ? "#fff" : masked ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.06)"
                  }
                  strokeWidth={isHover ? 1.5 : 1}
                  onMouseEnter={() => onHover?.({ i, j })}
                  onMouseLeave={() => onHover?.(null)}
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
          A[{hover.i}, {hover.j}] = {matrix[hover.i][hover.j].toFixed(4)} —{" "}
          token <span className="text-accent-soft">"{tokens[hover.i]?.trim()}"</span>{" "}
          attends to <span className="text-teal-300">"{tokens[hover.j]?.trim()}"</span>
        </div>
      )}
    </div>
  );
}

function MultiHeadVisual({
  tokens,
  numHeads,
}: {
  tokens: { token: string; id: number }[];
  numHeads: number;
}) {
  const n = Math.max(tokens.length, 2);
  const heads = Math.min(numHeads, 6);
  const matrices = useMemo(
    () => Array.from({ length: heads }, (_, h) => fakeAttention(n, h + 1)),
    [n, heads]
  );

  return (
    <>
      <Definition>
        Instead of one big attention, GPT splits{" "}
        <InlineMath math={String.raw`d_{model}`} /> into <InlineMath math="h" /> smaller{" "}
        <strong>heads</strong>. Each head learns its own attention pattern —
        some focus on previous tokens, others on the subject, others on
        syntactic structure — and they are concatenated and projected back to{" "}
        <InlineMath math={String.raw`d_{model}`} />.
      </Definition>
      <Section title={`${heads} of ${numHeads} heads shown`}>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {matrices.map((m, h) => (
            <MiniHead key={h} tokens={tokens.map((t) => t.token)} matrix={m} index={h} />
          ))}
        </div>
        {heads < numHeads && (
          <p className="text-[11px] text-ink-400 mt-3">
            … and {numHeads - heads} more heads not shown here.
          </p>
        )}
      </Section>
      <BlockMath
        math={String.raw`\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O`}
      />
    </>
  );
}

function MiniHead({
  matrix,
  index,
}: {
  tokens: string[];
  matrix: number[][];
  index: number;
}) {
  const n = matrix.length;
  const cell = 20;
  const size = n * cell;
  return (
    <div className="rounded-xl bg-ink-900/70 border border-ink-800 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] text-ink-300 font-mono">head #{index + 1}</span>
        <span
          className="w-2 h-2 rounded-full"
          style={{ background: TOKEN_COLORS[index % TOKEN_COLORS.length] }}
        />
      </div>
      <svg width={size} height={size} style={{ maxWidth: "100%" }} viewBox={`0 0 ${size} ${size}`}>
        {matrix.map((row, i) =>
          row.map((v, j) => {
            const masked = j > i;
            const hue = 250 - v * 90;
            return (
              <rect
                key={`${i}-${j}`}
                x={j * cell + 1}
                y={i * cell + 1}
                width={cell - 2}
                height={cell - 2}
                rx={2}
                fill={
                  masked
                    ? "rgba(40, 44, 56, 0.4)"
                    : `hsl(${hue}, 70%, ${18 + v * 50}%)`
                }
              />
            );
          })
        )}
      </svg>
    </div>
  );
}

function ResidualVisual({ which }: { which: GPTComponentId }) {
  return (
    <>
      <Definition>
        Each sublayer adds its <em>output</em> to its <em>input</em>. This
        residual highway makes optimization easy and lets gradients flow
        straight from the loss back to early tokens.
      </Definition>
      <Section title="Schematic">
        <svg viewBox="0 0 360 160" className="w-full">
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
              <path d="M0,0 L0,6 L9,3 z" fill="#a48bff" />
            </marker>
          </defs>
          <rect x="20" y="60" width="80" height="40" rx="10" fill="#1f2330" stroke="#7c5cff" />
          <text x="60" y="85" fill="#fff" fontSize="12" textAnchor="middle">x</text>

          <rect x="140" y="20" width="120" height="40" rx="10" fill="#161922" stroke="#22d3ee" />
          <text x="200" y="45" fill="#fff" fontSize="12" textAnchor="middle">sublayer(x)</text>

          <path d="M100 80 H 200 V 60" stroke="#a48bff" strokeWidth="1.5" fill="none" markerEnd="url(#arrow)" />
          <path d="M100 80 H 320" stroke="#a48bff" strokeDasharray="4 4" strokeWidth="1.5" fill="none" />
          <path d="M260 40 H 320 V 80" stroke="#22d3ee" strokeWidth="1.5" fill="none" />
          <circle cx="320" cy="80" r="14" fill="#0f1117" stroke="#fff" />
          <text x="320" y="84" fill="#fff" fontSize="14" textAnchor="middle">+</text>
          <path d="M320 94 V 140" stroke="#a48bff" strokeWidth="1.5" fill="none" markerEnd="url(#arrow)" />
          <text x="340" y="120" fill="#a48bff" fontSize="11">y = x + sublayer(x)</text>
        </svg>
      </Section>
      <BlockMath math={String.raw`y = x + \text{Sublayer}(\text{LN}(x))`} />
      <div className="text-[11px] text-ink-400">
        ID: <span className="font-mono">{which}</span>
      </div>
    </>
  );
}

function FFNVisual({ dModel }: { dModel: number }) {
  const inner = dModel * 4;
  return (
    <>
      <Definition>
        After attention mixes tokens, the feed-forward network mixes{" "}
        <em>features</em> per token. It's two linear layers with a non-linearity
        (GELU in GPT) and an expansion factor of <strong>4×</strong> in
        between.
      </Definition>
      <Section title="Shape transformation">
        <div className="flex items-center gap-3 justify-center text-center py-2">
          <ShapeBlock label="d_model" value={dModel} color="#7c5cff" />
          <Arrowy label="Linear" />
          <ShapeBlock label="4 · d_model" value={inner} color="#22d3ee" big />
          <Arrowy label="GELU" />
          <ShapeBlock label="4 · d_model" value={inner} color="#22d3ee" big />
          <Arrowy label="Linear" />
          <ShapeBlock label="d_model" value={dModel} color="#7c5cff" />
        </div>
      </Section>
      <BlockMath math={String.raw`\text{FFN}(x) = \text{GELU}(x W_1 + b_1)\, W_2 + b_2`} />
      <div className="grid grid-cols-2 gap-3 text-sm">
        <Metric label="Parameters per FFN" value={`${(2 * dModel * inner).toLocaleString()}`} />
        <Metric label="Activation" value="GELU" />
      </div>
    </>
  );
}

function ShapeBlock({
  label,
  value,
  color,
  big,
}: {
  label: string;
  value: number;
  color: string;
  big?: boolean;
}) {
  return (
    <div
      className={[
        "rounded-xl border px-3 py-2.5 flex flex-col items-center justify-center",
        big ? "min-w-[110px]" : "min-w-[90px]",
      ].join(" ")}
      style={{ background: `${color}1a`, borderColor: `${color}55` }}
    >
      <div className="text-[10px] uppercase tracking-wider text-ink-300">
        {label}
      </div>
      <div className="font-mono text-base text-white">{value}</div>
    </div>
  );
}

function Arrowy({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center text-ink-400">
      <span className="text-[10px]">{label}</span>
      <svg width="36" height="14">
        <line x1="0" y1="7" x2="32" y2="7" stroke="#5b6478" strokeWidth="1.5" />
        <polygon points="30,3 36,7 30,11" fill="#5b6478" />
      </svg>
    </div>
  );
}

function LMHeadVisual({ dModel }: { dModel: number }) {
  const V = 50257;
  return (
    <>
      <Definition>
        The final hidden state of each token is projected to vocabulary{" "}
        <strong>logits</strong> with a linear layer (often weight-tied to the
        token embedding matrix). This gives a score for every token in the
        vocabulary.
      </Definition>
      <Section title="Shape">
        <div className="flex items-center gap-3 justify-center py-3">
          <ShapeBlock label="hidden" value={dModel} color="#7c5cff" />
          <Arrowy label={`W ∈ ℝ^${V}×${dModel}`} />
          <ShapeBlock label="logits" value={V} color="#22d3ee" big />
        </div>
      </Section>
      <BlockMath math={String.raw`\text{logits} = h_T \cdot W_E^\top \in \mathbb{R}^{V}`} />
      <div className="text-xs text-ink-400">
        In GPT-2/3, <InlineMath math="W_E" /> is reused (weight tying), saving{" "}
        {(V * dModel).toLocaleString()} parameters.
      </div>
    </>
  );
}

function SamplingVisual({ prompt }: { prompt: string }) {
  const [T, setT] = useState(1);
  const dist = useMemo(() => fakeNextTokenDist(prompt, T), [prompt, T]);
  const total = dist.reduce((a, b) => a + b.prob, 0);
  return (
    <>
      <Definition>
        Logits go through <strong>softmax</strong> to become a probability
        distribution over the vocabulary. We then <em>sample</em> from it —
        with temperature, top-k, or nucleus (top-p) — to pick the next token.
      </Definition>
      <Section title="Next-token distribution (top 10)">
        <div className="flex flex-col gap-2">
          {dist.map((d, i) => (
            <div key={i} className="grid grid-cols-[5.5rem_1fr_3.5rem] items-center gap-3">
              <span className="font-mono text-sm">
                {i + 1}. {d.token}
              </span>
              <div className="h-2.5 rounded-full bg-ink-800 overflow-hidden">
                <motion.div
                  className="h-full"
                  style={{
                    background: `linear-gradient(90deg, ${TOKEN_COLORS[i % TOKEN_COLORS.length]}, #22d3ee)`,
                  }}
                  initial={false}
                  animate={{ width: `${(d.prob / dist[0].prob) * 100}%` }}
                  transition={{ type: "spring", stiffness: 200, damping: 24 }}
                />
              </div>
              <span className="font-mono text-xs text-ink-300 text-right">
                {(d.prob * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </Section>
      <Section title="Temperature">
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={0.1}
            max={3}
            step={0.01}
            value={T}
            onChange={(e) => setT(parseFloat(e.target.value))}
          />
          <span className="font-mono text-sm w-16 text-right">T = {T.toFixed(2)}</span>
        </div>
        <p className="text-[11px] text-ink-400 mt-2">
          Lower T → more deterministic (peaky). Higher T → more random (flat).
          Same machinery as the Softmax visualization in the Functions tab.
        </p>
      </Section>
      <BlockMath math={String.raw`p_i = \frac{\exp(\text{logits}_i / T)}{\sum_j \exp(\text{logits}_j / T)}`} />
      <div className="text-[11px] text-ink-400">Σ probs ≈ {total.toFixed(3)}</div>
    </>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-ink-900/60 border border-ink-800 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-ink-400">
        {label}
      </div>
      <div className="font-mono text-sm text-white">{value}</div>
    </div>
  );
}
