import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Info } from "lucide-react";
import { Tex as InlineMath, TexBlock as BlockMath } from "../Tex";
import {
  bertAttention,
  bertPositionEmbed,
  bertSegmentEmbed,
  bertTokenEmbed,
  bertTokenize,
  mlmPredict,
  nspPredict,
  type BertToken,
} from "../../lib/bertDemo";
import { bertComponents, type BertComponentId } from "../../data/bertComponents";

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
const SPECIAL_COLOR = "#fb923c";

interface Props {
  selected: BertComponentId;
  textA: string;
  textB: string;
  numHeads: number;
  dModel: number;
  maskedIndex: number | null;
  embedded?: boolean;
}

export default function BertDetail({
  selected,
  textA,
  textB,
  numHeads,
  dModel,
  maskedIndex,
  embedded,
}: Props) {
  const meta = bertComponents[selected];
  const tokens = useMemo(
    () =>
      bertTokenize(textA, textB, {
        maskIndices: maskedIndex !== null ? [maskedIndex] : undefined,
      }),
    [textA, textB, maskedIndex]
  );

  const container = embedded
    ? "flex flex-col gap-4"
    : "card p-5 lg:p-6 flex flex-col gap-5";

  return (
    <div className={container}>
      <header className="flex flex-col gap-1.5">
        <div className="text-[11px] uppercase tracking-[0.14em] text-cyan-300 font-semibold">
          {meta.group}
        </div>
        <h2
          className={
            embedded
              ? "text-xl font-bold tracking-tight"
              : "text-2xl font-bold tracking-tight"
          }
        >
          {meta.full}
        </h2>
        <p className="text-ink-300 text-sm">{meta.tagline}</p>
      </header>

      <div className="grid gap-4">
        <ComponentVisual
          selected={selected}
          tokens={tokens}
          textA={textA}
          textB={textB}
          numHeads={numHeads}
          dModel={dModel}
          maskedIndex={maskedIndex}
        />
      </div>
    </div>
  );
}

function ComponentVisual({
  selected,
  tokens,
  textA,
  textB,
  numHeads,
  dModel,
  maskedIndex,
}: {
  selected: BertComponentId;
  tokens: BertToken[];
  textA: string;
  textB: string;
  numHeads: number;
  dModel: number;
  maskedIndex: number | null;
}) {
  switch (selected) {
    case "input":
      return <InputVisual textA={textA} textB={textB} />;
    case "tokenization":
      return <TokenizationVisual tokens={tokens} />;
    case "tokenEmbedding":
      return <TokenEmbeddingVisual tokens={tokens} d={dModel} />;
    case "positionalEmbedding":
      return <PositionalEmbeddingVisual tokens={tokens} d={dModel} />;
    case "segmentEmbedding":
      return <SegmentEmbeddingVisual tokens={tokens} d={dModel} />;
    case "embeddingSum":
      return <EmbeddingSumVisual tokens={tokens} d={dModel} />;
    case "embedLayerNorm":
    case "layerNorm1":
    case "layerNorm2":
      return <LayerNormVisual which={selected} />;
    case "selfAttention":
      return <BidirectionalAttentionVisual tokens={tokens} />;
    case "multiHead":
      return <MultiHeadVisual tokens={tokens} numHeads={numHeads} />;
    case "ffn":
      return <FFNVisual dModel={dModel} />;
    case "residual1":
    case "residual2":
      return <ResidualVisual which={selected} />;
    case "pooler":
      return <PoolerVisual />;
    case "nspHead":
      return <NSPVisual textA={textA} textB={textB} />;
    case "mlmHead":
      return (
        <MLMVisual
          tokens={tokens}
          maskedIndex={maskedIndex}
        />
      );
    default:
      return null;
  }
}

/* ------------------------------ shared visuals ----------------------------- */

function Section({
  title,
  hint,
  children,
}: {
  title?: string;
  hint?: string;
  children: React.ReactNode;
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
    <div className="rounded-xl bg-cyan-500/[0.06] border border-cyan-500/30 p-4 text-sm text-ink-200 leading-relaxed">
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
        const r = Math.round(120 + (255 - 120) * t);
        const g = Math.round(220 * (1 - t) + 100 * t * (1 - t) * 2);
        const b = Math.round(255 * (1 - t) + 80 * t);
        return (
          <div
            key={i}
            style={{
              background: `rgb(${r}, ${g}, ${b})`,
              width: `${100 / values.length}%`,
            }}
            title={v.toFixed(3)}
          />
        );
      })}
    </div>
  );
}

function TokenChip({ tok, index }: { tok: BertToken; index: number }) {
  const color = tok.isSpecial
    ? SPECIAL_COLOR
    : TOKEN_COLORS[index % TOKEN_COLORS.length];
  const seg = tok.segment;
  return (
    <div className="flex flex-col items-center gap-1" style={{ minWidth: "fit-content" }}>
      <span
        className="px-2.5 py-1 rounded-md font-mono text-sm border whitespace-pre"
        style={{
          background: `${color}22`,
          borderColor: `${color}88`,
          color: "#fff",
        }}
      >
        {tok.token}
      </span>
      <span className="text-[9px] font-mono text-ink-500">
        #{tok.id} · seg {seg}
      </span>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-ink-900/60 border border-ink-800 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-ink-400">{label}</div>
      <div className="font-mono text-sm text-white">{value}</div>
    </div>
  );
}

/* ----------------------------- Per-component ----------------------------- */

function InputVisual({ textA, textB }: { textA: string; textB: string }) {
  return (
    <>
      <Definition>
        BERT consumes a sentence or a <strong>sentence pair</strong>. It wraps
        the input with special tokens: <code className="font-mono">[CLS]</code>{" "}
        at the start (used as a sequence-level summary) and{" "}
        <code className="font-mono">[SEP]</code> between/after sentences. Every
        token also gets a <em>segment ID</em>: 0 for sentence A, 1 for sentence
        B.
      </Definition>
      <Section title="Sentence A">
        <div className="rounded-lg bg-ink-900/80 border border-ink-700 px-3 py-2 font-mono text-sm">
          {textA || "—"}
        </div>
      </Section>
      <Section title="Sentence B (optional, for NSP)">
        <div className="rounded-lg bg-ink-900/80 border border-ink-700 px-3 py-2 font-mono text-sm">
          {textB || "—"}
        </div>
      </Section>
      <div className="text-xs text-ink-400">
        Resulting structure:
        <span className="font-mono text-ink-200"> [CLS] A [SEP] B [SEP]</span>
      </div>
    </>
  );
}

function TokenizationVisual({ tokens }: { tokens: BertToken[] }) {
  return (
    <>
      <Definition>
        BERT uses <strong>WordPiece</strong> tokenization. Frequent words stay
        whole; rarer words are split into subword units. Continuation pieces
        get a <code className="font-mono">##</code> prefix
        (e.g. <code className="font-mono">peace</code> +{" "}
        <code className="font-mono">##fully</code>). Vocabulary size ≈ 30K.
      </Definition>
      <Section title="Tokens" hint="orange = special token">
        <div className="flex flex-wrap gap-2">
          {tokens.map((t, i) => (
            <TokenChip key={i} tok={t} index={i} />
          ))}
        </div>
        <div className="mt-4 grid grid-cols-3 gap-2 text-sm">
          <Metric label="# tokens" value={tokens.length.toString()} />
          <Metric
            label="Special"
            value={tokens.filter((t) => t.isSpecial).length.toString()}
          />
          <Metric
            label="Continuations"
            value={tokens.filter((t) => t.isContinuation).length.toString()}
          />
        </div>
      </Section>
    </>
  );
}

function TokenEmbeddingVisual({
  tokens,
  d,
}: {
  tokens: BertToken[];
  d: number;
}) {
  const rowDim = Math.min(48, d);
  return (
    <>
      <Definition>
        Each token ID indexes into a learned matrix{" "}
        <InlineMath math={String.raw`W_E \in \mathbb{R}^{V \times d_{model}}`} />.
        Same as GPT — what differs is the vocabulary (WordPiece) and that
        special tokens have their own learned vectors.
      </Definition>
      <Section title="Embedding rows" hint={`each row = ${rowDim}-d`}>
        <div className="rounded-xl overflow-hidden border border-ink-800">
          <table className="w-full text-sm">
            <tbody>
              {tokens.map((t, i) => (
                <tr key={i} className="border-t border-ink-800/80 first:border-t-0">
                  <td className="py-2 pl-3 pr-2 w-32 align-middle">
                    <TokenChip tok={t} index={i} />
                  </td>
                  <td className="py-2 pr-3">
                    <HeatRow values={bertTokenEmbed(t.id, rowDim)} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </>
  );
}

function PositionalEmbeddingVisual({
  tokens,
  d,
}: {
  tokens: BertToken[];
  d: number;
}) {
  const rowDim = Math.min(48, d);
  return (
    <>
      <Definition>
        Unlike the original Transformer's sinusoidal embeddings, BERT uses{" "}
        <strong>learned positional embeddings</strong>. The model has a fixed
        maximum length (typically 512); positions beyond that are not allowed.
      </Definition>
      <Section title="Position embeddings" hint="learned vectors">
        <div className="rounded-xl overflow-hidden border border-ink-800">
          <table className="w-full text-sm">
            <tbody>
              {tokens.map((_, p) => (
                <tr key={p} className="border-t border-ink-800/80 first:border-t-0">
                  <td className="py-2 pl-3 pr-2 w-20 font-mono text-xs text-ink-300">
                    pos = {p}
                  </td>
                  <td className="py-2 pr-3">
                    <HeatRow values={bertPositionEmbed(p, rowDim)} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <BlockMath math={String.raw`P_p = W_P[p],\quad W_P \in \mathbb{R}^{L_{max} \times d_{model}},\quad L_{max}=512`} />
    </>
  );
}

function SegmentEmbeddingVisual({
  tokens,
  d,
}: {
  tokens: BertToken[];
  d: number;
}) {
  const rowDim = Math.min(48, d);
  return (
    <>
      <Definition>
        BERT needs to know which sentence a token belongs to (for NSP and
        pair-classification tasks). It learns a tiny{" "}
        <strong>segment-embedding matrix with just 2 rows</strong> —
        segment A and segment B — and looks each token's segment ID up in it.
        This is BERT's distinctive addition vs. GPT.
      </Definition>
      <Section title="Segment vectors per token" hint="just A or B">
        <div className="rounded-xl overflow-hidden border border-ink-800">
          <table className="w-full text-sm">
            <tbody>
              {tokens.map((t, i) => (
                <tr
                  key={i}
                  className="border-t border-ink-800/80 first:border-t-0"
                >
                  <td className="py-2 pl-3 pr-2 w-32">
                    <TokenChip tok={t} index={i} />
                  </td>
                  <td className="py-2 px-2 w-14 font-mono text-xs text-ink-300">
                    seg {t.segment}
                  </td>
                  <td className="py-2 pr-3">
                    <HeatRow values={bertSegmentEmbed(t.segment, rowDim)} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <BlockMath math={String.raw`S_t = W_S[\text{seg}(t)],\quad W_S \in \mathbb{R}^{2 \times d_{model}}`} />
    </>
  );
}

function EmbeddingSumVisual({
  tokens,
  d,
}: {
  tokens: BertToken[];
  d: number;
}) {
  const rowDim = Math.min(48, d);
  return (
    <>
      <Definition>
        The three embeddings are <strong>added element-wise</strong> — this is
        the actual sequence of vectors{" "}
        <InlineMath math={String.raw`X \in \mathbb{R}^{n \times d_{model}}`} />{" "}
        that flows into the encoder.
      </Definition>
      <Section title="E_token + E_pos + E_seg = X">
        <div className="rounded-xl overflow-hidden border border-ink-800">
          {tokens.map((t, i) => {
            const e = bertTokenEmbed(t.id, rowDim);
            const p = bertPositionEmbed(i, rowDim);
            const s = bertSegmentEmbed(t.segment, rowDim);
            const sum = e.map((v, k) => v + p[k] + s[k]);
            return (
              <div
                key={i}
                className="grid grid-cols-[6rem_1fr] gap-2 items-center border-t border-ink-800/80 first:border-t-0 py-2 px-3"
              >
                <TokenChip tok={t} index={i} />
                <div className="flex flex-col gap-1">
                  <RowWithLabel label="E_t" values={e} />
                  <RowWithLabel label="+P" values={p} />
                  <RowWithLabel label="+S" values={s} />
                  <RowWithLabel label="= X" values={sum} accent />
                </div>
              </div>
            );
          })}
        </div>
      </Section>
      <BlockMath math={String.raw`X_t = E_t + P_t + S_t`} />
    </>
  );
}

function RowWithLabel({
  label,
  values,
  accent,
}: {
  label: string;
  values: number[];
  accent?: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={
          accent
            ? "font-mono text-[10px] text-cyan-300 w-8"
            : "font-mono text-[10px] text-ink-400 w-8"
        }
      >
        {label}
      </span>
      <div className="flex-1">
        <HeatRow values={values} height={accent ? 12 : 10} />
      </div>
    </div>
  );
}

function LayerNormVisual({ which }: { which: BertComponentId }) {
  const raw = useMemo(
    () =>
      Array.from({ length: 24 }, (_, i) =>
        3 * Math.sin(i * 0.7) + 2 * Math.cos(i * 1.3) + (i % 5)
      ),
    []
  );
  const mean = raw.reduce((a, b) => a + b, 0) / raw.length;
  const variance = raw.reduce((a, b) => a + (b - mean) ** 2, 0) / raw.length;
  const eps = 1e-5;
  const normed = raw.map((v) => (v - mean) / Math.sqrt(variance + eps));
  const isPostNorm = which === "layerNorm1" || which === "layerNorm2";
  return (
    <>
      <Definition>
        LayerNorm normalizes the features of each token independently across
        the <InlineMath math={String.raw`d_{model}`} /> dimensions. <br />
        {isPostNorm ? (
          <>
            <strong>BERT uses POST-norm</strong>: the LN is applied{" "}
            <em>after</em> the residual add — i.e.{" "}
            <span className="font-mono text-ink-200">
              y = LN(x + Sublayer(x))
            </span>
            . (Modern decoders like GPT-2 use the pre-norm variant.)
          </>
        ) : (
          <>
            Applied to the summed embeddings (before any encoder block) plus a
            dropout. Dropout in BERT-base is 0.1.
          </>
        )}
      </Definition>
      <Section title="Single token's features" hint="before vs after">
        <div className="flex flex-col gap-3">
          <div>
            <div className="text-[11px] text-ink-400 mb-1">x (raw)</div>
            <HeatRow values={raw} />
          </div>
          <div>
            <div className="text-[11px] text-ink-400 mb-1">
              LN(x) = γ · (x − μ) / √(σ² + ε) + β
            </div>
            <HeatRow values={normed} />
          </div>
          <div className="grid grid-cols-3 gap-2 text-sm">
            <Metric label="μ before" value={mean.toFixed(3)} />
            <Metric label="σ² before" value={variance.toFixed(3)} />
            <Metric label="ε" value="1e-12" />
          </div>
        </div>
      </Section>
      {isPostNorm ? (
        <BlockMath math={String.raw`y = \text{LN}\bigl(x + \text{Sublayer}(x)\bigr)`} />
      ) : (
        <BlockMath math={String.raw`\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta`} />
      )}
    </>
  );
}

function BidirectionalAttentionVisual({
  tokens,
}: {
  tokens: BertToken[];
}) {
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const n = Math.max(tokens.length, 2);
  const matrix = useMemo(() => bertAttention(n, 0), [n]);
  return (
    <>
      <Definition>
        BERT's attention is <strong>bidirectional</strong>: each token attends
        to <em>all</em> other tokens — past <em>and</em> future. There is{" "}
        <strong>no causal mask</strong>. That is exactly what makes BERT a good
        encoder for understanding tasks (and a poor generator — it can't sample
        left-to-right).
      </Definition>
      <Section
        title="Attention matrix"
        hint="row = query, col = key — full square"
      >
        <AttentionHeatmap
          tokens={tokens}
          matrix={matrix}
          hover={hover}
          onHover={setHover}
        />
        <p className="text-xs text-ink-400 mt-3 leading-relaxed">
          The matrix is fully populated (no upper-triangle mask). Each row sums
          to 1 after row-wise softmax. Notice how{" "}
          <span className="text-ink-200">[CLS]</span> and{" "}
          <span className="text-ink-200">[SEP]</span> often receive significant
          attention — that's how the model builds its sequence-level summary.
        </p>
      </Section>
      <Section title="Formula">
        <BlockMath
          math={String.raw`\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V`}
        />
        <div className="text-xs text-ink-300 leading-relaxed">
          No <InlineMath math="M" /> term — every position can see every other
          position.
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
  tokens: BertToken[];
  matrix: number[][];
  onHover?: (h: { i: number; j: number } | null) => void;
  hover?: { i: number; j: number } | null;
}) {
  const n = matrix.length;
  const cell = 30;
  const labelW = 64;
  const labelH = 68;
  const width = labelW + n * cell;
  const height = labelH + n * cell;
  return (
    <div className="overflow-auto">
      <svg width={width} height={height} className="block" style={{ maxWidth: "100%" }}>
        {tokens.map((t, j) => (
          <text
            key={`c-${j}`}
            x={labelW + j * cell + cell / 2}
            y={labelH - 8}
            textAnchor="end"
            transform={`rotate(-40, ${labelW + j * cell + cell / 2}, ${labelH - 8})`}
            fontSize={10}
            fill={t.isSpecial ? "#fb923c" : "#c2c7d4"}
            fontFamily="JetBrains Mono, monospace"
          >
            {t.token}
          </text>
        ))}
        {tokens.map((t, i) => (
          <text
            key={`r-${i}`}
            x={labelW - 8}
            y={labelH + i * cell + cell / 2 + 4}
            textAnchor="end"
            fontSize={10}
            fill={t.isSpecial ? "#fb923c" : "#c2c7d4"}
            fontFamily="JetBrains Mono, monospace"
          >
            {t.token}
          </text>
        ))}
        {matrix.map((row, i) =>
          row.map((v, j) => {
            const intensity = v;
            const hue = 195 - intensity * 70; // cyan → magenta
            const isHover = hover?.i === i && hover?.j === j;
            return (
              <rect
                key={`${i}-${j}`}
                x={labelW + j * cell + 1.5}
                y={labelH + i * cell + 1.5}
                width={cell - 3}
                height={cell - 3}
                rx={4}
                fill={`hsl(${hue}, 70%, ${18 + intensity * 55}%)`}
                stroke={isHover ? "#fff" : "rgba(255,255,255,0.06)"}
                strokeWidth={isHover ? 1.5 : 1}
                onMouseEnter={() => onHover?.({ i, j })}
                onMouseLeave={() => onHover?.(null)}
              />
            );
          })
        )}
      </svg>
      {hover && (
        <div className="mt-2 text-xs text-ink-300 font-mono">
          A[{hover.i}, {hover.j}] = {matrix[hover.i][hover.j].toFixed(4)} —{" "}
          <span className="text-cyan-300">"{tokens[hover.i]?.token}"</span>{" "}
          attends to{" "}
          <span className="text-pink-300">"{tokens[hover.j]?.token}"</span>
        </div>
      )}
    </div>
  );
}

function MultiHeadVisual({
  tokens,
  numHeads,
}: {
  tokens: BertToken[];
  numHeads: number;
}) {
  const n = Math.max(tokens.length, 2);
  const heads = Math.min(numHeads, 6);
  const matrices = useMemo(
    () => Array.from({ length: heads }, (_, h) => bertAttention(n, h + 1)),
    [n, heads]
  );
  return (
    <>
      <Definition>
        BERT runs <InlineMath math="h" /> attention heads in parallel; each
        head learns a different relational pattern (subject ↔ verb, adjective ↔
        noun, [CLS] ↔ whole sentence, etc.). Their outputs are concatenated and
        projected back to <InlineMath math={String.raw`d_{model}`} />.
      </Definition>
      <Section title={`${heads} of ${numHeads} heads shown`}>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {matrices.map((m, h) => (
            <MiniHead key={h} matrix={m} index={h} />
          ))}
        </div>
      </Section>
      <BlockMath math={String.raw`\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O`} />
    </>
  );
}

function MiniHead({ matrix, index }: { matrix: number[][]; index: number }) {
  const n = matrix.length;
  const cell = 18;
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
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ maxWidth: "100%" }}>
        {matrix.map((row, i) =>
          row.map((v, j) => {
            const hue = 195 - v * 70;
            return (
              <rect
                key={`${i}-${j}`}
                x={j * cell + 1}
                y={i * cell + 1}
                width={cell - 2}
                height={cell - 2}
                rx={2}
                fill={`hsl(${hue}, 70%, ${18 + v * 55}%)`}
              />
            );
          })
        )}
      </svg>
    </div>
  );
}

function FFNVisual({ dModel }: { dModel: number }) {
  const inner = dModel * 4;
  return (
    <>
      <Definition>
        Identical to GPT's MLP: a per-token <strong>2-layer MLP</strong> with{" "}
        <strong>4× expansion</strong> and <strong>GELU</strong>. Mixes
        features within each token.
      </Definition>
      <Section title="Shape transformation">
        <div className="flex items-center gap-3 justify-center text-center py-2 flex-wrap">
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
      <div className="font-mono text-base text-white">
        {value.toLocaleString()}
      </div>
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

function ResidualVisual({ which }: { which: BertComponentId }) {
  return (
    <>
      <Definition>
        Both sublayers (attention and FFN) wrap their output in a residual
        connection. In BERT, the LayerNorm comes <strong>after</strong> the
        add: <span className="font-mono text-ink-100">LN(x + Sublayer(x))</span>.
      </Definition>
      <Section title="Schematic">
        <svg viewBox="0 0 360 160" className="w-full">
          <defs>
            <marker id="b-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
              <path d="M0,0 L0,6 L9,3 z" fill="#a48bff" />
            </marker>
          </defs>
          <rect x="20" y="60" width="80" height="40" rx="10" fill="#1f2330" stroke="#22d3ee" />
          <text x="60" y="85" fill="#fff" fontSize="12" textAnchor="middle">x</text>
          <rect x="140" y="20" width="120" height="40" rx="10" fill="#161922" stroke="#22d3ee" />
          <text x="200" y="45" fill="#fff" fontSize="12" textAnchor="middle">sublayer(x)</text>
          <path d="M100 80 H 200 V 60" stroke="#a48bff" strokeWidth="1.5" fill="none" markerEnd="url(#b-arrow)" />
          <path d="M100 80 H 320" stroke="#a48bff" strokeDasharray="4 4" strokeWidth="1.5" fill="none" />
          <path d="M260 40 H 320 V 80" stroke="#22d3ee" strokeWidth="1.5" fill="none" />
          <circle cx="320" cy="80" r="14" fill="#0f1117" stroke="#fff" />
          <text x="320" y="84" fill="#fff" fontSize="14" textAnchor="middle">+</text>
          <path d="M320 94 V 140" stroke="#a48bff" strokeWidth="1.5" fill="none" markerEnd="url(#b-arrow)" />
          <text x="340" y="120" fill="#a48bff" fontSize="11">y = LN(x + sublayer(x))</text>
        </svg>
      </Section>
      <div className="text-[11px] text-ink-400">
        ID: <span className="font-mono">{which}</span>
      </div>
    </>
  );
}

function PoolerVisual() {
  return (
    <>
      <Definition>
        For sentence-level tasks BERT uses only the final hidden state of the{" "}
        <code className="font-mono text-orange-300">[CLS]</code> token. The
        pooler runs a small <strong>Linear → tanh</strong> transformation on
        that vector to produce a summary representation. NSP, sentiment
        classification, and similar tasks build their classifier head on top of
        this.
      </Definition>
      <Section title="Schematic">
        <div className="flex items-center gap-3 justify-center text-center py-2 flex-wrap">
          <ShapeBlock label="h_[CLS]" value={768} color="#fb923c" />
          <Arrowy label="Linear" />
          <ShapeBlock label="d_model" value={768} color="#a78bfa" />
          <Arrowy label="tanh" />
          <ShapeBlock label="pooled" value={768} color="#22d3ee" />
        </div>
      </Section>
      <BlockMath math={String.raw`p_{[CLS]} = \tanh(W_p \, h_{[CLS]} + b_p)`} />
    </>
  );
}

function NSPVisual({ textA, textB }: { textA: string; textB: string }) {
  const probs = useMemo(() => nspPredict(textA, textB), [textA, textB]);
  return (
    <>
      <Definition>
        <strong>Next-Sentence Prediction</strong> is one of BERT's two
        pre-training objectives. Given the pooled [CLS] representation, a small
        linear layer outputs two logits — "B follows A" vs "not next" — that
        are softmaxed into a binary distribution.
      </Definition>
      <Section title="Prediction">
        <div className="flex flex-col gap-3">
          <ProbBar label="IsNext" value={probs.isNext} color="#34d399" />
          <ProbBar label="NotNext" value={probs.notNext} color="#f87171" />
        </div>
        <p className="text-[11px] text-ink-400 mt-3">
          {textB.trim()
            ? `Pair: "${textA}" → "${textB}"`
            : "Add sentence B above to see the prediction."}
        </p>
      </Section>
      <BlockMath math={String.raw`P(\text{IsNext} \mid A, B) = \text{softmax}(W_{nsp}\, p_{[CLS]})_0`} />
    </>
  );
}

function ProbBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="grid grid-cols-[6rem_1fr_3.5rem] items-center gap-3">
      <span className="text-sm">{label}</span>
      <div className="h-2.5 rounded-full bg-ink-800 overflow-hidden">
        <motion.div
          className="h-full"
          style={{ background: color }}
          initial={false}
          animate={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
          transition={{ type: "spring", stiffness: 220, damping: 24 }}
        />
      </div>
      <span className="font-mono text-xs text-ink-200 text-right">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function MLMVisual({
  tokens,
  maskedIndex,
}: {
  tokens: BertToken[];
  maskedIndex: number | null;
}) {
  const masked = maskedIndex !== null ? tokens[maskedIndex] : null;
  const original = masked?.maskedFor;
  const preds = useMemo(
    () => mlmPredict(tokens.map((t) => t.token).join(" "), original),
    [tokens, original]
  );
  return (
    <>
      <Definition>
        BERT's main pre-training objective. ~15% of input tokens are randomly
        replaced with <code className="font-mono text-orange-300">[MASK]</code>
        ; the model predicts the original token from its <em>bidirectional</em>{" "}
        context. The MLM head is: Linear → GELU → LN → projection to vocab
        (often tied to the input embedding).
      </Definition>

      <Section title="Masked input">
        <div className="flex flex-wrap gap-2">
          {tokens.map((t, i) => (
            <div
              key={i}
              className={
                i === maskedIndex
                  ? "ring-2 ring-orange-400 rounded-md"
                  : undefined
              }
            >
              <TokenChip tok={t} index={i} />
            </div>
          ))}
        </div>
        {maskedIndex === null && (
          <p className="text-[11px] text-ink-400 mt-2">
            Toggle a token to mask above the diagram to see the prediction.
          </p>
        )}
      </Section>

      {maskedIndex !== null && (
        <Section
          title={`Predictions for [MASK] (originally "${original}")`}
          hint="top-8 from vocab"
        >
          <div className="flex flex-col gap-1.5">
            {preds.slice(0, 8).map((p, i) => (
              <div
                key={i}
                className="grid grid-cols-[5.5rem_1fr_3.5rem] items-center gap-3"
              >
                <span className="font-mono text-sm flex items-center gap-1.5">
                  {p.isOriginal && (
                    <span
                      className="w-1.5 h-1.5 rounded-full bg-emerald-400"
                      title="original token"
                    />
                  )}
                  {i + 1}. {p.token}
                </span>
                <div className="h-2.5 rounded-full bg-ink-800 overflow-hidden">
                  <motion.div
                    className="h-full"
                    style={{
                      background: p.isOriginal
                        ? "linear-gradient(90deg, #34d399, #22d3ee)"
                        : `linear-gradient(90deg, ${TOKEN_COLORS[i % TOKEN_COLORS.length]}, #22d3ee)`,
                    }}
                    initial={false}
                    animate={{ width: `${(p.prob / preds[0].prob) * 100}%` }}
                    transition={{ type: "spring", stiffness: 220, damping: 24 }}
                  />
                </div>
                <span className="font-mono text-xs text-ink-300 text-right">
                  {(p.prob * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}

      <BlockMath
        math={String.raw`P(t_i \mid \text{context}) = \text{softmax}\bigl(W_E^\top \cdot \text{LN}(\text{GELU}(W_{mlm} h_i))\bigr)`}
      />
    </>
  );
}
