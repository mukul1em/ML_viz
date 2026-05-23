import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Compass,
  Layers,
  Sparkles,
  Workflow,
  Zap,
} from "lucide-react";
import { Tex as InlineMath, TexBlock as BlockMath } from "../Tex";
import {
  qwenComponents,
  type QwenComponent,
  type QwenComponentId,
} from "../../data/qwenComponents";
import {
  applyRoPE,
  gqaLayout,
  layerNorm,
  qwenTokenize,
  rmsNorm,
  seeded,
  silu,
  swiglu,
  tokenEmbed,
  type QwenConfig,
} from "../../lib/qwenDemo";

interface Props {
  selected: QwenComponentId;
  prompt: string;
  config: QwenConfig;
  embedded?: boolean;
  /**
   * Optional override for the component-metadata map (used by LLaMA, which
   * shares the same architectural graph but ships its own taglines). Defaults
   * to the Qwen metadata.
   */
  components?: Record<QwenComponentId, QwenComponent>;
  /** Optional accent color for the "SIGNATURE" badge. */
  accentColor?: string;
}

export default function QwenDetail({
  selected,
  prompt,
  config,
  embedded,
  components,
  accentColor,
}: Props) {
  const meta = (components ?? qwenComponents)[selected];
  const badge = accentColor
    ? {
        text: { color: accentColor },
        wrap: {
          background: `${accentColor}1f`,
          border: `1px solid ${accentColor}66`,
          color: accentColor,
        },
      }
    : null;
  const wrap = embedded
    ? "flex flex-col gap-5"
    : "card p-5 flex flex-col gap-5";

  return (
    <div className={wrap}>
      <header className="flex flex-col gap-2">
        <div
          className="flex items-center gap-2 text-[10px] uppercase tracking-[0.18em] font-bold"
          style={badge ? badge.text : undefined}
        >
          {meta.signature && (
            <span
              className={
                badge
                  ? "px-1.5 py-0.5 rounded"
                  : "px-1.5 py-0.5 rounded bg-violet-500/20 border border-violet-400/40 text-violet-200"
              }
              style={badge ? badge.wrap : undefined}
            >
              SIGNATURE
            </span>
          )}
          <span className={badge ? "" : "text-violet-300"}>{meta.group}</span>
        </div>
        <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight leading-tight">
          {meta.full}
        </h3>
        <p className="text-sm text-ink-300 leading-relaxed">{meta.tagline}</p>
      </header>

      <Body selected={selected} prompt={prompt} config={config} />
    </div>
  );
}

function Body({
  selected,
  prompt,
  config,
}: {
  selected: QwenComponentId;
  prompt: string;
  config: QwenConfig;
}) {
  switch (selected) {
    case "input":
      return <InputDetail prompt={prompt} />;
    case "tokenization":
      return <TokenizationDetail prompt={prompt} />;
    case "tokenEmbedding":
      return <TokenEmbedDetail prompt={prompt} config={config} />;
    case "rmsNorm1":
    case "rmsNorm2":
    case "finalRmsNorm":
      return <RMSNormDetail prompt={prompt} />;
    case "selfAttention":
    case "qkvProj":
    case "scaledDotProduct":
    case "attnOut":
      return <AttentionDetail prompt={prompt} config={config} />;
    case "rope":
      return <RopeDetail config={config} />;
    case "gqa":
      return <GqaDetail config={config} />;
    case "residual1":
    case "residual2":
      return <ResidualDetail />;
    case "swiglu":
    case "gateUp":
    case "silu":
    case "downProj":
      return <SwigluDetail config={config} />;
    case "lmHead":
      return <LmHeadDetail config={config} />;
    case "sampling":
      return <SamplingDetail />;
    default:
      return null;
  }
}

/* =============================== INPUT / TOKENIZER =============================== */

function InputDetail({ prompt }: { prompt: string }) {
  return (
    <Section icon={<Workflow className="w-3.5 h-3.5" />} title="The starting point">
      <p className="text-sm text-ink-300 leading-relaxed">
        Just text. Qwen treats it as a byte stream so it can ingest any
        Unicode — Latin script, CJK, emoji, even raw bytes — without
        special pre-processing.
      </p>
      <pre className="rounded-lg bg-ink-950/70 border border-ink-800 p-3 text-sm font-mono whitespace-pre-wrap">
        {prompt || "(type something in the prompt bar)"}
      </pre>
    </Section>
  );
}

function TokenizationDetail({ prompt }: { prompt: string }) {
  const tokens = qwenTokenize(prompt);
  return (
    <>
      <Section icon={<Workflow className="w-3.5 h-3.5" />} title="Byte-level BPE">
        <p className="text-sm text-ink-300 leading-relaxed">
          Qwen uses a byte-level BPE tokenizer with{" "}
          <strong>~152,064 tokens</strong>. About 3× GPT-2's vocabulary,
          giving better coverage of CJK characters and emoji at the cost of
          a slightly larger output projection.
        </p>
      </Section>
      <Section title="Tokens">
        <div className="flex flex-wrap gap-1.5">
          {tokens.map((t, i) => (
            <div key={i} className="flex flex-col items-start">
              <span className="px-2 py-1 rounded-md bg-violet-500/15 border border-violet-400/40 text-violet-200 font-mono text-sm">
                {t.token}
              </span>
              <span className="text-[10px] text-ink-400 font-mono ml-1 mt-0.5">
                {t.id}
              </span>
            </div>
          ))}
          {tokens.length === 0 && (
            <span className="text-xs text-ink-400">No tokens yet — enter a prompt.</span>
          )}
        </div>
      </Section>
    </>
  );
}

function TokenEmbedDetail({
  prompt,
  config,
}: {
  prompt: string;
  config: QwenConfig;
}) {
  const tokens = qwenTokenize(prompt).slice(0, 5);
  const rows = tokens.map((t) => ({ tok: t.token, vec: tokenEmbed(t.id, 12) }));
  return (
    <>
      <Section icon={<Workflow className="w-3.5 h-3.5" />} title="Why no positional embedding?">
        <p className="text-sm text-ink-300 leading-relaxed">
          GPT and BERT add a learned <em>positional embedding</em> to every
          token vector. Qwen <strong>doesn't</strong>. Position information
          is injected later, inside attention, by{" "}
          <span className="text-pink-300">RoPE rotations</span> applied to{" "}
          <InlineMath math="Q" /> and <InlineMath math="K" />. This lets the
          same model handle much longer context lengths than it was trained
          on.
        </p>
        <BlockMath math={String.raw`x_i \;=\; W_E[\text{tok\_id}_i] \quad\in\;\mathbb{R}^{d_{model}}`} />
        <p className="text-[11px] text-ink-400">
          Shape: <span className="font-mono">[batch, seq_len, {config.d_model}]</span>
        </p>
      </Section>
      <Section title="First few embeddings (12 dims shown)">
        <div className="flex flex-col gap-1.5">
          {rows.map((r, i) => (
            <div key={i} className="grid grid-cols-[5rem_1fr] gap-2 items-center">
              <span className="text-xs font-mono text-violet-200 truncate">{r.tok}</span>
              <HeatRow values={r.vec} />
            </div>
          ))}
        </div>
      </Section>
    </>
  );
}

/* =============================== RMSNORM =============================== */

function RMSNormDetail({ prompt }: { prompt: string }) {
  const tok = qwenTokenize(prompt)[0];
  const x = tok ? tokenEmbed(tok.id, 16) : Array.from({ length: 16 }, (_, i) => seeded(7, i));
  const ln = layerNorm(x);
  const rms = rmsNorm(x);

  return (
    <>
      <Section icon={<Zap className="w-3.5 h-3.5" />} title="Just rescale, don't center">
        <p className="text-sm text-ink-300 leading-relaxed">
          LayerNorm subtracts the mean and divides by the std. RMSNorm{" "}
          <em>only</em> divides — by the root-mean-square. Slightly less
          expressive, but cheaper (no mean computation, no bias) and
          empirically just as good. Qwen uses it everywhere.
        </p>
        <BlockMath
          math={String.raw`\text{RMSNorm}(x) \;=\; \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}} \;\odot\; \gamma`}
        />
        <BlockMath
          math={String.raw`\text{LayerNorm}(x) \;=\; \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \;\odot\; \gamma \;+\; \beta`}
        />
      </Section>

      <Section title="Same input, three transformations">
        <div className="flex flex-col gap-2">
          <RowLabeled label="x (raw)" values={x} accent="#a78bfa" />
          <RowLabeled label="LayerNorm" values={ln} accent="#22d3ee" />
          <RowLabeled label="RMSNorm" values={rms} accent="#34d399" />
        </div>
        <div className="grid grid-cols-3 gap-2 text-[11px] mt-3">
          <Metric label="Params" value="2d → d" hint="no β bias" color="#34d399" />
          <Metric label="FLOPs" value="≈ 3× less" hint="vs LN" color="#22d3ee" />
          <Metric label="Centered?" value="no" hint="just rescaled" color="#a78bfa" />
        </div>
      </Section>
    </>
  );
}

/* =============================== ATTENTION =============================== */

function AttentionDetail({
  prompt,
  config,
}: {
  prompt: string;
  config: QwenConfig;
}) {
  const tokens = qwenTokenize(prompt).slice(0, 8);
  const n = Math.max(tokens.length, 3);

  // Causal attention matrix (mocked)
  const attn = useMemo(() => {
    const out: number[][] = [];
    for (let i = 0; i < n; i++) {
      const logits = new Array(n).fill(-Infinity);
      for (let j = 0; j <= i; j++) {
        const diag = i === j ? 1.0 : 0;
        const prev = i - j === 1 ? 0.8 : 0;
        const noise = 0.4 * seeded(i * 31 + j, 7);
        logits[j] = 0.2 + diag + prev + noise;
      }
      const m = Math.max(...logits.filter((v) => Number.isFinite(v)));
      const e = logits.map((v) => (Number.isFinite(v) ? Math.exp(v - m) : 0));
      const s = e.reduce((a, b) => a + b, 0) || 1;
      out.push(e.map((x) => x / s));
    }
    return out;
  }, [n]);

  return (
    <>
      <Section icon={<Layers className="w-3.5 h-3.5" />} title="Causal self-attention, Qwen-flavored">
        <p className="text-sm text-ink-300 leading-relaxed">
          Inside Qwen's attention, three things distinguish it from vanilla
          GPT-style attention:
        </p>
        <ol className="text-sm text-ink-200 list-decimal pl-5 space-y-1">
          <li>
            <strong className="text-pink-300">RoPE</strong> rotates Q and K
            before the dot product — that's where positions live.
          </li>
          <li>
            <strong className="text-amber-300">GQA</strong> — many query heads
            but only a few KV heads, sharing.
          </li>
          <li>
            <strong className="text-violet-300">No bias</strong> in the Q/K/V/out
            linears.
          </li>
        </ol>
        <BlockMath
          math={String.raw`\text{Attn}(Q, K, V) \;=\; \text{softmax}\!\left(\frac{\text{RoPE}(Q)\;\text{RoPE}(K)^\top}{\sqrt{d_k}} + M\right) V`}
        />
        <p className="text-[11px] text-ink-400">
          <InlineMath math="M" /> is the causal mask (−∞ above the diagonal).
        </p>
      </Section>

      <Section title={`Attention pattern · ${tokens.length} tokens · 1 head`}>
        <Heatmap matrix={attn} labels={tokens.map((t) => t.token)} />
      </Section>

      <div className="grid grid-cols-3 gap-2 text-[11px]">
        <Metric label="q_heads" value={String(config.q_heads)} color="#34d399" />
        <Metric label="kv_heads" value={String(config.kv_heads)} color="#facc15" />
        <Metric
          label="head_dim"
          value={String(Math.floor(config.d_model / config.q_heads))}
          color="#22d3ee"
        />
      </div>
    </>
  );
}

/* =============================== ROPE =============================== */

function RopeDetail({ config }: { config: QwenConfig }) {
  const headDim = 8; // small dim for visualization
  const [pos, setPos] = useState(3);

  // Baseline Q vector
  const q = useMemo(
    () => Array.from({ length: headDim }, (_, i) => seeded(11, i + 1)),
    []
  );
  const rotated = useMemo(() => applyRoPE(q, pos), [q, pos]);

  // For the 2D rotation diagram, focus on one pair
  const pairIdx = 0;
  const theta = Math.pow(10_000, -(2 * pairIdx) / headDim);
  const angle = pos * theta;

  return (
    <>
      <Section icon={<Workflow className="w-3.5 h-3.5" />} title="Position via rotation">
        <p className="text-sm text-ink-300 leading-relaxed">
          RoPE encodes position as a <strong>rotation</strong> on pairs of
          dimensions. Pair <InlineMath math="(2i, 2i+1)" /> of Q (and K) gets
          rotated by an angle that depends on the token's position{" "}
          <InlineMath math="m" /> and a per-pair frequency{" "}
          <InlineMath math="\theta_i" />.
        </p>
        <BlockMath
          math={String.raw`\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \;\;\cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}`}
        />
        <BlockMath math={String.raw`\theta_i \;=\; 10000^{-2i/d}`} />
      </Section>

      <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-3">
        <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold mb-2">
          Position m
        </div>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={0}
            max={32}
            step={1}
            value={pos}
            onChange={(e) => setPos(Number(e.target.value))}
            className="flex-1 accent-pink-500"
          />
          <span className="font-mono text-pink-200 text-sm w-12 text-right">m={pos}</span>
        </div>
      </div>

      <Section title={`Watch pair (0, 1) rotate as position m grows`}>
        <RopeRotationViz angle={angle} />
        <div className="text-[11px] text-ink-400 leading-relaxed">
          Pair 0 has the fastest frequency (θ₀ = 1). Later pairs rotate slower
          (θᵢ → 0 as i grows), so they encode coarser-grained position info.
        </div>
      </Section>

      <Section title="Full Q vector before vs after RoPE">
        <RowLabeled label="Q (pre)" values={q} accent="#a78bfa" />
        <RowLabeled label="Q (post-RoPE)" values={rotated} accent="#ec4899" />
        <div className="text-[11px] text-ink-400 mt-2 leading-relaxed">
          Same magnitudes per pair (rotation preserves length), different
          phases. Because both Q and K get rotated, their dot product
          becomes a function of <em>relative</em> position{" "}
          <InlineMath math="m - n" /> — that's the magic.
        </div>
      </Section>

      <div className="text-[11px] text-ink-400">
        Real head_dim = <span className="font-mono">{Math.floor(config.d_model / config.q_heads)}</span> (visualizing 8 here).
      </div>
    </>
  );
}

function RopeRotationViz({ angle }: { angle: number }) {
  const size = 220;
  const cx = size / 2;
  const cy = size / 2;
  const r = 70;
  const ax = cx + r;
  const ay = cy;
  const bx = cx + r * Math.cos(angle);
  const by = cy - r * Math.sin(angle);
  return (
    <svg width={size} height={size} className="mx-auto block">
      {/* Axes */}
      <line x1={cx - r - 10} y1={cy} x2={cx + r + 10} y2={cy} stroke="rgba(255,255,255,0.12)" strokeWidth={1} />
      <line x1={cx} y1={cy - r - 10} x2={cx} y2={cy + r + 10} stroke="rgba(255,255,255,0.12)" strokeWidth={1} />
      <text x={cx + r + 16} y={cy + 4} fontSize={10} fill="#8b94a8">q₀</text>
      <text x={cx + 6} y={cy - r - 12} fontSize={10} fill="#8b94a8">q₁</text>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
      {/* Original vector */}
      <line x1={cx} y1={cy} x2={ax} y2={ay} stroke="#a78bfa" strokeWidth={2.5} />
      <circle cx={ax} cy={ay} r={4} fill="#a78bfa" />
      <text x={ax + 6} y={ay - 5} fontSize={10} fill="#a78bfa">pre</text>
      {/* Rotated */}
      <motion.line
        x1={cx}
        y1={cy}
        x2={bx}
        y2={by}
        stroke="#ec4899"
        strokeWidth={2.5}
        initial={false}
        animate={{ x2: bx, y2: by }}
        transition={{ type: "spring", stiffness: 220, damping: 22 }}
      />
      <motion.circle
        cx={bx}
        cy={by}
        r={4.5}
        fill="#ec4899"
        initial={false}
        animate={{ cx: bx, cy: by }}
        transition={{ type: "spring", stiffness: 220, damping: 22 }}
      />
      <text x={bx + 8} y={by - 6} fontSize={10} fill="#ec4899" fontWeight={700}>
        post
      </text>
      {/* Arc */}
      <path
        d={`M ${cx + 22} ${cy} A 22 22 0 ${Math.abs(angle) > Math.PI ? 1 : 0} 0 ${cx + 22 * Math.cos(angle)} ${cy - 22 * Math.sin(angle)}`}
        fill="none"
        stroke="#fbbf24"
        strokeWidth={1.4}
      />
      <text x={cx + 38} y={cy - 18} fontSize={10} fill="#fbbf24" fontFamily="JetBrains Mono, monospace">
        m·θ₀ ≈ {angle.toFixed(2)}
      </text>
    </svg>
  );
}

/* =============================== GQA =============================== */

function GqaDetail({ config }: { config: QwenConfig }) {
  const layout = gqaLayout(config.q_heads, config.kv_heads);
  // Cap for visualization
  const visQ = Math.min(config.q_heads, 16);
  const visKv = Math.min(config.kv_heads, 8);
  const visLayout = gqaLayout(visQ, visKv);

  return (
    <>
      <Section icon={<Layers className="w-3.5 h-3.5" />} title="One KV head, many Q heads">
        <p className="text-sm text-ink-300 leading-relaxed">
          Classic multi-head attention gives every Q head its own K and V
          head. That's wasteful: at inference, the KV cache grows linearly
          with the number of heads × layers × tokens. <strong>Grouped-Query
          Attention</strong> shares one K/V head across a <em>group</em> of
          Q heads — same model quality, much smaller cache.
        </p>
        <BlockMath
          math={String.raw`\text{head}_h \;=\; \text{Attn}(Q_h,\; K_{g(h)},\; V_{g(h)})`}
        />
        <p className="text-[11px] text-ink-400">
          <InlineMath math="g(h) = \lfloor h \cdot \frac{n_{kv}}{n_q} \rfloor" /> maps each Q head to a KV head.
        </p>
      </Section>

      <Section title={`This model: ${config.q_heads} Q heads → ${config.kv_heads} KV heads`}>
        <GqaDiagram qHeads={visQ} kvHeads={visKv} mapping={visLayout.mapping} />
        {(config.q_heads !== visQ || config.kv_heads !== visKv) && (
          <div className="text-[10px] text-ink-400 mt-1.5">
            Showing {visQ}/{visKv} for clarity. Real groups: {config.q_heads / config.kv_heads}×.
          </div>
        )}
      </Section>

      <div className="grid grid-cols-3 gap-2">
        <Metric label="Group size" value={`${Math.floor(layout.qHeads / layout.kvHeads)}×`} color="#34d399" hint="Q per KV head" />
        <Metric
          label="KV cache vs MHA"
          value={`${Math.round((layout.kvHeads / layout.qHeads) * 100)}%`}
          color="#facc15"
          hint="of the original size"
        />
        <Metric
          label="At extreme"
          value={layout.kvHeads === 1 ? "MQA" : "GQA"}
          color="#ec4899"
          hint="MQA = 1 KV head total"
        />
      </div>

      <div className="rounded-xl border border-amber-400/30 bg-amber-400/[0.05] p-3 text-[12px] text-amber-100 leading-relaxed">
        <strong>Why it works:</strong> in well-trained transformers, K and V
        across heads carry highly correlated information. Sharing K/V
        forces the network to make these "lookup keys" more general, and
        actually improves robustness while shrinking inference cost.
      </div>
    </>
  );
}

function GqaDiagram({
  qHeads,
  kvHeads,
  mapping,
}: {
  qHeads: number;
  kvHeads: number;
  mapping: number[];
}) {
  const W = 360;
  const H = 220;
  const qSpacing = (W - 40) / qHeads;
  const kvSpacing = (W - 40) / kvHeads;
  const qY = 36;
  const kvY = 170;
  const groupColors = ["#34d399", "#22d3ee", "#a78bfa", "#fbbf24", "#ec4899", "#f87171", "#60a5fa", "#facc15"];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
      {/* Lines from Q to KV */}
      {mapping.map((kv, qi) => {
        const x1 = 20 + qSpacing * (qi + 0.5);
        const x2 = 20 + kvSpacing * (kv + 0.5);
        return (
          <line
            key={qi}
            x1={x1}
            y1={qY + 14}
            x2={x2}
            y2={kvY - 14}
            stroke={`${groupColors[kv % groupColors.length]}88`}
            strokeWidth={1.2}
          />
        );
      })}

      {/* Q heads */}
      {Array.from({ length: qHeads }).map((_, i) => (
        <g key={`q-${i}`}>
          <rect
            x={20 + qSpacing * i + 4}
            y={qY - 14}
            width={qSpacing - 8}
            height={28}
            rx={6}
            fill={`${groupColors[mapping[i] % groupColors.length]}22`}
            stroke={groupColors[mapping[i] % groupColors.length]}
            strokeWidth={1}
          />
          <text
            x={20 + qSpacing * (i + 0.5)}
            y={qY + 5}
            textAnchor="middle"
            fontSize={9}
            fill="#fff"
            fontFamily="JetBrains Mono, monospace"
          >
            Q{i}
          </text>
        </g>
      ))}

      {/* KV heads */}
      {Array.from({ length: kvHeads }).map((_, i) => (
        <g key={`kv-${i}`}>
          <rect
            x={20 + kvSpacing * i + 4}
            y={kvY - 14}
            width={kvSpacing - 8}
            height={36}
            rx={6}
            fill={`${groupColors[i % groupColors.length]}33`}
            stroke={groupColors[i % groupColors.length]}
            strokeWidth={1.5}
          />
          <text
            x={20 + kvSpacing * (i + 0.5)}
            y={kvY + 0}
            textAnchor="middle"
            fontSize={10}
            fontWeight={700}
            fill="#fff"
            fontFamily="JetBrains Mono, monospace"
          >
            KV{i}
          </text>
          <text
            x={20 + kvSpacing * (i + 0.5)}
            y={kvY + 13}
            textAnchor="middle"
            fontSize={8}
            fill="rgba(255,255,255,0.6)"
          >
            shared
          </text>
        </g>
      ))}

      {/* Labels */}
      <text x={20} y={20} fontSize={11} fontWeight={700} fill="#34d399">{qHeads} Query heads</text>
      <text x={20} y={H - 10} fontSize={11} fontWeight={700} fill="#fbbf24">{kvHeads} Key/Value heads</text>
    </svg>
  );
}

/* =============================== SWIGLU =============================== */

function SwigluDetail({ config }: { config: QwenConfig }) {
  const inDim = 8;
  const hidDim = 12;
  const x = useMemo(
    () => Array.from({ length: inDim }, (_, i) => seeded(13, i + 1)),
    []
  );
  const { gate, up, gated, product } = useMemo(
    () => swiglu(x, hidDim),
    [x]
  );

  return (
    <>
      <Section icon={<Zap className="w-3.5 h-3.5" />} title="Gated FFN, three projections">
        <p className="text-sm text-ink-300 leading-relaxed">
          A vanilla transformer FFN is{" "}
          <InlineMath math="W_2(\text{GELU}(W_1 x))" /> — two linears with an
          activation in between. SwiGLU adds a third projection that
          *gates* the activation. Empirically: same parameter budget, better
          performance.
        </p>
        <BlockMath
          math={String.raw`\text{SwiGLU}(x) \;=\; W_{down}\,\bigl(\,\text{SiLU}(W_{gate}\,x) \;\odot\; W_{up}\,x\,\bigr)`}
        />
        <BlockMath math={String.raw`\text{SiLU}(z) \;=\; z \cdot \sigma(z) \;=\; \frac{z}{1+e^{-z}}`} />
      </Section>

      <Section title="SiLU activation">
        <SiluCurve />
        <div className="text-[11px] text-ink-400 mt-2 leading-relaxed">
          A smooth, non-monotonic activation — nearly linear for big positive
          values, gracefully kills off negatives. (Also called "Swish".)
        </div>
      </Section>

      <Section title="Step-by-step on one position">
        <div className="flex flex-col gap-1.5">
          <RowLabeled label="x" values={x} accent="#a78bfa" />
          <Arrow />
          <RowLabeled label="gate(x)" values={gate} accent="#22d3ee" />
          <RowLabeled label="up(x)" values={up} accent="#34d399" />
          <Arrow />
          <RowLabeled label="SiLU(gate)" values={gated} accent="#60a5fa" />
          <Arrow centerLabel="⊙ (element-wise ×)" />
          <RowLabeled label="result" values={product} accent="#7c5cff" />
        </div>
      </Section>

      <div className="grid grid-cols-3 gap-2">
        <Metric label="d_model" value={String(config.d_model)} color="#a78bfa" />
        <Metric label="ffn_dim" value={String(config.ffn)} color="#7c5cff" />
        <Metric label="Linears" value="3" color="#34d399" hint="gate + up + down" />
      </div>
    </>
  );
}

function SiluCurve() {
  const w = 320;
  const h = 140;
  const xMin = -6;
  const xMax = 6;
  const yMin = -1.5;
  const yMax = 5;
  const sx = (x: number) => ((x - xMin) / (xMax - xMin)) * w;
  const sy = (y: number) => h - ((y - yMin) / (yMax - yMin)) * h;
  const N = 100;
  const pts: string[] = [];
  for (let i = 0; i <= N; i++) {
    const x = xMin + ((xMax - xMin) * i) / N;
    const y = silu(x);
    pts.push(`${sx(x)},${sy(y)}`);
  }

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      {/* Axes */}
      <line x1={0} y1={sy(0)} x2={w} y2={sy(0)} stroke="rgba(255,255,255,0.12)" strokeWidth={1} />
      <line x1={sx(0)} y1={0} x2={sx(0)} y2={h} stroke="rgba(255,255,255,0.12)" strokeWidth={1} />
      {/* Ticks */}
      {[-4, -2, 2, 4].map((v) => (
        <text key={v} x={sx(v)} y={sy(0) + 12} textAnchor="middle" fontSize={9} fill="#8b94a8">{v}</text>
      ))}
      {[1, 3].map((v) => (
        <text key={v} x={sx(0) - 6} y={sy(v) + 3} textAnchor="end" fontSize={9} fill="#8b94a8">{v}</text>
      ))}
      {/* Reference: identity line */}
      <line
        x1={sx(xMin)}
        y1={sy(xMin)}
        x2={sx(xMax)}
        y2={sy(xMax)}
        stroke="rgba(255,255,255,0.08)"
        strokeDasharray="3 3"
      />
      {/* SiLU curve */}
      <polyline points={pts.join(" ")} fill="none" stroke="#60a5fa" strokeWidth={2.5} />
    </svg>
  );
}

/* =============================== RESIDUAL =============================== */

function ResidualDetail() {
  return (
    <Section title="The residual stream">
      <p className="text-sm text-ink-300 leading-relaxed">
        Every block in Qwen reads from and writes to a residual stream. The
        attention sub-block adds a delta; the FFN sub-block adds another.
        Information flows through unchanged unless a layer chooses to
        intervene — making very deep networks trainable.
      </p>
      <BlockMath math={String.raw`x_{l+1} \;=\; x_l \;+\; \text{Attn}(\text{RMSNorm}(x_l)) \;+\; \text{FFN}(\text{RMSNorm}(\,\dots\,))`} />
    </Section>
  );
}

/* =============================== LM HEAD =============================== */

function LmHeadDetail({ config }: { config: QwenConfig }) {
  return (
    <>
      <Section icon={<Sparkles className="w-3.5 h-3.5" />} title="Project to the vocabulary">
        <p className="text-sm text-ink-300 leading-relaxed">
          A single linear map from <InlineMath math="d_{model}" /> to{" "}
          <InlineMath math="V \approx 152\text{k}" />. Logits in, softmax
          out, sampled token returned.
        </p>
        <BlockMath math={String.raw`\text{logits} = h_t \, W_{LM}^\top, \quad W_{LM} \in \mathbb{R}^{V \times d_{model}}`} />
      </Section>

      <Section title={config.tied_embeddings ? "Tied embeddings ✨" : "Untied embeddings"}>
        <p className="text-sm text-ink-300 leading-relaxed">
          {config.tied_embeddings ? (
            <>
              On this small Qwen variant, the LM head <strong>reuses</strong>{" "}
              the input embedding matrix:{" "}
              <InlineMath math="W_{LM} = W_E" />. Saves{" "}
              <span className="font-mono">{config.d_model.toLocaleString()} × {config.vocab.toLocaleString()}</span>{" "}
              parameters — a huge fraction of small models.
            </>
          ) : (
            <>
              Larger Qwen models keep <InlineMath math="W_{LM}" /> separate from{" "}
              <InlineMath math="W_E" />. With{" "}
              <span className="font-mono">{config.d_model.toLocaleString()}</span> hidden and{" "}
              <span className="font-mono">{config.vocab.toLocaleString()}</span> vocab, this is a{" "}
              <span className="font-mono">{((config.d_model * config.vocab) / 1e6).toFixed(0)}M</span>-param matrix in itself.
            </>
          )}
        </p>
      </Section>
    </>
  );
}

/* =============================== SAMPLING =============================== */

function SamplingDetail() {
  return (
    <>
      <Section icon={<Compass className="w-3.5 h-3.5" />} title="Logits → next token">
        <p className="text-sm text-ink-300 leading-relaxed">
          Softmax converts logits to a probability distribution. Then a
          sampling strategy (greedy / top-k / top-p / temperature) picks
          one. The chosen token is appended to the sequence and the whole
          pipeline runs again, autoregressively.
        </p>
        <BlockMath math={String.raw`p(\text{token}_t \mid \text{token}_{<t}) = \text{softmax}(\text{logits}_t / T)`} />
      </Section>
      <div className="rounded-xl border border-cyan-400/30 bg-cyan-400/[0.05] p-3 text-[12px] text-cyan-100 leading-relaxed">
        Want to dig into temperature, entropy, Jacobians? See the{" "}
        <a href="/softmax" className="underline text-cyan-300 hover:text-cyan-200">
          dedicated Softmax page
        </a>{" "}— it powers what happens here.
      </div>
    </>
  );
}

/* =================================================================== */
/* =============================== UTILS ============================= */
/* =================================================================== */

function Section({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-3.5 flex flex-col gap-2.5">
      <div className="flex items-center gap-2 text-[10.5px] uppercase tracking-wider text-ink-300 font-bold">
        {icon}
        {title}
      </div>
      {children}
    </div>
  );
}

function Metric({
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
    <div className="rounded-md bg-ink-950/70 border border-ink-800/80 p-2">
      <div className="text-[9px] uppercase tracking-wider text-ink-400">{label}</div>
      <div
        className="text-sm font-bold font-mono mt-0.5"
        style={color ? { color } : undefined}
      >
        {value}
      </div>
      {hint && <div className="text-[9px] text-ink-400 mt-0.5">{hint}</div>}
    </div>
  );
}

function HeatRow({ values, accent }: { values: number[]; accent?: string }) {
  const max = Math.max(...values.map((v) => Math.abs(v)), 0.001);
  return (
    <div className="flex h-3.5 rounded-md overflow-hidden border border-ink-800/80">
      {values.map((v, i) => {
        const t = Math.abs(v) / max;
        let bg: string;
        if (accent) {
          const a = Math.round(40 + 200 * t);
          bg = `${accent}${a.toString(16).padStart(2, "0")}`;
        } else {
          bg = v >= 0 ? `rgba(244,114,182,${0.15 + t * 0.85})` : `rgba(34,211,238,${0.15 + t * 0.85})`;
        }
        return (
          <div
            key={i}
            style={{ background: bg, width: `${100 / values.length}%` }}
            title={`d${i}: ${v.toFixed(3)}`}
          />
        );
      })}
    </div>
  );
}

function RowLabeled({
  label,
  values,
  accent,
}: {
  label: string;
  values: number[];
  accent?: string;
}) {
  return (
    <div className="grid grid-cols-[6rem_1fr] gap-2 items-center">
      <span className="text-[11px] font-mono" style={accent ? { color: accent } : undefined}>
        {label}
      </span>
      <HeatRow values={values} accent={accent} />
    </div>
  );
}

function Arrow({ centerLabel }: { centerLabel?: string }) {
  return (
    <div className="grid grid-cols-[6rem_1fr] gap-2 items-center text-[10px] text-ink-400">
      <span></span>
      <div className="flex items-center gap-2">
        <ArrowRight className="w-3 h-3" />
        {centerLabel && <span className="text-ink-300">{centerLabel}</span>}
      </div>
    </div>
  );
}

function Heatmap({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const n = matrix.length;
  const cellSize = 28;
  const labelW = 70;
  const w = labelW + n * cellSize + 8;
  const h = labelW + n * cellSize + 8;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ maxHeight: 320 }}>
      {/* Top labels */}
      {labels.slice(0, n).map((t, j) => (
        <text
          key={`top-${j}`}
          x={labelW + j * cellSize + cellSize / 2}
          y={labelW - 8}
          textAnchor="end"
          fontSize={9}
          fill="#a3a8b8"
          fontFamily="JetBrains Mono, monospace"
          transform={`rotate(-40, ${labelW + j * cellSize + cellSize / 2}, ${labelW - 8})`}
        >
          {t}
        </text>
      ))}
      {/* Left labels */}
      {labels.slice(0, n).map((t, i) => (
        <text
          key={`left-${i}`}
          x={labelW - 4}
          y={labelW + i * cellSize + cellSize / 2 + 3}
          textAnchor="end"
          fontSize={9}
          fill="#a3a8b8"
          fontFamily="JetBrains Mono, monospace"
        >
          {t}
        </text>
      ))}
      {/* Cells */}
      {matrix.map((row, i) =>
        row.map((v, j) => {
          const intensity = v;
          const bg = j > i ? "rgba(30,32,42,0.4)" : `rgba(124,92,255,${0.1 + intensity * 0.85})`;
          return (
            <rect
              key={`${i}-${j}`}
              x={labelW + j * cellSize + 1}
              y={labelW + i * cellSize + 1}
              width={cellSize - 2}
              height={cellSize - 2}
              rx={4}
              fill={bg}
              stroke="rgba(255,255,255,0.04)"
            />
          );
        })
      )}
    </svg>
  );
}
