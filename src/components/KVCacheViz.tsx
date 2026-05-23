import { useEffect, useMemo, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  cacheBytesAt,
  cumulativeFlops,
  flopsRecompute,
  flopsWithCache,
  formatBytes,
  formatFlops,
  fragmentation,
  paginate,
  totalBlocks,
  type KVCacheConfig,
} from "../lib/kvCache";

const C_AXIS = "rgba(255,255,255,0.10)";
const C_Q = "#a78bfa";
const C_K = "#22d3ee";
const C_V = "#34d399";
const C_GREEN = "#34d399";
const C_AMBER = "#fbbf24";
const C_PINK = "#f472b6";

const SAMPLE_TOKENS = [
  "The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog",
  ".", " It", " was", " a", " sunny", " day",
];

export default function KVCacheViz() {
  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <AutoregressivePlayback />
      <MemoryVsLengthPanel />
      <FlopsCounter />
      <PagedAttentionPanel />

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
          At step <InlineMath math={String.raw`t`} /> the only{" "}
          <em>new</em> attention work is computing one Q row and dotting it
          against the cached <InlineMath math={String.raw`K_{:t}, V_{:t}`} />.
          The <span className="text-cyan-300">KV cache</span> stores those
          accumulated keys and values so we never re-project the prefix. It is
          what turns autoregressive decoding from quadratic to{" "}
          <span className="text-emerald-300">linear</span> per token.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`q_t = x_t W_Q,\quad k_t = x_t W_K,\quad v_t = x_t W_V`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`K_{:t} \leftarrow [K_{:t-1}; k_t],\;\; V_{:t} \leftarrow [V_{:t-1}; v_t]`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`o_t = \mathrm{softmax}\!\left(\tfrac{q_t K_{:t}^\top}{\sqrt{d_h}}\right) V_{:t}`} />
      </MathBox>
    </section>
  );
}

/* =============================== autoregressive playback =============================== */

function AutoregressivePlayback() {
  const tokens = SAMPLE_TOKENS;
  const dHead = 8; // # cells per K/V row, visualized
  const [step, setStep] = useState(3);
  const [playing, setPlaying] = useState(false);

  // Animation loop
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setStep((s) => (s + 1 > tokens.length ? 1 : s + 1));
    }, 700);
    return () => window.clearInterval(id);
  }, [playing, tokens.length]);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Autoregressive playback · K/V grow one row per token
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPlaying((p) => !p)}
            className="text-[10px] font-mono text-ink-300 hover:text-white border border-ink-700 px-2 py-0.5 rounded"
          >
            {playing ? "pause" : "play"}
          </button>
          <button
            onClick={() => {
              setStep(1);
              setPlaying(false);
            }}
            className="text-[10px] font-mono text-ink-400 hover:text-white border border-ink-700 px-2 py-0.5 rounded"
          >
            reset
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-1.5 px-1">
        {tokens.map((t, i) => (
          <span
            key={i}
            className={`px-1.5 py-0.5 rounded text-[12px] font-mono ${
              i < step
                ? "bg-violet-500/20 text-violet-100"
                : i === step
                  ? "bg-amber-500/30 text-amber-100 outline outline-1 outline-amber-300"
                  : "bg-ink-900 text-ink-500"
            }`}
          >
            {t === " " ? "·" : t.replace(/ /g, "·")}
          </span>
        ))}
      </div>

      <CacheGrid step={step} tokens={tokens} dHead={dHead} />

      <SliderRow
        label="step t"
        value={step}
        onChange={(v) => setStep(Math.round(v))}
        min={1}
        max={tokens.length}
        step={1}
        fmt={(v) => v.toFixed(0)}
      />

      <p className="text-[11px] text-ink-400 leading-snug">
        Only the highlighted Q row (the latest token) is freshly computed; the
        K and V rows below it accumulate from previous steps and stay constant.
        That asymmetry — one row of work + a growing context — is the entire
        reason inference latency scales linearly, not quadratically, in
        sequence length.
      </p>
    </section>
  );
}

function CacheGrid({
  step,
  tokens,
  dHead,
}: {
  step: number;
  tokens: string[];
  dHead: number;
}) {
  const L = tokens.length;
  const W = 720;
  const H = 320;
  const PAD = { l: 80, r: 16, t: 22, b: 14 };
  const innerW = W - PAD.l - PAD.r;
  const rowH = (H - PAD.t - PAD.b) / (L + 1.2); // a little spacing

  const colorFor = (kind: "Q" | "K" | "V", t: number) => {
    if (t >= step) return "rgba(48,53,72,0.45)";
    return kind === "Q" ? C_Q : kind === "K" ? C_K : C_V;
  };

  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        <text x={PAD.l - 6} y={PAD.t - 6} fontSize={10} fill="#7a8094" fontFamily="JetBrains Mono, monospace" textAnchor="end">
          token
        </text>
        <text x={PAD.l + 6} y={PAD.t - 6} fontSize={10} fill={C_Q} fontFamily="JetBrains Mono, monospace" fontWeight={700}>
          Q · K · V  (one row per token, d_head = {dHead})
        </text>

        {tokens.map((tok, i) => {
          const y = PAD.t + i * rowH;
          const isCurrent = i === step - 1;
          const isPast = i < step - 1;
          return (
            <g key={i}>
              <text x={PAD.l - 8} y={y + rowH * 0.62} fontSize={10} fill={isCurrent ? "#fef3c7" : isPast ? "#cbd5e1" : "#475569"} fontFamily="JetBrains Mono, monospace" textAnchor="end">
                {i + 1}: {tok.replace(/ /g, "·").slice(0, 8)}
              </text>
            </g>
          );
        })}

        {/* Three vertical bands: Q (only step row), K (rows 1..step), V (rows 1..step) */}
        <BandLabel x={PAD.l + 6} y={PAD.t - 6 + 0} color={C_Q} label="Q (current row)" innerW={innerW / 3.05 - 6} />
        <BandLabel x={PAD.l + innerW / 3.05 + 6} y={PAD.t - 6 + 0} color={C_K} label="K cache" innerW={innerW / 3.05 - 6} />
        <BandLabel x={PAD.l + 2 * innerW / 3.05 + 6} y={PAD.t - 6 + 0} color={C_V} label="V cache" innerW={innerW / 3.05 - 6} />

        {/* Render Q band */}
        {Array.from({ length: dHead }, (_, j) => {
          const bandW = innerW / 3.05;
          const cw = bandW / dHead;
          const ch = rowH * 0.86;
          const xBand = PAD.l;
          const y = PAD.t + (step - 1) * rowH;
          return (
            <rect
              key={`q-${j}`}
              x={xBand + j * cw}
              y={y}
              width={cw + 0.5}
              height={ch}
              fill={colorFor("Q", step - 1)}
              opacity={0.85}
            />
          );
        })}

        {/* Render K band rows */}
        {Array.from({ length: step }, (_, i) =>
          Array.from({ length: dHead }, (_, j) => {
            const bandW = innerW / 3.05;
            const cw = bandW / dHead;
            const ch = rowH * 0.86;
            const xBand = PAD.l + bandW + 8;
            const y = PAD.t + i * rowH;
              return (
              <rect
                key={`k-${i}-${j}`}
                x={xBand + j * cw}
                y={y}
                width={cw + 0.5}
                height={ch}
                fill={colorFor("K", i)}
                opacity={i === step - 1 ? 0.95 : 0.55}
                stroke={i === step - 1 ? "#fff" : "none"}
                strokeOpacity={0.35}
                strokeWidth={i === step - 1 ? 0.6 : 0}
              />
            );
          })
        )}

        {/* Render V band rows */}
        {Array.from({ length: step }, (_, i) =>
          Array.from({ length: dHead }, (_, j) => {
            const bandW = innerW / 3.05;
            const cw = bandW / dHead;
            const ch = rowH * 0.86;
            const xBand = PAD.l + 2 * bandW + 16;
            const y = PAD.t + i * rowH;
              return (
              <rect
                key={`v-${i}-${j}`}
                x={xBand + j * cw}
                y={y}
                width={cw + 0.5}
                height={ch}
                fill={colorFor("V", i)}
                opacity={i === step - 1 ? 0.95 : 0.55}
                stroke={i === step - 1 ? "#fff" : "none"}
                strokeOpacity={0.35}
                strokeWidth={i === step - 1 ? 0.6 : 0}
              />
            );
          })
        )}
      </svg>
    </div>
  );
}

function BandLabel({
  x,
  y,
  color,
  label,
  innerW,
}: {
  x: number;
  y: number;
  color: string;
  label: string;
  innerW: number;
}) {
  return (
    <g>
      <text x={x + innerW / 2} y={y} fontSize={10} fill={color} fontFamily="JetBrains Mono, monospace" fontWeight={700} textAnchor="middle">
        {label}
      </text>
    </g>
  );
}

/* =============================== memory vs length =============================== */

function MemoryVsLengthPanel() {
  const [nHeadsKv, setNHeadsKv] = useState(8);
  const [dHead, setDHead] = useState(128);
  const [nLayers, setNLayers] = useState(32);
  const [batch, setBatch] = useState(1);

  const cfg: KVCacheConfig = { nHeadsKv, dHead, nLayers, batch, dtypeBytes: 2 };

  const W = 720;
  const H = 240;
  const PAD = { l: 56, r: 14, t: 18, b: 28 };
  const xMax = 131072; // 128k
  const yMax = Math.max(cacheBytesAt(cfg, xMax), 1);

  const sx = (v: number) => PAD.l + (v / xMax) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - (v / yMax) * (H - PAD.t - PAD.b);

  const path = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    for (let l = 0; l <= xMax; l += 1024) pts.push({ x: l, y: cacheBytesAt(cfg, l) });
    return pts.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x).toFixed(2)} ${sy(p.y).toFixed(2)}`).join(" ");
  }, [cfg, sx, sy]);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          KV cache vs sequence length
        </span>
        <span className="text-[10px] font-mono text-ink-500">
          {formatBytes(cacheBytesAt(cfg, 8192))} @ 8k tokens
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
        <SliderRow label="H_kv" value={nHeadsKv} onChange={(v) => setNHeadsKv(Math.round(v))} min={1} max={64} step={1} fmt={(v) => v.toFixed(0)} color={C_K} />
        <SliderRow label="d_head" value={dHead} onChange={(v) => setDHead(Math.round(v))} min={32} max={256} step={8} fmt={(v) => v.toFixed(0)} color={C_K} />
        <SliderRow label="layers" value={nLayers} onChange={(v) => setNLayers(Math.round(v))} min={1} max={120} step={1} fmt={(v) => v.toFixed(0)} color={C_V} />
        <SliderRow label="batch B" value={batch} onChange={(v) => setBatch(Math.round(v))} min={1} max={32} step={1} fmt={(v) => v.toFixed(0)} color={C_V} />
      </div>

      <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
          <Grid sx={sx} sy={sy} yTicks={[0, yMax / 4, yMax / 2, (3 * yMax) / 4, yMax]} yFmt={(v) => formatBytes(v)} xTicks={[0, 16384, 32768, 65536, 98304, 131072]} xFmt={(v) => (v >= 1024 ? `${v / 1024}k` : v.toString())} W={W} H={H} PAD={PAD} />
          <path d={path} stroke={C_PINK} strokeWidth={2.4} fill="none" />
          {[2048, 8192, 32768].map((L) => (
            <g key={L}>
              <circle cx={sx(L)} cy={sy(cacheBytesAt(cfg, L))} r={3.5} fill={C_PINK} />
              <text x={sx(L) + 6} y={sy(cacheBytesAt(cfg, L)) - 6} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                {L >= 1024 ? `${L / 1024}k` : L}: {formatBytes(cacheBytesAt(cfg, L))}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </section>
  );
}

/* =============================== flops counter =============================== */

function FlopsCounter() {
  const cfg: KVCacheConfig = { nHeadsKv: 32, dHead: 128, nLayers: 32, batch: 1, dtypeBytes: 2 };
  const [seqLen, setSeqLen] = useState(2048);

  const stepCache = flopsWithCache(cfg, seqLen);
  const stepRecompute = flopsRecompute(cfg, seqLen);
  const cumCache = cumulativeFlops(cfg, seqLen, true);
  const cumRecompute = cumulativeFlops(cfg, seqLen, false);
  const ratio = cumRecompute / Math.max(cumCache, 1);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          With cache vs without · attention FLOPs
        </span>
        <BlockMath math={String.raw`O_{\text{cache}}(L) = O(L),\quad O_{\text{recompute}}(L) = O(L^2)`} />
      </div>

      <SliderRow label="seq_len L" value={seqLen} onChange={(v) => setSeqLen(Math.round(v))} min={64} max={8192} step={64} fmt={(v) => v.toFixed(0)} color={C_GREEN} />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 font-mono text-[12px]">
        <Stat label="Per-step (cache)" value={formatFlops(stepCache)} color={C_GREEN} hint="O(L)" />
        <Stat label="Per-step (recompute)" value={formatFlops(stepRecompute)} color={C_AMBER} hint="O(L²)" />
        <Stat label="Cumulative (cache)" value={formatFlops(cumCache)} color={C_GREEN} hint="O(L²)" />
        <Stat label="Cumulative (recompute)" value={formatFlops(cumRecompute)} color={C_AMBER} hint="O(L³)" />
      </div>

      <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-3 flex items-center gap-3">
        <span className="text-[11px] font-mono text-ink-300">
          recompute / cache ratio at L = {seqLen}:
        </span>
        <span className="text-2xl font-bold tabular-nums text-amber-300">
          {ratio < 100 ? ratio.toFixed(1) : ratio.toExponential(1)}×
        </span>
        <span className="text-[11px] font-mono text-ink-500 ml-auto">
          the cache makes decoding ~L× cheaper
        </span>
      </div>
    </section>
  );
}

/* =============================== paged attention =============================== */

function PagedAttentionPanel() {
  const [blockSize, setBlockSize] = useState(16);
  const seqLensRaw: number[] = [40, 60, 25, 80, 12];

  const blocks = paginate(seqLensRaw, blockSize);
  const total = totalBlocks(seqLensRaw, blockSize);
  const frag = fragmentation(seqLensRaw, blockSize);

  const W = 720;
  const H = 160;
  const PAD = { l: 80, r: 12, t: 14, b: 12 };
  const cell = 18;
  const maxBlocksPerReq = Math.max(...blocks.map((bs) => bs.length));

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Paged attention · vLLM-style block table
        </span>
        <BlockMath math={String.raw`\text{cache} = \sum_r \lceil L_r / B_s\rceil \cdot B_s \cdot \text{(per-token bytes)}`} />
      </div>

      <SliderRow label="block size B_s" value={blockSize} onChange={(v) => setBlockSize(Math.round(v))} min={4} max={64} step={2} fmt={(v) => v.toFixed(0)} color={C_PINK} />

      <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
          {blocks.map((bs, r) => (
            <g key={r}>
              <text x={PAD.l - 8} y={PAD.t + r * (cell + 6) + cell * 0.72} textAnchor="end" fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                req {r} · L={seqLensRaw[r]}
              </text>
              {Array.from({ length: maxBlocksPerReq }, (_, b) => {
                const blk = bs[b];
                if (!blk) {
                  return <rect key={b} x={PAD.l + b * (cell + 2)} y={PAD.t + r * (cell + 6)} width={cell} height={cell} rx={3} fill="rgba(60,67,89,0.35)" />;
                }
                const fill = blk.tokens === blockSize ? C_K : C_AMBER;
                return (
                  <g key={b}>
                    <rect x={PAD.l + b * (cell + 2)} y={PAD.t + r * (cell + 6)} width={cell} height={cell} rx={3} fill={fill} fillOpacity={0.7} stroke={fill} />
                    <text x={PAD.l + b * (cell + 2) + cell / 2} y={PAD.t + r * (cell + 6) + cell * 0.72} fontSize={8} fill="#0b0f19" fontFamily="JetBrains Mono, monospace" textAnchor="middle" fontWeight={700}>
                      {blk.tokens}
                    </text>
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 font-mono text-[12px]">
        <Stat label="blocks allocated" value={total.toString()} color={C_K} />
        <Stat label="capacity" value={`${total * blockSize} tokens`} color={C_K} />
        <Stat label="used" value={`${seqLensRaw.reduce((s, L) => s + L, 0)} tokens`} color={C_GREEN} />
        <Stat label="fragmentation" value={`${(frag.ratio * 100).toFixed(1)}%`} color={C_AMBER} hint={`${frag.wasted} wasted slots`} />
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        Without paging, every active request reserves the worst-case max_seq_len
        contiguous slab — most of which stays empty. Paging lets the GPU
        allocate small fixed-size blocks on demand (cyan = full, amber =
        partial). This is the trick that lets vLLM serve many concurrent
        requests on the same KV memory budget.
      </p>
    </section>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="Time complexity per token">
        <InlineMath math={String.raw`O(L \cdot d_h)\text{ with cache vs } O(L^2 d_h)\text{ without}`} />
      </PropCard>
      <PropCard title="Total decode cost">
        <InlineMath math={String.raw`O(L^2 d_h)\text{ vs } O(L^3 d_h)`} />
      </PropCard>
      <PropCard title="Cache size">
        <InlineMath math={String.raw`2\cdot B\cdot H_{kv}\cdot d_h\cdot L\cdot N_l\cdot b`} />
      </PropCard>
      <PropCard title="Memory-bound decoding">
        <InlineMath math={String.raw`\text{HBM bandwidth} \gg \text{FLOPs at decode}`} />
      </PropCard>
      <PropCard title="Quantization wins">
        <InlineMath math={String.raw`b: 2 \to 1\text{ halves cache, doubles capacity}`} />
      </PropCard>
      <PropCard title="Paged blocks">
        <InlineMath math={String.raw`\#\text{blocks}_r = \lceil L_r / B_s\rceil`} />
      </PropCard>
      <PropCard title="Fragmentation upper bound">
        <InlineMath math={String.raw`\text{waste} \le B_s - 1\text{ tokens per request}`} />
      </PropCard>
      <PropCard title="Prefix sharing">
        <InlineMath math={String.raw`\text{shared prompt} \Rightarrow \text{shared blocks}`} />
      </PropCard>
      <PropCard title="Speculative decoding">
        <InlineMath math={String.raw`\text{verify multiple } q_t\text{ against same KV cache}`} />
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
        <UsageItem head="Every decoder LLM" body="GPT, LLaMA, Claude, Gemini, Qwen, Mistral — autoregressive decoding without a KV cache is unthinkable." />
        <UsageItem head="vLLM / TGI / SGLang" body="paged attention is the default backend layout." />
        <UsageItem head="Continuous batching" body="cache lets requests at different positions share a forward pass." />
        <UsageItem head="Prefix caching" body="shared system prompts cached across requests (Claude, vLLM)." />
        <UsageItem head="Speculative decoding" body="cache is reused while a draft model proposes multiple tokens." />
        <UsageItem head="Long-context inference" body="100k+ contexts only feasible with KV quant + paged attention." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-emerald-300">
        <UsageItem head="Quadratic → linear per step" body="reuse instead of recompute the prefix every token." />
        <UsageItem head="Predictable memory" body="cache size is closed-form, easy to budget." />
        <UsageItem head="Composable with GQA / MLA" body="orthogonal optimizations stack multiplicatively." />
        <UsageItem head="GPU bandwidth fit" body="HBM is the real bottleneck; smaller cache = faster decode." />
        <UsageItem head="Paging eliminates max-len waste" body="serve many requests per GPU with no contiguous allocation." />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem head="Cache grows linearly in L" body="at L=128k, K+V eclipses model weights." />
        <UsageItem head="Bandwidth-bound at scale" body="loading the cache from HBM each step is the limit, not FLOPs." />
        <UsageItem head="Cache invalidation on edit" body="any prefix change re-prefills the whole cache." />
        <UsageItem head="Multi-turn churn" body="long conversations keep evicting / re-prefilling." />
        <UsageItem head="MLA needs decompression" body="latent storage saves memory but adds per-step matmul." />
        <UsageItem head="Streaming decode + paging" body="block management has nontrivial CUDA-side cost." />
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
      <span className="w-16 text-right tabular-nums">
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
  yFmt?: (v: number) => string;
  xFmt?: (v: number) => string;
}

function Grid({ sx, sy, yTicks, xTicks, W, H, PAD, yFmt, xFmt }: GridProps) {
  return (
    <g>
      <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke={C_AXIS} />
      <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke={C_AXIS} />
      {yTicks.map((v, i) => (
        <g key={`y-${i}`}>
          <line x1={PAD.l - 4} y1={sy(v)} x2={W - PAD.r} y2={sy(v)} stroke="rgba(255,255,255,0.04)" />
          <text x={PAD.l - 6} y={sy(v) + 3} textAnchor="end" fontSize={9} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
            {yFmt ? yFmt(v) : v.toString()}
          </text>
        </g>
      ))}
      {xTicks.map((v, i) => (
        <g key={`x-${i}`}>
          <line x1={sx(v)} y1={H - PAD.b} x2={sx(v)} y2={H - PAD.b + 4} stroke={C_AXIS} />
          <text x={sx(v)} y={H - PAD.b + 14} textAnchor="middle" fontSize={9} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
            {xFmt ? xFmt(v) : v.toString()}
          </text>
        </g>
      ))}
    </g>
  );
}
