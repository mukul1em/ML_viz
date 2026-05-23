import { useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  MODEL_PRESETS,
  VARIANTS,
  effectiveKvDim,
  effectiveKvHeads,
  formatBytes,
  kvCacheBytes,
  type VariantConfig,
  type VariantId,
} from "../lib/attentionVariants";

const C_Q = "#a78bfa";
const C_K = "#22d3ee";
const C_V = "#34d399";
const C_LATENT = "#f472b6";

const VARIANT_ORDER: VariantId[] = ["mha", "gqa", "mqa", "mla"];

export default function AttentionVariantsViz() {
  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <VariantsLab />
      <KVMemoryBar />
      <ModelTable />

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
          MHA gives every Q head its own K and V — clean but the KV cache
          dominates inference memory. <span className="text-cyan-300">MQA</span>{" "}
          shares one K/V across all heads,{" "}
          <span className="text-pink-300">GQA</span> groups them, and{" "}
          <span className="text-amber-300">MLA</span> compresses K/V to a tiny
          shared latent. All four compute attention with the same formula —
          they only differ in how many K/V projections live in memory.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`\text{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_h}}\right)V`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`|\mathrm{KV}|_{\text{bytes}} = 2 \cdot B \cdot H_{kv} \cdot d_{kv} \cdot L \cdot N_l \cdot b`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\mathrm{MHA}: H_{kv}{=}H_q\ \ \mathrm{GQA}: H_{kv}{=}H_q/g\ \ \mathrm{MQA}: H_{kv}{=}1`} />
      </MathBox>
    </section>
  );
}

/* =============================== variants lab =============================== */

function VariantsLab() {
  const [variant, setVariant] = useState<VariantId>("gqa");
  const [nHeadsQ, setNHeadsQ] = useState(32);
  const [groupSize, setGroupSize] = useState(8);
  const [dHead, setDHead] = useState(128);
  const [seqLen, setSeqLen] = useState(4096);
  const [nLayers, setNLayers] = useState(32);
  const [batch, setBatch] = useState(1);

  const cfg: VariantConfig = {
    nHeadsQ,
    groupSize,
    dHead,
    seqLen,
    nLayers,
    batch,
    dtypeBytes: 2,
  };

  const headsKv = effectiveKvHeads(variant, cfg);
  const dKv = effectiveKvDim(variant, cfg);
  const cache = kvCacheBytes(variant, cfg);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Variant lab · head grouping + KV-cache math
        </span>
        <VariantToggleBar value={variant} onChange={setVariant} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-4">
        <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-3">
          <HeadDiagram variant={variant} nHeadsQ={nHeadsQ} groupSize={groupSize} />
        </div>

        <div className="flex flex-col gap-3">
          <SliderRow label="n_heads_q" value={nHeadsQ} onChange={(v) => setNHeadsQ(Math.max(1, Math.round(v)))} min={1} max={64} step={1} fmt={(v) => v.toFixed(0)} color={C_Q} />
          {variant === "gqa" && (
            <SliderRow
              label="group size g"
              value={groupSize}
              onChange={(v) => {
                const g = Math.max(1, Math.round(v));
                setGroupSize(g);
              }}
              min={1}
              max={nHeadsQ}
              step={1}
              fmt={(v) => v.toFixed(0)}
              color={C_K}
            />
          )}
          <SliderRow label="d_head" value={dHead} onChange={(v) => setDHead(Math.round(v))} min={32} max={256} step={8} fmt={(v) => v.toFixed(0)} color={C_K} />
          <SliderRow label="seq_len L" value={seqLen} onChange={(v) => setSeqLen(Math.round(v))} min={128} max={32768} step={128} fmt={(v) => v.toFixed(0)} color={C_V} />
          <SliderRow label="n_layers" value={nLayers} onChange={(v) => setNLayers(Math.round(v))} min={1} max={120} step={1} fmt={(v) => v.toFixed(0)} color={C_V} />
          <SliderRow label="batch B" value={batch} onChange={(v) => setBatch(Math.round(v))} min={1} max={64} step={1} fmt={(v) => v.toFixed(0)} color={C_V} />

          <hr className="border-ink-800/80 my-1" />
          <div className="grid grid-cols-2 gap-2 text-[12px] font-mono">
            <Stat label="H_kv (effective)" value={headsKv.toString()} color={C_K} hint={`of ${nHeadsQ} Q heads`} />
            <Stat label="d_kv" value={dKv.toString()} color={C_K} hint={variant === "mla" ? "compressed" : "= d_head"} />
            <Stat label="KV cache total" value={formatBytes(cache)} color={C_LATENT} hint={`@ ${batch}·B`} />
            <Stat label="ratio vs MHA" value={`${(cache / kvCacheBytes("mha", cfg)).toFixed(2)}×`} color="#fff" />
          </div>

          <p className="text-[11px] text-ink-400 leading-snug">
            The whole point of MQA / GQA / MLA: shrink <InlineMath math={String.raw`H_{kv}`} /> (or{" "}
            <InlineMath math={String.raw`d_{kv}`} />) so KV-cache memory stops being the inference
            bottleneck. Quality stays close to MHA when grouping is moderate (LLaMA-3 ships 8:1).
          </p>
        </div>
      </div>
    </section>
  );
}

function VariantToggleBar({ value, onChange }: { value: VariantId; onChange: (v: VariantId) => void }) {
  return (
    <div className="flex gap-1 text-[11px] font-mono">
      {VARIANT_ORDER.map((id) => (
        <button
          key={id}
          onClick={() => onChange(id)}
          className={`px-2 py-1 rounded border ${
            value === id
              ? "border-violet-400/60 bg-violet-500/10 text-violet-200"
              : "border-ink-700 text-ink-400 hover:text-white"
          }`}
        >
          {VARIANTS[id].short}
        </button>
      ))}
    </div>
  );
}

function HeadDiagram({
  variant,
  nHeadsQ,
  groupSize,
}: {
  variant: VariantId;
  nHeadsQ: number;
  groupSize: number;
}) {
  const W = 580;
  const H = 240;
  const PAD = 14;
  const headW = (W - PAD * 2) / nHeadsQ;
  const qY = 36;
  const kY = 110;
  const vY = 174;
  const boxH = 26;

  let kvBlocks: { start: number; end: number; idx: number }[] = [];
  if (variant === "mha") {
    kvBlocks = Array.from({ length: nHeadsQ }, (_, i) => ({ start: i, end: i + 1, idx: i }));
  } else if (variant === "mqa" || variant === "mla") {
    kvBlocks = [{ start: 0, end: nHeadsQ, idx: 0 }];
  } else if (variant === "gqa") {
    const g = Math.max(1, Math.min(groupSize, nHeadsQ));
    const numGroups = Math.max(1, Math.ceil(nHeadsQ / g));
    kvBlocks = Array.from({ length: numGroups }, (_, idx) => ({
      start: idx * g,
      end: Math.min((idx + 1) * g, nHeadsQ),
      idx,
    }));
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
      {/* labels */}
      <text x={6} y={qY + boxH / 2 + 4} fontSize={10} fill={C_Q} fontFamily="JetBrains Mono, monospace" fontWeight={700}>Q</text>
      <text x={6} y={kY + boxH / 2 + 4} fontSize={10} fill={C_K} fontFamily="JetBrains Mono, monospace" fontWeight={700}>K</text>
      <text x={6} y={vY + boxH / 2 + 4} fontSize={10} fill={C_V} fontFamily="JetBrains Mono, monospace" fontWeight={700}>V</text>

      {/* Q heads */}
      {Array.from({ length: nHeadsQ }, (_, i) => (
        <g key={`q-${i}`}>
          <rect
            x={PAD + i * headW + 1}
            y={qY}
            width={headW - 2}
            height={boxH}
            rx={3}
            fill={C_Q}
            fillOpacity={0.18}
            stroke={C_Q}
            strokeOpacity={0.6}
          />
        </g>
      ))}

      {/* K + V blocks */}
      {kvBlocks.map((blk) => {
        const x = PAD + blk.start * headW + 1;
        const w = (blk.end - blk.start) * headW - 2;
        const isLatent = variant === "mla";
        return (
          <g key={`kv-${blk.idx}`}>
            <rect x={x} y={kY} width={w} height={boxH} rx={3} fill={isLatent ? C_LATENT : C_K} fillOpacity={0.2} stroke={isLatent ? C_LATENT : C_K} strokeOpacity={0.7} />
            <rect x={x} y={vY} width={w} height={boxH} rx={3} fill={isLatent ? C_LATENT : C_V} fillOpacity={0.2} stroke={isLatent ? C_LATENT : C_V} strokeOpacity={0.7} />
            {/* fan-out lines from K/V centroid up to each covered Q head */}
            {Array.from({ length: blk.end - blk.start }, (_, k) => {
              const qx = PAD + (blk.start + k) * headW + headW / 2;
              const kx = x + w / 2;
              return (
                <line
                  key={`l-${blk.idx}-${k}`}
                  x1={qx}
                  y1={qY + boxH}
                  x2={kx}
                  y2={kY}
                  stroke="rgba(255,255,255,0.18)"
                  strokeWidth={0.8}
                />
              );
            })}
          </g>
        );
      })}

      <text x={W - PAD} y={H - 12} textAnchor="end" fontSize={10} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
        {variant === "mla" ? "K/V projected to latent → decompress per head" : `K/V heads = ${kvBlocks.length}`}
      </text>
    </svg>
  );
}

/* =============================== memory bar comparison =============================== */

function KVMemoryBar() {
  const cfg: VariantConfig = {
    nHeadsQ: 32,
    groupSize: 8,
    dHead: 128,
    seqLen: 8192,
    nLayers: 32,
    batch: 1,
    dtypeBytes: 2,
  };

  const rows = VARIANT_ORDER.map((v) => ({ id: v, bytes: kvCacheBytes(v, cfg) }));
  const maxBytes = Math.max(...rows.map((r) => r.bytes));

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          KV-cache memory comparison · 32 Q heads, d=128, L=8192, 32 layers, fp16
        </span>
      </div>

      <div className="flex flex-col gap-2">
        {rows.map((r) => {
          const pct = r.bytes / maxBytes;
          const color =
            r.id === "mha" ? C_Q : r.id === "gqa" ? C_K : r.id === "mqa" ? C_V : C_LATENT;
          return (
            <div key={r.id} className="flex items-center gap-3 font-mono text-[12px]">
              <span className="w-32 text-ink-200 font-semibold">{VARIANTS[r.id].short}</span>
              <div className="flex-1 h-5 rounded bg-ink-950/80 border border-ink-800/60 overflow-hidden">
                <div className="h-full rounded" style={{ width: `${pct * 100}%`, background: color, opacity: 0.85 }} />
              </div>
              <span className="w-24 text-right tabular-nums">{formatBytes(r.bytes)}</span>
              <span className="w-20 text-right text-ink-500">
                {r.id === "mha" ? "1.00×" : `${(r.bytes / rows[0].bytes).toFixed(2)}×`}
              </span>
            </div>
          );
        })}
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        For an 8k-token context, MHA needs <strong>{formatBytes(rows[0].bytes)}</strong> per
        request just for K and V. GQA with 8:1 grouping is{" "}
        <strong>{(rows[0].bytes / rows[1].bytes).toFixed(1)}× smaller</strong>, MQA{" "}
        <strong>{(rows[0].bytes / rows[2].bytes).toFixed(0)}×</strong>, and MLA roughly
        <strong> {(rows[0].bytes / rows[3].bytes).toFixed(0)}×</strong>. That ratio is
        what lets a 70B model serve dozens of concurrent requests on a single H100.
      </p>
    </section>
  );
}

/* =============================== model table =============================== */

function ModelTable() {
  const seqLen = 8192;
  const dtypeBytes = 2;
  const batch = 1;

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Real models · per-request KV cache @ L = {seqLen} tokens
        </span>
      </div>

      <div className="overflow-x-auto -mx-2 px-2">
        <table className="w-full text-[12px] font-mono">
          <thead>
            <tr className="text-left text-ink-500 border-b border-ink-800/60">
              <th className="py-1.5 pr-3">Model</th>
              <th className="py-1.5 pr-3">Variant</th>
              <th className="py-1.5 pr-3">H_q</th>
              <th className="py-1.5 pr-3">H_kv</th>
              <th className="py-1.5 pr-3">d_head</th>
              <th className="py-1.5 pr-3">layers</th>
              <th className="py-1.5 pr-3 text-right">KV cache</th>
            </tr>
          </thead>
          <tbody>
            {MODEL_PRESETS.map((m) => {
              const cfg: VariantConfig = {
                nHeadsQ: m.nHeadsQ,
                groupSize: m.groupSize,
                dHead: m.dHead,
                seqLen,
                nLayers: m.nLayers,
                batch,
                dtypeBytes,
              };
              const headsKv = effectiveKvHeads(m.variant, cfg);
              const cache = kvCacheBytes(m.variant, cfg);
              return (
                <tr key={m.name} className="border-b border-ink-900/60 hover:bg-ink-900/40">
                  <td className="py-1.5 pr-3 text-ink-100">{m.name}</td>
                  <td className="py-1.5 pr-3">
                    <span
                      className="px-1.5 py-0.5 rounded text-[10px]"
                      style={{
                        background:
                          m.variant === "mha"
                            ? "rgba(167,139,250,0.18)"
                            : m.variant === "gqa"
                              ? "rgba(34,211,238,0.18)"
                              : m.variant === "mqa"
                                ? "rgba(52,211,153,0.18)"
                                : "rgba(244,114,182,0.18)",
                        color:
                          m.variant === "mha"
                            ? C_Q
                            : m.variant === "gqa"
                              ? C_K
                              : m.variant === "mqa"
                                ? C_V
                                : C_LATENT,
                      }}
                    >
                      {VARIANTS[m.variant].short}
                    </span>
                  </td>
                  <td className="py-1.5 pr-3 text-ink-200 tabular-nums">{m.nHeadsQ}</td>
                  <td className="py-1.5 pr-3 text-ink-200 tabular-nums">{headsKv}</td>
                  <td className="py-1.5 pr-3 text-ink-200 tabular-nums">{m.dHead}</td>
                  <td className="py-1.5 pr-3 text-ink-200 tabular-nums">{m.nLayers}</td>
                  <td className="py-1.5 pr-3 text-right text-ink-100 tabular-nums">{formatBytes(cache)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        These numbers explain a lot of the LLaMA-3 → 3.1 → 3.2 architecture
        choices: dropping from 32 K/V heads to 4 (GQA 8:1) cuts the cache 8×
        with negligible accuracy loss, which is why every modern open model has
        adopted it.
      </p>
    </section>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="MHA params">
        <InlineMath math={String.raw`O(H_q\cdot d_h\cdot d_{model})\text{ for }K,V`} />
      </PropCard>
      <PropCard title="MQA params">
        <InlineMath math={String.raw`O(d_h\cdot d_{model})\text{ for }K,V`} />
      </PropCard>
      <PropCard title="GQA params">
        <InlineMath math={String.raw`O\!\left(\tfrac{H_q}{g}\cdot d_h\cdot d_{model}\right)`} />
      </PropCard>
      <PropCard title="MLA latent">
        <InlineMath math={String.raw`d_{kv}^{\text{latent}} \approx d_h/8`} />
      </PropCard>
      <PropCard title="KV cache (per layer)">
        <InlineMath math={String.raw`2\cdot B\cdot H_{kv}\cdot d_{kv}\cdot L\cdot b`} />
      </PropCard>
      <PropCard title="Decode FLOPs ratio">
        <InlineMath math={String.raw`\text{MHA}/\text{MQA} \to H_q\text{ at large }L`} />
      </PropCard>
      <PropCard title="Quality (empirical)">
        <InlineMath math={String.raw`\text{MHA}\approx\text{GQA}_{8:1} > \text{MQA}`} />
      </PropCard>
      <PropCard title="MLA decompress cost">
        <InlineMath math={String.raw`+\,O(H_q\cdot d_h^2)\text{ matmul/layer}`} />
      </PropCard>
      <PropCard title="Memory-bound regime">
        <InlineMath math={String.raw`\text{decode is bandwidth, not FLOP, limited}`} />
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
        <UsageItem head="MHA" body="GPT-2, BERT, LLaMA-1/2, GPT-3 — anything pre-2023." />
        <UsageItem head="MQA" body="PaLM, Falcon-180B, StarCoder. Smallest cache, fast decode." />
        <UsageItem head="GQA (g=8)" body="LLaMA-3, Mistral 7B, Mixtral 8×7B, Qwen-2/2.5 — current default." />
        <UsageItem head="MLA" body="DeepSeek-V2/V3 — latent compression for >100k context." />
        <UsageItem head="Sliding-window attn" body="Mistral (combines with GQA), Gemma — orthogonal trick." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-cyan-300">
        <UsageItem head="KV memory dominates" body="at L=128k, the KV cache can exceed model weights." />
        <UsageItem head="Bandwidth-bound decode" body="loading K/V from HBM is the bottleneck — shrink them, decode faster." />
        <UsageItem head="Higher batch sizes" body="smaller per-request cache → more concurrent users on one GPU." />
        <UsageItem head="Same compute, smaller params" body="GQA saves ~10–15% of attention parameters at iso-quality." />
        <UsageItem head="Latent attention (MLA)" body="cache is tiny but a small per-step decompress restores per-head expressiveness." />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem head="MQA quality drop" body="single-head K/V loses some retrieval ability; GQA mitigates." />
        <UsageItem head="Conversion isn't free" body="up-cycling MHA → GQA requires fine-tuning to recover quality." />
        <UsageItem head="MLA implementation complexity" body="needs custom kernels; rotary embedding interacts non-trivially." />
        <UsageItem head="Sequence length still matters" body="even GQA grows linearly in L; you still need paged attention." />
        <UsageItem head="Training memory unchanged" body="all variants help inference more than training (activations dominate)." />
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
      <span className="w-14 text-right tabular-nums">
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
