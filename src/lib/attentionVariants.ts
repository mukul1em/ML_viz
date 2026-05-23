/* =============================== attention variants =============================== */

export type VariantId = "mha" | "mqa" | "gqa" | "mla";

export interface VariantSpec {
  id: VariantId;
  name: string;
  short: string;
  blurb: string;
  /** How many K/V heads given n_heads_q. For MLA, n_heads_kv = 1 (latent compressed). */
  nHeadsKv: (nHeadsQ: number, group?: number) => number;
  /** Optional latent-compression factor for MLA (kv dim is divided by this). */
  latentRatio?: number;
}

export const VARIANTS: Record<VariantId, VariantSpec> = {
  mha: {
    id: "mha",
    name: "Multi-Head Attention",
    short: "MHA",
    blurb: "Every Q head has its own K and V — the OG transformer.",
    nHeadsKv: (nHeadsQ) => nHeadsQ,
  },
  mqa: {
    id: "mqa",
    name: "Multi-Query Attention",
    short: "MQA",
    blurb: "All Q heads share a single K and V. Smallest cache, fastest decode.",
    nHeadsKv: () => 1,
  },
  gqa: {
    id: "gqa",
    name: "Grouped-Query Attention",
    short: "GQA",
    blurb: "Q heads split into G groups; each group shares one K/V. The LLaMA-3 sweet spot.",
    nHeadsKv: (nHeadsQ, group = 8) => Math.max(1, Math.floor(nHeadsQ / group)),
  },
  mla: {
    id: "mla",
    name: "Multi-head Latent Attention",
    short: "MLA",
    blurb: "DeepSeek-style. K/V projected to a small shared latent (~512 dims), decompressed per head.",
    nHeadsKv: () => 1,
    latentRatio: 8,
  },
};

export interface VariantConfig {
  nHeadsQ: number;
  /** For GQA: the head-group size (Q heads per K/V head). */
  groupSize?: number;
  dHead: number;
  seqLen: number;
  nLayers: number;
  batch: number;
  dtypeBytes: number; // 2 for fp16/bf16, 1 for int8
}

/** Effective K+V heads for a given variant under config. */
export function effectiveKvHeads(variant: VariantId, cfg: VariantConfig): number {
  return VARIANTS[variant].nHeadsKv(cfg.nHeadsQ, cfg.groupSize);
}

/** Effective per-head K/V dimension (MLA compresses this). */
export function effectiveKvDim(variant: VariantId, cfg: VariantConfig): number {
  if (variant === "mla") return Math.max(1, Math.round(cfg.dHead / (VARIANTS.mla.latentRatio ?? 8)));
  return cfg.dHead;
}

/**
 * Per-layer KV cache size in bytes:
 *   2 (K + V) · batch · n_heads_kv · d_kv · seq_len · dtype_bytes
 * Total across all layers = above · n_layers.
 */
export function kvCacheBytes(variant: VariantId, cfg: VariantConfig): number {
  const heads = effectiveKvHeads(variant, cfg);
  const dKv = effectiveKvDim(variant, cfg);
  return 2 * cfg.batch * heads * dKv * cfg.seqLen * cfg.nLayers * cfg.dtypeBytes;
}

/** Attention-only FLOPs per layer per token. Rough order-of-magnitude. */
export function attnFlopsPerToken(variant: VariantId, cfg: VariantConfig): number {
  // Q projection: d_model · d_model
  // K, V projection: d_model · (n_heads_kv · d_kv)
  // Attention: 2 · n_heads_q · seq_len · d_head
  const dModel = cfg.nHeadsQ * cfg.dHead;
  const headsKv = effectiveKvHeads(variant, cfg);
  const dKv = effectiveKvDim(variant, cfg);
  const projQ = 2 * dModel * dModel;
  const projKv = 2 * 2 * dModel * (headsKv * dKv);
  const attn = 2 * cfg.nHeadsQ * cfg.seqLen * cfg.dHead;
  return projQ + projKv + attn;
}

/** Format a byte count with sensible units. */
export function formatBytes(n: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u += 1;
  }
  return `${v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)} ${units[u]}`;
}

/** Real-model presets for the comparison row. */
export interface ModelPreset {
  name: string;
  variant: VariantId;
  nHeadsQ: number;
  groupSize?: number;
  dHead: number;
  nLayers: number;
}

export const MODEL_PRESETS: ModelPreset[] = [
  { name: "GPT-2", variant: "mha", nHeadsQ: 12, dHead: 64, nLayers: 12 },
  { name: "LLaMA-2 7B", variant: "mha", nHeadsQ: 32, dHead: 128, nLayers: 32 },
  { name: "LLaMA-3 8B (GQA 4:1)", variant: "gqa", nHeadsQ: 32, groupSize: 4, dHead: 128, nLayers: 32 },
  { name: "LLaMA-3 70B (GQA 8:1)", variant: "gqa", nHeadsQ: 64, groupSize: 8, dHead: 128, nLayers: 80 },
  { name: "Falcon-180B (MQA)", variant: "mqa", nHeadsQ: 232, dHead: 64, nLayers: 80 },
  { name: "DeepSeek-V3 (MLA)", variant: "mla", nHeadsQ: 128, dHead: 128, nLayers: 61 },
];
