// Deterministic mocks for the Qwen visualizer — we synthesize believable
// activations, RoPE rotations, GQA grouping, and SwiGLU values so the
// visualization is illustrative without needing to actually run a model.

export const SAMPLE_PROMPT = "Quantum computers will";

/** Qwen 2.5 family architectural presets. */
export const QWEN_PRESETS = {
  "Qwen2.5-0.5B": {
    layers: 24,
    d_model: 896,
    q_heads: 14,
    kv_heads: 2,
    ffn: 4864,
    vocab: 151_936,
    tied_embeddings: true,
  },
  "Qwen2.5-1.5B": {
    layers: 28,
    d_model: 1536,
    q_heads: 12,
    kv_heads: 2,
    ffn: 8960,
    vocab: 151_936,
    tied_embeddings: true,
  },
  "Qwen2.5-7B": {
    layers: 28,
    d_model: 3584,
    q_heads: 28,
    kv_heads: 4,
    ffn: 18_944,
    vocab: 152_064,
    tied_embeddings: false,
  },
  "Qwen2.5-72B": {
    layers: 80,
    d_model: 8192,
    q_heads: 64,
    kv_heads: 8,
    ffn: 29_568,
    vocab: 152_064,
    tied_embeddings: false,
  },
} as const;

export type QwenPresetName = keyof typeof QWEN_PRESETS;

/* ---------------------------- tokenizer & embed --------------------------- */

export function qwenTokenize(text: string): { token: string; id: number }[] {
  if (!text.trim()) return [];
  const pieces = text.split(/(\s+)/).filter((s) => s.length > 0);
  const out: { token: string; id: number }[] = [];
  for (const w of pieces) {
    if (/^\s+$/.test(w)) continue;
    // Toy "byte-level BPE": for words longer than 6 chars, split into 2 pieces
    if (w.length > 6) {
      const cut = Math.ceil(w.length / 2);
      out.push({ token: w.slice(0, cut), id: hashId(w.slice(0, cut)) });
      out.push({ token: "▁" + w.slice(cut), id: hashId(w.slice(cut)) });
    } else {
      out.push({ token: w, id: hashId(w) });
    }
  }
  return out.slice(0, 16);
}

function hashId(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) % 152064;
}

export function seeded(seed: number, dim: number): number {
  const x = Math.sin(seed * 12.9898 + dim * 78.233) * 43758.5453;
  return 2 * (x - Math.floor(x)) - 1;
}

export function tokenEmbed(id: number, d = 24): number[] {
  return Array.from({ length: d }, (_, i) => seeded(id + 1, i + 1));
}

/* -------------------------------- RoPE ----------------------------------- */

/**
 * Compute RoPE base frequencies for `d` dimensions.
 * Returns d/2 angles θ_i = base^(-2i/d).
 */
export function ropeBaseAngles(d: number, base = 10_000): number[] {
  const pairs = Math.floor(d / 2);
  return Array.from({ length: pairs }, (_, i) => Math.pow(base, -(2 * i) / d));
}

/**
 * Apply RoPE to a single d-dim vector at position `pos`.
 * Pair (2i, 2i+1) gets rotated by angle pos · θ_i.
 */
export function applyRoPE(vec: number[], pos: number, base = 10_000): number[] {
  const out = vec.slice();
  const d = vec.length;
  for (let i = 0; i < Math.floor(d / 2); i++) {
    const theta = Math.pow(base, -(2 * i) / d);
    const angle = pos * theta;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const a = vec[2 * i];
    const b = vec[2 * i + 1];
    out[2 * i] = a * c - b * s;
    out[2 * i + 1] = a * s + b * c;
  }
  return out;
}

/* -------------------------------- RMSNorm -------------------------------- */

export function rmsNorm(vec: number[], gamma?: number[], eps = 1e-6): number[] {
  let sumSq = 0;
  for (const x of vec) sumSq += x * x;
  const rms = Math.sqrt(sumSq / Math.max(vec.length, 1) + eps);
  return vec.map((x, i) => (x / rms) * (gamma ? gamma[i] : 1));
}

export function layerNorm(vec: number[], eps = 1e-5): number[] {
  let m = 0;
  for (const x of vec) m += x;
  m /= Math.max(vec.length, 1);
  let v = 0;
  for (const x of vec) v += (x - m) ** 2;
  v /= Math.max(vec.length, 1);
  const std = Math.sqrt(v + eps);
  return vec.map((x) => (x - m) / std);
}

/* -------------------------------- SwiGLU --------------------------------- */

export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

export function swiglu(x: number[], dHidden: number, seed = 7) {
  // Fake `gate` and `up` projections — random linear maps so the demo is
  // illustrative rather than meaningful.
  const dIn = x.length;
  const gate = new Array(dHidden).fill(0);
  const up = new Array(dHidden).fill(0);
  for (let i = 0; i < dHidden; i++) {
    let g = 0;
    let u = 0;
    for (let j = 0; j < dIn; j++) {
      g += x[j] * seeded(seed * 100 + i, j + 1);
      u += x[j] * seeded(seed * 200 + i, j + 1);
    }
    gate[i] = g / Math.sqrt(dIn);
    up[i] = u / Math.sqrt(dIn);
  }
  const gated = gate.map((g) => silu(g));
  const product = gated.map((g, i) => g * up[i]);
  return { gate, up, gated, product };
}

/* -------------------------------- GQA ----------------------------------- */

export interface GqaLayout {
  qHeads: number;
  kvHeads: number;
  /** For each Q head, which KV head it consumes. */
  mapping: number[];
}

export function gqaLayout(qHeads: number, kvHeads: number): GqaLayout {
  const groupSize = Math.max(1, Math.floor(qHeads / kvHeads));
  const mapping = Array.from({ length: qHeads }, (_, i) =>
    Math.min(Math.floor(i / groupSize), kvHeads - 1)
  );
  return { qHeads, kvHeads, mapping };
}

/* ---------------------- parameter counting (approx) ---------------------- */

export interface QwenConfig {
  layers: number;
  d_model: number;
  q_heads: number;
  kv_heads: number;
  ffn: number;
  vocab: number;
  tied_embeddings: boolean;
}

/** Returns parameter breakdown in millions (M). */
export function qwenParamBreakdown(c: QwenConfig) {
  const d = c.d_model;
  const headDim = Math.floor(d / c.q_heads);
  // Attention: Q (d*d), K (d*kv_heads*headDim), V (d*kv_heads*headDim), out (d*d)
  const attnPerLayer = d * d + 2 * d * c.kv_heads * headDim + d * d;
  // SwiGLU FFN: gate, up, down — each d × ffn
  const ffnPerLayer = 3 * d * c.ffn;
  // RMSNorm: 2 per block + 1 final → all just d weights each (no bias)
  const normPerLayer = 2 * d;

  const blockTotal = (attnPerLayer + ffnPerLayer + normPerLayer) * c.layers;
  const embeddings = d * c.vocab;
  const finalNorm = d;
  const lmHead = c.tied_embeddings ? 0 : d * c.vocab;
  const total = blockTotal + embeddings + finalNorm + lmHead;

  const toM = (n: number) => n / 1e6;

  return {
    attention: toM((attnPerLayer + normPerLayer / 2) * c.layers),
    ffn: toM((ffnPerLayer + normPerLayer / 2) * c.layers),
    embedding: toM(embeddings + lmHead),
    other: toM(finalNorm),
    total: toM(total),
    perBlock: toM(attnPerLayer + ffnPerLayer + normPerLayer),
  };
}
