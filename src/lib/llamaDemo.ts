// LLaMA shares its core building blocks with Qwen (RMSNorm, RoPE, GQA, SwiGLU),
// so we re-export the shared math helpers and ship LLaMA-specific presets and
// configuration metadata here.

export {
  ropeBaseAngles,
  applyRoPE,
  rmsNorm,
  layerNorm,
  silu,
  swiglu,
  gqaLayout,
  tokenEmbed,
  seeded,
  type GqaLayout,
} from "./qwenDemo";

export const SAMPLE_PROMPT = "The future of open AI is";

/**
 * LLaMA architectural presets — sizes that ship publicly with open weights.
 */
export const LLAMA_PRESETS = {
  "Llama-3.2-1B": {
    layers: 16,
    d_model: 2048,
    q_heads: 32,
    kv_heads: 8,
    ffn: 8192,
    vocab: 128_256,
    tied_embeddings: true,
    rope_theta: 500_000,
    context: 128_000,
  },
  "Llama-3.2-3B": {
    layers: 28,
    d_model: 3072,
    q_heads: 24,
    kv_heads: 8,
    ffn: 8192,
    vocab: 128_256,
    tied_embeddings: true,
    rope_theta: 500_000,
    context: 128_000,
  },
  "Llama-3.1-8B": {
    layers: 32,
    d_model: 4096,
    q_heads: 32,
    kv_heads: 8,
    ffn: 14_336,
    vocab: 128_256,
    tied_embeddings: false,
    rope_theta: 500_000,
    context: 128_000,
  },
  "Llama-3.1-70B": {
    layers: 80,
    d_model: 8192,
    q_heads: 64,
    kv_heads: 8,
    ffn: 28_672,
    vocab: 128_256,
    tied_embeddings: false,
    rope_theta: 500_000,
    context: 128_000,
  },
  "Llama-3.1-405B": {
    layers: 126,
    d_model: 16_384,
    q_heads: 128,
    kv_heads: 16,
    ffn: 53_248,
    vocab: 128_256,
    tied_embeddings: false,
    rope_theta: 500_000,
    context: 128_000,
  },
} as const;

export type LlamaPresetName = keyof typeof LLAMA_PRESETS;

/* ---------------------------- tokenizer ---------------------------- */

/**
 * LLaMA-3 uses a tiktoken-style BPE tokenizer with a 128k vocabulary (a sharp
 * break from LLaMA-1 / LLaMA-2, which used a 32k SentencePiece tokenizer). This
 * is a toy split that keeps the demo deterministic.
 */
export function llamaTokenize(text: string): { token: string; id: number }[] {
  if (!text.trim()) return [];
  const pieces = text.split(/(\s+)/).filter((s) => s.length > 0);
  const out: { token: string; id: number }[] = [];
  for (const w of pieces) {
    if (/^\s+$/.test(w)) continue;
    // 128k vocab swallows most short tokens whole.
    if (w.length > 7) {
      const cut = Math.ceil(w.length / 2);
      out.push({ token: w.slice(0, cut), id: hashId(w.slice(0, cut)) });
      out.push({ token: "Ġ" + w.slice(cut), id: hashId(w.slice(cut)) });
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
  return (h >>> 0) % 128_256;
}

/* ---------------------- parameter counting (approx) ---------------------- */

export interface LlamaConfig {
  layers: number;
  d_model: number;
  q_heads: number;
  kv_heads: number;
  ffn: number;
  vocab: number;
  tied_embeddings: boolean;
  rope_theta: number;
  context: number;
}

/** Returns parameter breakdown in millions (M). LLaMA has no bias on linears. */
export function llamaParamBreakdown(c: LlamaConfig) {
  const d = c.d_model;
  const headDim = Math.floor(d / c.q_heads);
  // Attention: Q (d*d), K (d*kv_heads*headDim), V (d*kv_heads*headDim), out (d*d). No bias.
  const attnPerLayer = d * d + 2 * d * c.kv_heads * headDim + d * d;
  // SwiGLU FFN: gate, up, down — each d × ffn. No bias.
  const ffnPerLayer = 3 * d * c.ffn;
  // RMSNorm: 2 per block + 1 final → d weights each (no bias, no centering).
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

/** Effective KV cache saving from GQA, expressed as a percentage. */
export function gqaSavings(c: LlamaConfig): number {
  return 1 - c.kv_heads / c.q_heads;
}
