/* =============================== low-rank adapters =============================== */

export function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

/** N(0, σ) via Box-Muller. */
function gaussian(rng: () => number, sigma = 1): number {
  const u1 = Math.max(rng(), 1e-9);
  const u2 = rng();
  return sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Random Gaussian (n × m) matrix. */
export function randomGaussian(n: number, m: number, sigma: number, seed: number): number[][] {
  const rng = lcg(seed);
  return Array.from({ length: n }, () => Array.from({ length: m }, () => gaussian(rng, sigma)));
}

/** Matrix multiply: (n × r) · (r × m) -> (n × m). */
export function matmul(A: number[][], B: number[][]): number[][] {
  const n = A.length;
  const r = A[0].length;
  const m = B[0].length;
  const out = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < r; k++) {
      const aik = A[i][k];
      for (let j = 0; j < m; j++) out[i][j] += aik * B[k][j];
    }
  }
  return out;
}

/** Element-wise difference. */
export function matSub(A: number[][], B: number[][]): number[][] {
  return A.map((row, i) => row.map((v, j) => v - B[i][j]));
}

/** Frobenius norm. */
export function frobenius(M: number[][]): number {
  let s = 0;
  for (const row of M) for (const v of row) s += v * v;
  return Math.sqrt(s);
}

/* =============================== lora math =============================== */

export interface LoRAConfig {
  /** Output dim of W (rows). */
  d: number;
  /** Input dim of W (cols). */
  k: number;
  /** Adapter rank. */
  r: number;
  /** Scaling factor α (effective scaling = α / r). */
  alpha: number;
}

/** Parameter count of dense W. */
export function denseParams(cfg: { d: number; k: number }): number {
  return cfg.d * cfg.k;
}

/** Parameter count of LoRA adapter B (d×r) + A (r×k). */
export function loraParams(cfg: LoRAConfig): number {
  return (cfg.d + cfg.k) * cfg.r;
}

/** Trainable-parameter ratio (LoRA / dense). */
export function loraRatio(cfg: LoRAConfig): number {
  return loraParams(cfg) / Math.max(denseParams(cfg), 1);
}

/**
 * Approximate W as B·A and compute the (scaled) reconstruction.
 * Returns W_approx = (α/r) · B A.
 */
export function loraReconstruct(_W: number[][], A: number[][], B: number[][], alpha: number, r: number): number[][] {
  const BA = matmul(B, A);
  const scale = alpha / Math.max(r, 1);
  return BA.map((row) => row.map((v) => v * scale));
}

/* =============================== rank sweep =============================== */

/** Reconstruction error (Frobenius) sweep for ranks 1..rMax. */
export function rankSweep(W: number[][], rMax: number, seed: number, alpha = 1): { r: number; error: number; ratio: number }[] {
  const d = W.length;
  const k = W[0].length;
  const wNorm = frobenius(W);
  const out: { r: number; error: number; ratio: number }[] = [];
  for (let r = 1; r <= rMax; r++) {
    // Initialize B = 0, A ~ N(0, 1/r). That makes the *initial* output zero
    // (the LoRA convention), but for the rank-sweep we want a meaningful
    // approximation, so we use random B, A and let the scale fall out.
    const B = randomGaussian(d, r, 1, seed + r);
    const A = randomGaussian(r, k, 1 / Math.sqrt(r), seed + r * 7919);
    const approx = loraReconstruct(W, A, B, alpha, r);
    const err = frobenius(matSub(W, approx));
    out.push({ r, error: err / Math.max(wNorm, 1e-9), ratio: loraParams({ d, k, r, alpha }) / denseParams({ d, k }) });
  }
  return out;
}

/* =============================== variants comparison =============================== */

export interface LoRAVariant {
  name: string;
  storageCost: string;
  computeCost: string;
  notes: string;
}

export const VARIANTS: LoRAVariant[] = [
  { name: "LoRA", storageCost: "(d+k)·r · 16-bit", computeCost: "fp16 fwd/bwd", notes: "the original — add B·A, freeze W." },
  { name: "QLoRA", storageCost: "same as LoRA", computeCost: "4-bit W + bf16 BA", notes: "quantize W to NF4, train BA in bf16 — fits 65B on one 48GB GPU." },
  { name: "DoRA", storageCost: "+ d·1 magnitude", computeCost: "fp16 fwd/bwd", notes: "decompose W into direction + magnitude, LoRA-tune the direction." },
  { name: "IA³", storageCost: "k vectors per layer", computeCost: "fp16", notes: "scale K/V/FFN by learned diagonal — fewer params than LoRA." },
  { name: "Prefix-tuning", storageCost: "n·d·N_l", computeCost: "fp16", notes: "learn virtual key/value vectors prepended to attention." },
];
