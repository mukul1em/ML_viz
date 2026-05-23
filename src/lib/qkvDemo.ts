// Deterministic small-scale QKV demo. We don't actually train anything —
// we synthesize Q, K, V via seeded "weight matrices" so the math is real and
// the visualization stays stable.

export const DEFAULT_TOKENS = ["The", "cat", "sat", "on", "mat"];
export const D_MODEL = 8;
export const D_K = 4;

export interface AttentionTrace {
  Q: number[][];
  K: number[][];
  V: number[][];
  scores: number[];
  scaled: number[];
  weights: number[];
  output: number[];
}

function seeded(seed: number, dim: number): number {
  const x = Math.sin(seed * 12.9898 + dim * 78.233) * 43758.5453;
  return 2 * (x - Math.floor(x)) - 1;
}

/** Token embedding: deterministic vector of length `d` for token at index i. */
export function inputEmbed(i: number, d: number = D_MODEL): number[] {
  return Array.from({ length: d }, (_, k) => seeded(i + 1, k + 1));
}

/** Apply a mock learned linear projection. `salt` distinguishes W_Q, W_K, W_V. */
function project(x: number[], salt: number, outDim: number = D_K): number[] {
  const out = new Array(outDim).fill(0);
  for (let i = 0; i < outDim; i++) {
    let s = 0;
    for (let j = 0; j < x.length; j++) {
      s += x[j] * seeded(salt * 100 + i, j + 7);
    }
    // Scale down so values stay in a readable range
    out[i] = s / Math.sqrt(x.length);
  }
  return out;
}

export function makeQ(i: number, d: number = D_MODEL): number[] {
  return project(inputEmbed(i, d), 11);
}
export function makeK(i: number, d: number = D_MODEL): number[] {
  return project(inputEmbed(i, d), 23);
}
export function makeV(i: number, d: number = D_MODEL): number[] {
  return project(inputEmbed(i, d), 37);
}

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) s += a[i] * b[i];
  return s;
}

export function softmax(values: number[]): number[] {
  const finite = values.filter((v) => Number.isFinite(v));
  const max = finite.length ? Math.max(...finite) : 0;
  const exps = values.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

/** Trace the full attention computation for a single query position. */
export function traceAttention(
  tokens: string[],
  queryIdx: number,
  options: { scale?: boolean; causal?: boolean } = {}
): AttentionTrace {
  const scale = options.scale ?? true;
  const causal = options.causal ?? false;

  const Q = tokens.map((_, i) => makeQ(i));
  const K = tokens.map((_, i) => makeK(i));
  const V = tokens.map((_, i) => makeV(i));

  const q = Q[queryIdx];
  const scores = K.map((k) => dot(q, k));
  const scaled = scores.map((s) => (scale ? s / Math.sqrt(D_K) : s));
  const masked = scaled.map((s, j) =>
    causal && j > queryIdx ? Number.NEGATIVE_INFINITY : s
  );
  const weights = softmax(masked);

  const output = new Array(D_K).fill(0);
  for (let j = 0; j < V.length; j++) {
    for (let k = 0; k < D_K; k++) {
      output[k] += weights[j] * V[j][k];
    }
  }

  return { Q, K, V, scores, scaled, weights, output };
}

/** Full N×N attention matrix (row = query, col = key). */
export function fullAttentionMatrix(
  tokens: string[],
  options: { scale?: boolean; causal?: boolean } = {}
): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < tokens.length; i++) {
    out.push(traceAttention(tokens, i, options).weights);
  }
  return out;
}
