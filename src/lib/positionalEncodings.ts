/* =============================== positional encodings =============================== */

/** Sinusoidal PE (Vaswani 2017): even dims use sin, odd use cos, freq = 1/base^{2i/d}. */
export function sinusoidal(pos: number, dim: number, base = 10000): number[] {
  const out = new Array(dim);
  for (let i = 0; i < dim; i++) {
    const half = Math.floor(i / 2);
    const omega = 1 / Math.pow(base, (2 * half) / dim);
    out[i] = i % 2 === 0 ? Math.sin(pos * omega) : Math.cos(pos * omega);
  }
  return out;
}

/** Deterministic "learned-looking" PE via a seeded LCG, normalized to [-1, 1]. */
export function learnedRandom(pos: number, dim: number, seed = 7): number[] {
  let s = (seed * 1000003 + pos * 2654435761) >>> 0;
  const out = new Array(dim);
  for (let i = 0; i < dim; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0x100000000) * 2 - 1;
  }
  return out;
}

/* =============================== RoPE =============================== */

/**
 * Rotary Position Embedding (Su et al. 2021). Rotates each (dim/2) pair of
 * features by angle pos·θ_i, with θ_i = base^{-2i/dim}. Returns the *rotated*
 * vector (so a heatmap shows the resulting embedding, not the rotation matrix).
 */
export function ropeApply(
  x: number[],
  pos: number,
  base = 10000
): number[] {
  const d = x.length;
  const out = new Array(d);
  for (let i = 0; i < d; i += 2) {
    const half = i / 2;
    const theta = Math.pow(base, (-2 * half) / d) * pos;
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const x0 = x[i];
    const x1 = i + 1 < d ? x[i + 1] : 0;
    out[i] = x0 * cos - x1 * sin;
    if (i + 1 < d) out[i + 1] = x0 * sin + x1 * cos;
  }
  return out;
}

/** Rotation angle for the j-th pair at position pos. */
export function ropeAngle(pairIdx: number, pos: number, dim: number, base = 10000): number {
  return Math.pow(base, (-2 * pairIdx) / dim) * pos;
}

/**
 * The "queries dotted with keys after RoPE" pattern, restricted to one feature
 * pair so we can show 2D rotation. Returns the 2D rotated vector.
 */
export function ropeRotate2D(
  vec2: [number, number],
  pos: number,
  pairIdx: number,
  dim: number,
  base = 10000
): [number, number] {
  const theta = ropeAngle(pairIdx, pos, dim, base);
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [vec2[0] * c - vec2[1] * s, vec2[0] * s + vec2[1] * c];
}

/* =============================== ALiBi =============================== */

/**
 * ALiBi bias added to the attention logit at (query=i, key=j) for head `h`.
 * Penalty grows linearly with distance |i - j| (causal so j ≤ i):
 *   bias = -m_h * (i - j)
 * where m_h = 2^{-8 h / nHeads}  (the original paper's geometric schedule).
 */
export function alibiSlope(headIdx: number, nHeads: number): number {
  // m_h schedule from the ALiBi paper
  return Math.pow(2, (-8 * (headIdx + 1)) / nHeads);
}

export function alibiBias(i: number, j: number, headIdx: number, nHeads: number): number {
  if (j > i) return -Infinity;
  return -alibiSlope(headIdx, nHeads) * Math.abs(i - j);
}

/* =============================== heatmap helpers =============================== */

export type Method = "sinusoidal" | "learned" | "rope" | "alibi";

/**
 * Build a (seqLen × dim) embedding matrix for the chosen method. For ALiBi,
 * we instead build the head-0 bias matrix (seqLen × seqLen) and return it
 * shaped to (seqLen × seqLen); callers should branch on `method`.
 */
export function buildEmbeddingMatrix(
  method: Exclude<Method, "alibi">,
  seqLen: number,
  dim: number,
  base = 10000
): number[][] {
  const out: number[][] = new Array(seqLen);
  for (let p = 0; p < seqLen; p++) {
    if (method === "sinusoidal") out[p] = sinusoidal(p, dim, base);
    else if (method === "learned") out[p] = learnedRandom(p, dim);
    else {
      // RoPE applied to a fixed unit vector [1, 0, 1, 0, ...] so the heatmap is
      // about *the rotation*, not a random base vector.
      const x = new Array(dim);
      for (let i = 0; i < dim; i++) x[i] = i % 2 === 0 ? 1 : 0;
      out[p] = ropeApply(x, p, base);
    }
  }
  return out;
}

/** ALiBi bias matrix at head `headIdx` over a `seqLen × seqLen` grid. */
export function buildAlibiMatrix(seqLen: number, headIdx: number, nHeads: number): number[][] {
  const out: number[][] = new Array(seqLen);
  for (let i = 0; i < seqLen; i++) {
    out[i] = new Array(seqLen);
    for (let j = 0; j < seqLen; j++) out[i][j] = alibiBias(i, j, headIdx, nHeads);
  }
  return out;
}

/* =============================== length extrapolation =============================== */

/**
 * Toy metric: how "similar" is the embedding at position `pos` to the
 * embedding at position `pos % trainLen`? Used to argue about length
 * extrapolation. Cosine similarity over the embedding vector.
 */
export function selfSimilarity(method: Exclude<Method, "alibi">, pos: number, trainLen: number, dim: number, base = 10000): number {
  const a = (method === "sinusoidal" && sinusoidal(pos, dim, base)) ||
    (method === "learned" && learnedRandom(pos, dim)) ||
    (() => {
      const x = new Array(dim);
      for (let i = 0; i < dim; i++) x[i] = i % 2 === 0 ? 1 : 0;
      return ropeApply(x, pos, base);
    })();
  const b = (method === "sinusoidal" && sinusoidal(pos % trainLen, dim, base)) ||
    (method === "learned" && learnedRandom(pos % trainLen, dim)) ||
    (() => {
      const x = new Array(dim);
      for (let i = 0; i < dim; i++) x[i] = i % 2 === 0 ? 1 : 0;
      return ropeApply(x, pos % trainLen, base);
    })();
  return cosineSimilarity(a as number[], b as number[]);
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 1e-12 ? dot / denom : 0;
}
