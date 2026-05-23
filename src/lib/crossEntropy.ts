import { sigmoid } from "./sigmoid";

const EPS = 1e-12;

/** Shannon entropy H(p) = −Σ pᵢ log pᵢ (in nats). */
export function entropy(p: number[]): number {
  let s = 0;
  for (const v of p) if (v > EPS) s -= v * Math.log(v);
  return s;
}

/** Cross-entropy H(p, q) = −Σ pᵢ log qᵢ (in nats). */
export function crossEntropy(p: number[], q: number[]): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > EPS) s -= p[i] * Math.log(Math.max(q[i], EPS));
  }
  return s;
}

/** KL divergence D_KL(p ‖ q) = Σ pᵢ log(pᵢ / qᵢ). */
export function klDivergence(p: number[], q: number[]): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > EPS) s += p[i] * Math.log(p[i] / Math.max(q[i], EPS));
  }
  return s;
}

/**
 * Stable binary cross-entropy from a logit `z` and target `y ∈ {0, 1}`.
 *   L = log(1 + eᶻ) − y · z          (== softplus(z) − y·z)
 *   ∂L/∂z = σ(z) − y
 */
export function bce(z: number, y: 0 | 1): { loss: number; grad: number; prob: number } {
  const softplus = z >= 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
  const prob = sigmoid(z);
  return { loss: softplus - y * z, grad: prob - y, prob };
}

/** Build a one-hot vector of length n with a 1 at `target`. */
export function oneHot(n: number, target: number): number[] {
  const v = new Array<number>(n).fill(0);
  v[target] = 1;
  return v;
}

/** Normalize a non-negative vector to a probability simplex; uniform if zero-sum. */
export function normalize(v: number[]): number[] {
  const s = v.reduce((a, b) => a + b, 0);
  if (s <= 0) return v.map(() => 1 / v.length);
  return v.map((x) => x / s);
}
