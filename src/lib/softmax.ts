export interface SoftmaxStep {
  logit: number;
  shifted: number;
  exp: number;
  prob: number;
}

export interface SoftmaxResult {
  steps: SoftmaxStep[];
  max: number;
  sumExp: number;
  entropy: number;
  argmax: number;
}

/**
 * Numerically stable softmax with temperature scaling.
 * temperature > 0. Lower T => sharper, T -> 0 approximates argmax.
 */
export function softmax(logits: number[], temperature = 1): SoftmaxResult {
  const T = Math.max(temperature, 1e-6);
  const scaled = logits.map((z) => z / T);
  const max = scaled.length ? Math.max(...scaled) : 0;
  const shifted = scaled.map((z) => z - max);
  const exps = shifted.map((z) => Math.exp(z));
  const sumExp = exps.reduce((a, b) => a + b, 0) || 1;
  const probs = exps.map((e) => e / sumExp);

  let entropy = 0;
  for (const p of probs) {
    if (p > 0) entropy -= p * Math.log(p);
  }

  let argmax = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[argmax]) argmax = i;
  }

  const steps: SoftmaxStep[] = logits.map((logit, i) => ({
    logit,
    shifted: shifted[i],
    exp: exps[i],
    prob: probs[i],
  }));

  return { steps, max, sumExp, entropy, argmax };
}

/** Maximum possible entropy for n classes (uniform distribution). */
export function maxEntropy(n: number): number {
  return n > 1 ? Math.log(n) : 0;
}

/** Just the probabilities — convenience wrapper. */
export function softmaxProbs(logits: number[], temperature = 1): number[] {
  return softmax(logits, temperature).steps.map((s) => s.prob);
}

/** Shannon entropy in nats of a probability vector. */
export function entropyOf(probs: number[]): number {
  let h = 0;
  for (const p of probs) if (p > 0) h -= p * Math.log(p);
  return h;
}

/**
 * Jacobian of softmax wrt logits (assumes T=1; divide by T for general T).
 * J[i][j] = p_i * (δ_ij - p_j)
 */
export function jacobian(probs: number[]): number[][] {
  const n = probs.length;
  const J: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = new Array(n);
    for (let j = 0; j < n; j++) {
      row[j] = probs[i] * ((i === j ? 1 : 0) - probs[j]);
    }
    J.push(row);
  }
  return J;
}

export interface SweepRow {
  T: number;
  probs: number[];
  entropy: number;
  maxProb: number;
}

/** Compute softmax across a temperature range. */
export function temperatureSweep(
  logits: number[],
  tMin = 0.05,
  tMax = 5,
  n = 80
): SweepRow[] {
  if (n < 2) n = 2;
  const out: SweepRow[] = [];
  const lo = Math.max(tMin, 1e-3);
  const step = (tMax - lo) / (n - 1);
  for (let i = 0; i < n; i++) {
    const T = lo + i * step;
    const p = softmaxProbs(logits, T);
    out.push({ T, probs: p, entropy: entropyOf(p), maxProb: Math.max(...p) });
  }
  return out;
}

/** Cross-entropy loss for a one-hot target. */
export function crossEntropyLoss(probs: number[], target: number): number {
  const p = Math.max(probs[target] ?? 1e-12, 1e-12);
  return -Math.log(p);
}

/** Gradient of cross-entropy wrt logits: p - y. */
export function crossEntropyGradient(probs: number[], target: number): number[] {
  return probs.map((p, i) => p - (i === target ? 1 : 0));
}

/**
 * Sample from a categorical distribution. Optional seed for determinism
 * using a tiny mulberry32 PRNG so demos are reproducible.
 */
export function sampleCategorical(
  probs: number[],
  n: number,
  seed?: number
): { counts: number[]; empirical: number[] } {
  const rng = seed === undefined ? Math.random : mulberry32(seed);
  const cdf: number[] = [];
  let s = 0;
  for (const p of probs) {
    s += p;
    cdf.push(s);
  }
  // Guard against floating-point sum slightly < 1
  cdf[cdf.length - 1] = 1;

  const counts = new Array(probs.length).fill(0);
  for (let i = 0; i < n; i++) {
    const r = rng();
    // Linear search is fine for ≤ a few dozen classes
    let idx = 0;
    while (idx < cdf.length - 1 && r > cdf[idx]) idx++;
    counts[idx]++;
  }
  return { counts, empirical: counts.map((c) => c / Math.max(n, 1)) };
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
