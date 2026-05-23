/* =============================== sampling strategies =============================== */

/** Softmax over logits with optional temperature τ ∈ (0, ∞). */
export function softmaxT(logits: number[], temperature = 1): number[] {
  const t = Math.max(temperature, 1e-6);
  const scaled = logits.map((z) => z / t);
  const m = Math.max(...scaled);
  const exps = scaled.map((z) => Math.exp(z - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

/** Shannon entropy of a probability vector (in nats). */
export function entropyOf(p: number[]): number {
  let h = 0;
  for (const pi of p) if (pi > 1e-12) h -= pi * Math.log(pi);
  return h;
}

/** Index sorted high-to-low by probability. */
function argsortDesc(p: number[]): number[] {
  return p
    .map((v, i) => [v, i] as const)
    .sort((a, b) => b[0] - a[0])
    .map(([, i]) => i);
}

/** Renormalize a sparse probability vector (zeros for filtered, others sum to 1). */
function renormalize(p: number[]): number[] {
  const s = p.reduce((a, b) => a + b, 0);
  if (s <= 0) return p.slice();
  return p.map((v) => v / s);
}

/** Keep only the top-k tokens, zero the rest, then renormalize. */
export function topKMask(p: number[], k: number): number[] {
  if (k <= 0 || k >= p.length) return p.slice();
  const order = argsortDesc(p);
  const keep = new Set(order.slice(0, k));
  return renormalize(p.map((v, i) => (keep.has(i) ? v : 0)));
}

/**
 * Nucleus / top-p: keep the smallest set whose cumulative probability ≥ p_target.
 * Returns the renormalized vector with everything outside the nucleus zeroed.
 */
export function topPMask(p: number[], pTarget: number): number[] {
  if (pTarget >= 1) return p.slice();
  if (pTarget <= 0) {
    const order = argsortDesc(p);
    return p.map((_, i) => (i === order[0] ? 1 : 0));
  }
  const order = argsortDesc(p);
  const keep = new Set<number>();
  let cum = 0;
  for (const i of order) {
    keep.add(i);
    cum += p[i];
    if (cum >= pTarget) break;
  }
  return renormalize(p.map((v, i) => (keep.has(i) ? v : 0)));
}

/** Min-p: keep tokens with prob ≥ minP · max(p), zero the rest, renormalize. */
export function minPMask(p: number[], minP: number): number[] {
  if (minP <= 0) return p.slice();
  const top = Math.max(...p);
  const threshold = minP * top;
  return renormalize(p.map((v) => (v >= threshold ? v : 0)));
}

/**
 * Repetition penalty (HuggingFace / CTRL convention).
 * Divide positive logits by `penalty`, multiply negative logits by `penalty`.
 */
export function applyRepetitionPenalty(
  logits: number[],
  seen: number[],
  penalty: number
): number[] {
  if (penalty === 1) return logits.slice();
  const seenSet = new Set(seen);
  return logits.map((z, i) => {
    if (!seenSet.has(i)) return z;
    return z > 0 ? z / penalty : z * penalty;
  });
}

/* =============================== seeded sampling =============================== */

/** 32-bit LCG: deterministic uniform stream. */
export function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

/** Draw one sample from a probability vector using a uniform u ∈ [0, 1). */
export function sampleFrom(p: number[], u: number): number {
  let cum = 0;
  for (let i = 0; i < p.length; i++) {
    cum += p[i];
    if (u < cum) return i;
  }
  return p.length - 1;
}

/** Greedy argmax. */
export function argmax(p: number[]): number {
  let best = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  return best;
}

/* =============================== beam search =============================== */

export interface Beam {
  tokens: number[];
  logProb: number;
}

/**
 * Single beam-search step: from each active beam, take all candidates and
 * keep the global top `width`. `nextLogits(beam)` returns log-probabilities
 * over the vocabulary for that beam.
 */
export function beamStep(
  beams: Beam[],
  width: number,
  nextLogits: (beam: Beam) => number[]
): Beam[] {
  const candidates: Beam[] = [];
  for (const b of beams) {
    const logp = nextLogits(b);
    for (let i = 0; i < logp.length; i++) {
      candidates.push({ tokens: [...b.tokens, i], logProb: b.logProb + logp[i] });
    }
  }
  candidates.sort((a, b) => b.logProb - a.logProb);
  return candidates.slice(0, width);
}

/* =============================== sample vocab =============================== */

/** Toy "next-word" vocabulary used by the visualization. */
export const SAMPLE_VOCAB: string[] = [
  "the",
  "a",
  "cat",
  "dog",
  "sat",
  "on",
  "in",
  "and",
  "ran",
  "quickly",
  "tree",
  "house",
  "very",
  "soft",
  "loud",
  "stone",
];

/** Default logits — descending with a moderate spread (top-1 prob ≈ 0.27 at τ=1). */
export const DEFAULT_LOGITS: number[] = [
  3.2, 2.4, 1.9, 1.6, 1.4, 1.1, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.6, -1.0, -1.6,
];
