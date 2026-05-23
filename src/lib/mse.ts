/* =============================== MSE / L2 loss =============================== */

/** Squared error of a single residual r = ŷ − y. */
export function squaredError(r: number): number {
  return r * r;
}

/** dL/dr for L = r²  →  2r. */
export function squaredErrorDeriv(r: number): number {
  return 2 * r;
}

/** Mean squared error over arrays of equal length. */
export function mse(yTrue: number[], yPred: number[]): number {
  if (yTrue.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const r = yPred[i] - yTrue[i];
    s += r * r;
  }
  return s / yTrue.length;
}

/** Root mean squared error. */
export function rmse(yTrue: number[], yPred: number[]): number {
  return Math.sqrt(mse(yTrue, yPred));
}

/** Mean absolute error. Used here only for outlier comparison vs MSE. */
export function mae(yTrue: number[], yPred: number[]): number {
  if (yTrue.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) s += Math.abs(yPred[i] - yTrue[i]);
  return s / yTrue.length;
}

/* =============================== closed-form regression =============================== */

export interface LinearFit {
  m: number;
  b: number;
}

/** OLS line of best fit (minimizes Σ (y − m·x − b)²). */
export function olsFit(xs: number[], ys: number[]): LinearFit {
  const n = xs.length;
  if (n === 0) return { m: 0, b: 0 };
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
    sxx += xs[i] * xs[i];
    sxy += xs[i] * ys[i];
  }
  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return { m: 0, b: sy / n };
  const m = (n * sxy - sx * sy) / denom;
  const b = (sy - m * sx) / n;
  return { m, b };
}

/** Median-anchored L1 line (1-step IRLS approximation). */
export function l1Fit(xs: number[], ys: number[], iters = 25): LinearFit {
  let { m, b } = olsFit(xs, ys);
  for (let it = 0; it < iters; it++) {
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    let sw = 0;
    for (let i = 0; i < xs.length; i++) {
      const r = ys[i] - (m * xs[i] + b);
      const w = 1 / Math.max(Math.abs(r), 1e-3);
      sw += w;
      sx += w * xs[i];
      sy += w * ys[i];
      sxx += w * xs[i] * xs[i];
      sxy += w * xs[i] * ys[i];
    }
    const denom = sw * sxx - sx * sx;
    if (Math.abs(denom) < 1e-12) break;
    m = (sw * sxy - sx * sy) / denom;
    b = (sy - m * sx) / sw;
  }
  return { m, b };
}

/* =============================== sample data =============================== */

/** Deterministic 32-bit LCG for reproducible scatter data. */
function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

/** Box-Muller from a seeded uniform. */
function gaussian(rand: () => number): number {
  const u1 = Math.max(rand(), 1e-9);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export interface ScatterPoint {
  x: number;
  y: number;
}

/**
 * Generate scatter data y = mx + b + ε with optional outliers.
 * Deterministic per `seed`.
 */
export function makeScatter({
  n = 30,
  m = 1.0,
  b = 0.5,
  noise = 0.5,
  outliers = 0,
  outlierMag = 6,
  xMin = -3,
  xMax = 3,
  seed = 42,
}: {
  n?: number;
  m?: number;
  b?: number;
  noise?: number;
  outliers?: number;
  outlierMag?: number;
  xMin?: number;
  xMax?: number;
  seed?: number;
}): ScatterPoint[] {
  const rand = lcg(seed);
  const out: ScatterPoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = xMin + ((xMax - xMin) * i) / (n - 1);
    const eps = noise * gaussian(rand);
    out.push({ x, y: m * x + b + eps });
  }
  // promote the last `outliers` points to vertical outliers
  const idxs: number[] = [];
  while (idxs.length < outliers) {
    const k = Math.floor(rand() * n);
    if (!idxs.includes(k)) idxs.push(k);
  }
  for (const k of idxs) {
    const sign = rand() > 0.5 ? 1 : -1;
    out[k].y += sign * outlierMag;
  }
  return out;
}
