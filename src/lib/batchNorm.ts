/** Statistics for a single feature / row. */
export interface Stats {
  mean: number;
  variance: number;
  std: number;
}

const EPS_DEFAULT = 1e-5;

export function statsOf(values: number[]): Stats {
  const n = values.length;
  if (n === 0) return { mean: 0, variance: 0, std: 0 };
  let m = 0;
  for (const v of values) m += v;
  m /= n;
  let s = 0;
  for (const v of values) s += (v - m) * (v - m);
  s /= n;
  return { mean: m, variance: s, std: Math.sqrt(s) };
}

/* =============================== normalizations =============================== */

/**
 * BatchNorm. Inputs: matrix `x` of shape [B, F]. Normalizes across the batch dimension
 * (column-wise). Returns: x̂, y = γ x̂ + β, and per-feature stats.
 */
export function batchNorm(
  x: number[][],
  gamma: number[],
  beta: number[],
  eps = EPS_DEFAULT
): { xhat: number[][]; y: number[][]; stats: Stats[] } {
  const B = x.length;
  const F = x[0]?.length ?? 0;
  const stats: Stats[] = [];
  for (let j = 0; j < F; j++) {
    const col: number[] = [];
    for (let i = 0; i < B; i++) col.push(x[i][j]);
    stats.push(statsOf(col));
  }
  const xhat: number[][] = [];
  const y: number[][] = [];
  for (let i = 0; i < B; i++) {
    const xr: number[] = [];
    const yr: number[] = [];
    for (let j = 0; j < F; j++) {
      const s = stats[j];
      const h = (x[i][j] - s.mean) / Math.sqrt(s.variance + eps);
      xr.push(h);
      yr.push(gamma[j] * h + beta[j]);
    }
    xhat.push(xr);
    y.push(yr);
  }
  return { xhat, y, stats };
}

/** LayerNorm: normalize across the feature dimension for each sample (row-wise). */
export function layerNorm(
  x: number[][],
  gamma: number[],
  beta: number[],
  eps = EPS_DEFAULT
): { xhat: number[][]; y: number[][]; stats: Stats[] } {
  const B = x.length;
  const F = x[0]?.length ?? 0;
  const stats: Stats[] = [];
  const xhat: number[][] = [];
  const y: number[][] = [];
  for (let i = 0; i < B; i++) {
    const s = statsOf(x[i]);
    stats.push(s);
    const xr: number[] = [];
    const yr: number[] = [];
    for (let j = 0; j < F; j++) {
      const h = (x[i][j] - s.mean) / Math.sqrt(s.variance + eps);
      xr.push(h);
      yr.push(gamma[j] * h + beta[j]);
    }
    xhat.push(xr);
    y.push(yr);
  }
  return { xhat, y, stats };
}

/** RMSNorm: divide by row RMS, no centering. */
export function rmsNorm(
  x: number[][],
  gamma: number[],
  eps = EPS_DEFAULT
): { xhat: number[][]; y: number[][]; rms: number[] } {
  const B = x.length;
  const F = x[0]?.length ?? 0;
  const rms: number[] = [];
  const xhat: number[][] = [];
  const y: number[][] = [];
  for (let i = 0; i < B; i++) {
    let s = 0;
    for (let j = 0; j < F; j++) s += x[i][j] * x[i][j];
    const r = Math.sqrt(s / F + eps);
    rms.push(r);
    const xr: number[] = [];
    const yr: number[] = [];
    for (let j = 0; j < F; j++) {
      const h = x[i][j] / r;
      xr.push(h);
      yr.push(gamma[j] * h);
    }
    xhat.push(xr);
    y.push(yr);
  }
  return { xhat, y, rms };
}

/** GroupNorm: split features into G groups, normalize within each group per sample. */
export function groupNorm(
  x: number[][],
  groups: number,
  gamma: number[],
  beta: number[],
  eps = EPS_DEFAULT
): { xhat: number[][]; y: number[][]; statsPerSampleGroup: Stats[][] } {
  const B = x.length;
  const F = x[0]?.length ?? 0;
  const groupSize = Math.max(1, Math.floor(F / groups));
  const xhat: number[][] = [];
  const y: number[][] = [];
  const statsPerSampleGroup: Stats[][] = [];
  for (let i = 0; i < B; i++) {
    const groupStats: Stats[] = [];
    for (let g = 0; g < groups; g++) {
      const start = g * groupSize;
      const end = g === groups - 1 ? F : start + groupSize;
      const slice = x[i].slice(start, end);
      groupStats.push(statsOf(slice));
    }
    statsPerSampleGroup.push(groupStats);
    const xr: number[] = [];
    const yr: number[] = [];
    for (let j = 0; j < F; j++) {
      const g = Math.min(groups - 1, Math.floor(j / groupSize));
      const s = groupStats[g];
      const h = (x[i][j] - s.mean) / Math.sqrt(s.variance + eps);
      xr.push(h);
      yr.push(gamma[j] * h + beta[j]);
    }
    xhat.push(xr);
    y.push(yr);
  }
  return { xhat, y, statsPerSampleGroup };
}

/* =============================== running stats (EMA) =============================== */

/** Exponential moving average update used at train-time to remember batch stats. */
export function emaUpdate(running: number, batch: number, momentum: number): number {
  return (1 - momentum) * running + momentum * batch;
}

/** Simulate `nSteps` of training: per-feature running mean / var via EMA. */
export function simulateRunningStats(
  batches: number[][][],
  momentum = 0.1
): { runningMean: number[][]; runningVar: number[][] } {
  const F = batches[0][0].length;
  let mean = new Array(F).fill(0);
  let varc = new Array(F).fill(1);
  const runningMean: number[][] = [];
  const runningVar: number[][] = [];
  for (const b of batches) {
    const colStats: Stats[] = [];
    for (let j = 0; j < F; j++) {
      const col = b.map((r) => r[j]);
      colStats.push(statsOf(col));
    }
    mean = mean.map((m, j) => emaUpdate(m, colStats[j].mean, momentum));
    varc = varc.map((v, j) => emaUpdate(v, colStats[j].variance, momentum));
    runningMean.push([...mean]);
    runningVar.push([...varc]);
  }
  return { runningMean, runningVar };
}

/* =============================== data generation =============================== */

/** Seeded PRNG (mulberry32) for deterministic demos. */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random normal via Box–Muller using a seeded uniform RNG. */
export function gauss(rng: () => number): number {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Make a synthetic activation batch with per-feature drift & scale to dramatize the
 * effect of BN — each feature has its own mean and std.
 */
export function makeBatch(B: number, F: number, seed = 7): number[][] {
  const rng = mulberry32(seed);
  // Per-feature mean & std vary widely.
  const featureMean = Array.from({ length: F }, (_, j) =>
    +(((j - F / 2) / (F / 2)) * 2 + (rng() - 0.5)).toFixed(3)
  );
  const featureStd = Array.from({ length: F }, () => 0.5 + rng() * 2.5);
  const out: number[][] = [];
  for (let i = 0; i < B; i++) {
    const row: number[] = [];
    for (let j = 0; j < F; j++) {
      row.push(+(featureMean[j] + featureStd[j] * gauss(rng)).toFixed(3));
    }
    out.push(row);
  }
  return out;
}
