/* =============================== continuous KL helpers =============================== */

/** N(x; μ, σ²) PDF. */
export function gaussianPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (Math.sqrt(2 * Math.PI) * sigma);
}

/** Two-component Gaussian mixture: π·N(μ₁, σ₁²) + (1−π)·N(μ₂, σ₂²). */
export interface GMM2 {
  pi: number;
  mu1: number;
  sigma1: number;
  mu2: number;
  sigma2: number;
}

export function mixturePdf(x: number, g: GMM2): number {
  return g.pi * gaussianPdf(x, g.mu1, g.sigma1) + (1 - g.pi) * gaussianPdf(x, g.mu2, g.sigma2);
}

/** Numerical KL(p ‖ q) via Riemann sum on a grid. */
export function klContinuous(
  p: (x: number) => number,
  q: (x: number) => number,
  xMin: number,
  xMax: number,
  n = 800
): number {
  const dx = (xMax - xMin) / n;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const x = xMin + (i + 0.5) * dx;
    const pv = p(x);
    const qv = q(x);
    if (pv > 1e-12 && qv > 1e-12) {
      s += pv * Math.log(pv / qv) * dx;
    } else if (pv > 1e-12) {
      // q is effectively zero where p > 0 → ∞; clamp to a large value
      s += pv * Math.log(pv / 1e-12) * dx;
    }
  }
  return Math.max(0, s);
}

/** Numerical Jensen-Shannon divergence (base e). */
export function jsDivergence(
  p: (x: number) => number,
  q: (x: number) => number,
  xMin: number,
  xMax: number,
  n = 800
): number {
  const m = (x: number) => 0.5 * (p(x) + q(x));
  return 0.5 * klContinuous(p, m, xMin, xMax, n) + 0.5 * klContinuous(q, m, xMin, xMax, n);
}

/* =============================== best-fit single Gaussians =============================== */

/** Moment-matching fit (the forward-KL minimizer for an exponential family). */
export function momentMatchingFit(
  g: GMM2
): { mu: number; sigma: number } {
  const mu = g.pi * g.mu1 + (1 - g.pi) * g.mu2;
  const var1 = g.sigma1 * g.sigma1 + g.mu1 * g.mu1;
  const var2 = g.sigma2 * g.sigma2 + g.mu2 * g.mu2;
  const second = g.pi * var1 + (1 - g.pi) * var2;
  const variance = Math.max(1e-6, second - mu * mu);
  return { mu, sigma: Math.sqrt(variance) };
}

/**
 * Mode-seeking fit (approximate reverse-KL minimizer): grid-search over μ, σ
 * that minimizes KL(q ‖ p). Cheap and robust for our visualization.
 */
export function modeSeekingFit(
  g: GMM2,
  xMin: number,
  xMax: number
): { mu: number; sigma: number } {
  const p = (x: number) => mixturePdf(x, g);
  const muCandidates: number[] = [];
  const sigmaCandidates: number[] = [];
  const nMu = 60;
  const nS = 40;
  for (let i = 0; i < nMu; i++) muCandidates.push(xMin + ((xMax - xMin) * i) / (nMu - 1));
  for (let j = 0; j < nS; j++) sigmaCandidates.push(0.2 + (2.5 * j) / (nS - 1));
  let best = { mu: 0, sigma: 1, kl: Infinity };
  for (const mu of muCandidates) {
    for (const sigma of sigmaCandidates) {
      const q = (x: number) => gaussianPdf(x, mu, sigma);
      const kl = klContinuous(q, p, xMin, xMax, 220);
      if (kl < best.kl) best = { mu, sigma, kl };
    }
  }
  return { mu: best.mu, sigma: best.sigma };
}
