/* =============================== regression losses =============================== */
/* All take a residual r = ŷ − y and return loss / derivative wrt ŷ. */

export function squaredLoss(r: number): number {
  return r * r;
}
export function squaredLossDeriv(r: number): number {
  return 2 * r;
}

export function absLoss(r: number): number {
  return Math.abs(r);
}
export function absLossDeriv(r: number): number {
  if (r > 0) return 1;
  if (r < 0) return -1;
  return 0;
}

/** Huber loss with breakpoint δ. Quadratic inside |r| ≤ δ, linear outside. */
export function huberLoss(r: number, delta = 1): number {
  const a = Math.abs(r);
  return a <= delta ? 0.5 * r * r : delta * (a - 0.5 * delta);
}
export function huberLossDeriv(r: number, delta = 1): number {
  if (r > delta) return delta;
  if (r < -delta) return -delta;
  return r;
}

/** Smooth-L1 (Fast-RCNN flavor) — Huber with δ = 1, divided by δ. */
export function smoothL1(r: number): number {
  return huberLoss(r, 1);
}
export function smoothL1Deriv(r: number): number {
  return huberLossDeriv(r, 1);
}

/** Log-cosh — twice differentiable everywhere, ≈ ½r² for small r, ≈ |r| − log 2 for large. */
export function logCosh(r: number): number {
  const a = Math.abs(r);
  return a + Math.log1p(Math.exp(-2 * a)) - Math.log(2);
}
export function logCoshDeriv(r: number): number {
  return Math.tanh(r);
}

/* =============================== classification losses =============================== */

/**
 * Margin-based "hinge" loss as a function of the margin m = y·z (binary y ∈ {±1}).
 *   L(m) = max(0, 1 − m)
 */
export function hingeLoss(m: number): number {
  return Math.max(0, 1 - m);
}
export function hingeLossDeriv(m: number): number {
  return m < 1 ? -1 : 0;
}

/** Squared-hinge: max(0, 1 − m)². */
export function squaredHinge(m: number): number {
  const v = Math.max(0, 1 - m);
  return v * v;
}
export function squaredHingeDeriv(m: number): number {
  return m < 1 ? -2 * (1 - m) : 0;
}

/** Logistic / softplus loss on the margin: log(1 + exp(−m)). */
export function logisticLoss(m: number): number {
  return m >= 0 ? Math.log1p(Math.exp(-m)) : -m + Math.log1p(Math.exp(m));
}
export function logisticLossDeriv(m: number): number {
  // d/dm log(1 + e^{-m}) = -σ(-m)
  if (m >= 0) {
    const e = Math.exp(-m);
    return -e / (1 + e);
  }
  const e = Math.exp(m);
  return -1 / (1 + e);
}

/** Exponential loss (AdaBoost): exp(−m). */
export function exponentialLoss(m: number): number {
  return Math.exp(-m);
}
export function exponentialLossDeriv(m: number): number {
  return -Math.exp(-m);
}

/**
 * Focal loss as a function of the predicted probability p of the true class.
 *   L(p) = −α (1 − p)^γ log(p)
 * γ = 0 → standard cross-entropy times α.
 */
export function focalLoss(p: number, gamma = 2, alpha = 1): number {
  const pc = clamp01(p);
  return -alpha * Math.pow(1 - pc, gamma) * Math.log(Math.max(pc, 1e-9));
}
export function focalLossDeriv(p: number, gamma = 2, alpha = 1): number {
  const pc = clamp01(p);
  const oneMp = 1 - pc;
  // d/dp [ (1-p)^γ · (−log p) ] = γ(1-p)^{γ−1}·log p − (1-p)^γ / p
  const a = -gamma * Math.pow(oneMp, gamma - 1) * Math.log(pc);
  const b = -Math.pow(oneMp, gamma) / pc;
  return alpha * (a + b);
}

function clamp01(p: number): number {
  return Math.min(Math.max(p, 1e-6), 1 - 1e-6);
}

/* =============================== registries =============================== */

export interface LossVariant {
  id: string;
  name: string;
  color: string;
  /** Independent variable for the plot (residual r, margin m, or probability p). */
  fn: (x: number) => number;
  deriv: (x: number) => number;
  formula: string;
  derivFormula: string;
}

export type RegressionVariantId =
  | "mse"
  | "mae"
  | "huber"
  | "smoothl1"
  | "logcosh";

export type ClassificationVariantId =
  | "hinge"
  | "sqhinge"
  | "logistic"
  | "exponential"
  | "focal";

export const REGRESSION_VARIANTS: LossVariant[] = [
  {
    id: "mse",
    name: "MSE (L2)",
    color: "#7c5cff",
    fn: squaredLoss,
    deriv: squaredLossDeriv,
    formula: String.raw`r^{2}`,
    derivFormula: String.raw`2r`,
  },
  {
    id: "mae",
    name: "MAE (L1)",
    color: "#22d3ee",
    fn: absLoss,
    deriv: absLossDeriv,
    formula: String.raw`|r|`,
    derivFormula: String.raw`\operatorname{sign}(r)`,
  },
  {
    id: "huber",
    name: "Huber (δ=1)",
    color: "#34d399",
    fn: (r) => huberLoss(r, 1),
    deriv: (r) => huberLossDeriv(r, 1),
    formula: String.raw`\begin{cases} \tfrac12 r^{2} & |r|\le\delta\\ \delta(|r|-\tfrac{\delta}{2}) & |r|>\delta\end{cases}`,
    derivFormula: String.raw`\operatorname{clip}(r,-\delta,\delta)`,
  },
  {
    id: "smoothl1",
    name: "Smooth-L1",
    color: "#f472b6",
    fn: smoothL1,
    deriv: smoothL1Deriv,
    formula: String.raw`\text{Huber}_{\delta=1}(r)`,
    derivFormula: String.raw`\operatorname{clip}(r,-1,1)`,
  },
  {
    id: "logcosh",
    name: "Log-cosh",
    color: "#fbbf24",
    fn: logCosh,
    deriv: logCoshDeriv,
    formula: String.raw`\log\cosh(r)`,
    derivFormula: String.raw`\tanh(r)`,
  },
];

export const CLASSIFICATION_VARIANTS: LossVariant[] = [
  {
    id: "hinge",
    name: "Hinge",
    color: "#7c5cff",
    fn: hingeLoss,
    deriv: hingeLossDeriv,
    formula: String.raw`\max(0,\,1-m)`,
    derivFormula: String.raw`-\mathbb{1}[m<1]`,
  },
  {
    id: "sqhinge",
    name: "Squared hinge",
    color: "#22d3ee",
    fn: squaredHinge,
    deriv: squaredHingeDeriv,
    formula: String.raw`\max(0,\,1-m)^{2}`,
    derivFormula: String.raw`-2(1-m)\,\mathbb{1}[m<1]`,
  },
  {
    id: "logistic",
    name: "Logistic",
    color: "#34d399",
    fn: logisticLoss,
    deriv: logisticLossDeriv,
    formula: String.raw`\log(1 + e^{-m})`,
    derivFormula: String.raw`-\sigma(-m)`,
  },
  {
    id: "exponential",
    name: "Exponential",
    color: "#fbbf24",
    fn: exponentialLoss,
    deriv: exponentialLossDeriv,
    formula: String.raw`e^{-m}`,
    derivFormula: String.raw`-e^{-m}`,
  },
];

/** Focal variants for different γ values. */
export const FOCAL_VARIANTS = [
  { id: "ce", name: "CE (γ=0)", gamma: 0, color: "#7c5cff" },
  { id: "focal-1", name: "γ = 1", gamma: 1, color: "#22d3ee" },
  { id: "focal-2", name: "γ = 2", gamma: 2, color: "#34d399" },
  { id: "focal-5", name: "γ = 5", gamma: 5, color: "#fbbf24" },
];

/* =============================== sampling =============================== */

export interface CurvePoint {
  x: number;
  y: number;
}

export function sample(
  fn: (x: number) => number,
  xMin: number,
  xMax: number,
  n = 220
): CurvePoint[] {
  const out: CurvePoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = xMin + ((xMax - xMin) * i) / (n - 1);
    out.push({ x, y: fn(x) });
  }
  return out;
}

/* =============================== outlier sweep =============================== */

/**
 * For each loss in the list, compute total loss over a noisy dataset with one outlier
 * of varying magnitude. Returns total loss vs outlier residual.
 */
export function outlierSweep(
  losses: LossVariant[],
  base: number[],
  outlierRange: number[]
): { variant: LossVariant; curve: CurvePoint[] }[] {
  const baseSum = (fn: (r: number) => number) =>
    base.reduce((s, r) => s + fn(r), 0);
  return losses.map((v) => ({
    variant: v,
    curve: outlierRange.map((r) => ({
      x: r,
      y: baseSum(v.fn) + v.fn(r),
    })),
  }));
}
