/** σ(x) = 1 / (1 + e^(-x)) — numerically stable for large |x|. */
export function sigmoid(x: number): number {
  if (x >= 0) {
    const e = Math.exp(-x);
    return 1 / (1 + e);
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

/** σ'(x) = σ(x) · (1 − σ(x)) */
export function sigmoidDeriv(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
}

/** Inverse: logit(p) = ln(p / (1 − p)) */
export function logit(p: number): number {
  const c = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
  return Math.log(c / (1 - c));
}

/** tanh(x) = 2σ(2x) − 1 */
export function tanh(x: number): number {
  return 2 * sigmoid(2 * x) - 1;
}

/** hard-sigmoid: piecewise linear clamp ≈ σ. */
export function hardSigmoid(x: number): number {
  return Math.max(0, Math.min(1, 0.2 * x + 0.5));
}

/**
 * Stable binary cross-entropy from a logit `z` and target `y ∈ {0, 1}`.
 *   L = log(1 + e^z) − y · z          (== softplus(z) − y·z)
 *   ∂L/∂z = σ(z) − y
 */
export function bce(z: number, y: 0 | 1): { loss: number; grad: number } {
  // softplus(z) computed stably
  const softplus = z >= 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
  return { loss: softplus - y * z, grad: sigmoid(z) - y };
}

export interface CurvePoint {
  x: number;
  y: number;
}

/** Sample N points of fn over [xMin, xMax]. */
export function sample(
  fn: (x: number) => number,
  xMin: number,
  xMax: number,
  n = 200
): CurvePoint[] {
  const out: CurvePoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = xMin + ((xMax - xMin) * i) / (n - 1);
    out.push({ x, y: fn(x) });
  }
  return out;
}
