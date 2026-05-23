import { sigmoid } from "./sigmoid";

/** tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ) — built-in is already numerically stable. */
export function tanh(x: number): number {
  return Math.tanh(x);
}

/** d/dx tanh = 1 − tanh²(x) = sech²(x) */
export function tanhDeriv(x: number): number {
  const t = Math.tanh(x);
  return 1 - t * t;
}

/** Inverse: artanh(p) = ½ ln((1 + p) / (1 − p)) */
export function artanh(p: number): number {
  const c = Math.min(Math.max(p, -1 + 1e-9), 1 - 1e-9);
  return 0.5 * Math.log((1 + c) / (1 - c));
}

/** Hard-tanh: clip(x, −1, 1) */
export function hardTanh(x: number): number {
  return Math.max(-1, Math.min(1, x));
}

/** Scaled tanh as used in classic NNs: 1.7159 · tanh(2/3 · x) (LeCun, 1998) */
export function scaledTanh(x: number): number {
  return 1.7159 * Math.tanh((2 / 3) * x);
}

/** Identity for quick mental conversion: tanh(x) = 2σ(2x) − 1 */
export function tanhFromSigmoid(x: number): number {
  return 2 * sigmoid(2 * x) - 1;
}

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
