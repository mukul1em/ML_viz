import { sigmoid } from "./sigmoid";

/* =============================== ReLU & friends =============================== */

/** ReLU(x) = max(0, x) */
export function relu(x: number): number {
  return x > 0 ? x : 0;
}
/** d/dx ReLU = 1 if x > 0 else 0   (sub-gradient at 0 is conventionally 0) */
export function reluDeriv(x: number): number {
  return x > 0 ? 1 : 0;
}

/** Leaky ReLU: f(x) = max(αx, x), default α = 0.01 */
export function leakyRelu(x: number, alpha = 0.01): number {
  return x >= 0 ? x : alpha * x;
}
export function leakyReluDeriv(x: number, alpha = 0.01): number {
  return x >= 0 ? 1 : alpha;
}

/** ELU: f(x) = x for x≥0, α(eˣ − 1) otherwise */
export function elu(x: number, alpha = 1): number {
  return x >= 0 ? x : alpha * (Math.exp(x) - 1);
}
export function eluDeriv(x: number, alpha = 1): number {
  return x >= 0 ? 1 : alpha * Math.exp(x);
}

/** GELU (exact): f(x) = x · Φ(x), where Φ is N(0,1) CDF */
export function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}
/** d/dx GELU = Φ(x) + x · φ(x) */
export function geluDeriv(x: number): number {
  const Phi = 0.5 * (1 + erf(x / Math.SQRT2));
  const phi = Math.exp(-(x * x) / 2) / Math.sqrt(2 * Math.PI);
  return Phi + x * phi;
}

/** SiLU / Swish: f(x) = x · σ(x) */
export function silu(x: number): number {
  return x * sigmoid(x);
}
/** d/dx SiLU = σ(x) + x · σ(x)(1 − σ(x)) */
export function siluDeriv(x: number): number {
  const s = sigmoid(x);
  return s + x * s * (1 - s);
}

/** Softplus(x) = log(1 + eˣ) — stable */
export function softplus(x: number): number {
  return x >= 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x));
}
/** d/dx softplus = σ(x) */
export function softplusDeriv(x: number): number {
  return sigmoid(x);
}

/** Mish: f(x) = x · tanh(softplus(x)) */
export function mish(x: number): number {
  return x * Math.tanh(softplus(x));
}
export function mishDeriv(x: number): number {
  const sp = softplus(x);
  const t = Math.tanh(sp);
  const sech2 = 1 - t * t;
  return t + x * sech2 * sigmoid(x);
}

/* =============================== erf approximation =============================== */

/** Abramowitz & Stegun 7.1.26 — max abs error ≈ 1.5e-7 */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-a * a);
  return sign * y;
}

/* =============================== registry =============================== */

export interface VariantSpec {
  id: VariantId;
  name: string;
  color: string;
  fn: (x: number) => number;
  deriv: (x: number) => number;
  formula: string;
  derivFormula: string;
}

export type VariantId =
  | "relu"
  | "leaky"
  | "elu"
  | "gelu"
  | "silu"
  | "mish"
  | "softplus";

export const VARIANTS: VariantSpec[] = [
  {
    id: "relu",
    name: "ReLU",
    color: "#7c5cff",
    fn: relu,
    deriv: reluDeriv,
    formula: String.raw`\mathrm{ReLU}(x) = \max(0,\,x)`,
    derivFormula: String.raw`\mathbb{1}[x > 0]`,
  },
  {
    id: "leaky",
    name: "Leaky",
    color: "#22d3ee",
    fn: (x) => leakyRelu(x, 0.1),
    deriv: (x) => leakyReluDeriv(x, 0.1),
    formula: String.raw`\max(\alpha x,\,x)\;\;(\alpha=0.1)`,
    derivFormula: String.raw`\mathbb{1}[x>0] + \alpha\,\mathbb{1}[x\le 0]`,
  },
  {
    id: "elu",
    name: "ELU",
    color: "#34d399",
    fn: (x) => elu(x, 1),
    deriv: (x) => eluDeriv(x, 1),
    formula: String.raw`x \text{ if } x \ge 0,\; \alpha(e^x-1) \text{ else}`,
    derivFormula: String.raw`1 \text{ if } x \ge 0,\; \alpha e^x \text{ else}`,
  },
  {
    id: "gelu",
    name: "GELU",
    color: "#f472b6",
    fn: gelu,
    deriv: geluDeriv,
    formula: String.raw`x \cdot \Phi(x)`,
    derivFormula: String.raw`\Phi(x) + x\,\varphi(x)`,
  },
  {
    id: "silu",
    name: "SiLU / Swish",
    color: "#fbbf24",
    fn: silu,
    deriv: siluDeriv,
    formula: String.raw`x \cdot \sigma(x)`,
    derivFormula: String.raw`\sigma(x)\,(1 + x(1-\sigma(x)))`,
  },
  {
    id: "mish",
    name: "Mish",
    color: "#818cf8",
    fn: mish,
    deriv: mishDeriv,
    formula: String.raw`x \cdot \tanh(\mathrm{softplus}(x))`,
    derivFormula: String.raw`\tanh(\zeta) + x\,\mathrm{sech}^2(\zeta)\,\sigma(x)`,
  },
  {
    id: "softplus",
    name: "Softplus",
    color: "#fb7185",
    fn: softplus,
    deriv: softplusDeriv,
    formula: String.raw`\log(1 + e^x)`,
    derivFormula: String.raw`\sigma(x)`,
  },
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
