/* =============================== 2-D loss surfaces =============================== */

export type Vec2 = [number, number];

export interface LossSpec {
  id: LossId;
  name: string;
  formula: string;
  description: string;
  domain: { xMin: number; xMax: number; yMin: number; yMax: number };
  minima: Vec2[];
  /** Default learning rate that "just works". */
  defaultLr: number;
  /** Default starting point. */
  start: Vec2;
  f: (p: Vec2) => number;
  grad: (p: Vec2) => Vec2;
}

export type LossId = "bowl" | "ravine" | "rosenbrock" | "saddle";

const BOWL: LossSpec = {
  id: "bowl",
  name: "Convex bowl",
  formula: String.raw`f(x,y) = \tfrac12 (x^{2} + y^{2})`,
  description: "Quadratic. Trivially convex. Every optimizer should converge.",
  domain: { xMin: -3, xMax: 3, yMin: -3, yMax: 3 },
  minima: [[0, 0]],
  defaultLr: 0.3,
  start: [-2.5, 2.2],
  f: ([x, y]) => 0.5 * (x * x + y * y),
  grad: ([x, y]) => [x, y],
};

const RAVINE: LossSpec = {
  id: "ravine",
  name: "Ill-conditioned ravine",
  formula: String.raw`f(x,y) = \tfrac12 x^{2} + 5\,y^{2}`,
  description:
    "10× curvature in y. Vanilla SGD bounces; momentum + Adam exploit the direction.",
  domain: { xMin: -3, xMax: 3, yMin: -1.5, yMax: 1.5 },
  minima: [[0, 0]],
  defaultLr: 0.1,
  start: [-2.5, 1.0],
  f: ([x, y]) => 0.5 * x * x + 5 * y * y,
  grad: ([x, y]) => [x, 10 * y],
};

const ROSENBROCK: LossSpec = {
  id: "rosenbrock",
  name: "Rosenbrock banana",
  formula: String.raw`f(x,y) = (1-x)^{2} + 100\,(y - x^{2})^{2}`,
  description: "Curved narrow valley. The classical non-convex benchmark.",
  domain: { xMin: -2, xMax: 2, yMin: -1, yMax: 3 },
  minima: [[1, 1]],
  defaultLr: 0.002,
  start: [-1.5, 2.5],
  f: ([x, y]) => {
    const a = 1 - x;
    const b = y - x * x;
    return a * a + 100 * b * b;
  },
  grad: ([x, y]) => {
    const b = y - x * x;
    return [-2 * (1 - x) - 400 * x * b, 200 * b];
  },
};

const SADDLE: LossSpec = {
  id: "saddle",
  name: "Saddle",
  formula: String.raw`f(x,y) = x^{2} - y^{2}`,
  description: "Plain GD lingers at the origin. Momentum / Adam break the symmetry.",
  domain: { xMin: -2, xMax: 2, yMin: -2, yMax: 2 },
  minima: [],
  defaultLr: 0.2,
  start: [0.001, 0.001],
  f: ([x, y]) => x * x - y * y,
  grad: ([x, y]) => [2 * x, -2 * y],
};

export const LOSSES: Record<LossId, LossSpec> = {
  bowl: BOWL,
  ravine: RAVINE,
  rosenbrock: ROSENBROCK,
  saddle: SADDLE,
};

/* =============================== optimizers =============================== */

export interface OptimizerSpec {
  id: OptimizerId;
  name: string;
  color: string;
  formula: string;
  /** One-step update; receives current parameters and the gradient at them. */
  step: (theta: Vec2, grad: Vec2, lr: number, state: OptState) => Vec2;
}

export type OptimizerId =
  | "sgd"
  | "momentum"
  | "nesterov"
  | "adagrad"
  | "rmsprop"
  | "adam"
  | "adamw"
  | "lion";

export interface OptState {
  t: number;
  m: Vec2;
  v: Vec2;
  beta1: number;
  beta2: number;
  eps: number;
  /** Decoupled weight-decay coefficient λ. Only AdamW / Lion use it. */
  wd: number;
}

export function makeState(beta1: number, beta2: number, eps: number, wd = 0): OptState {
  return { t: 0, m: [0, 0], v: [0, 0], beta1, beta2, eps, wd };
}

const OPTIMIZERS: Record<OptimizerId, OptimizerSpec> = {
  sgd: {
    id: "sgd",
    name: "SGD",
    color: "#7c5cff",
    formula: String.raw`\theta \leftarrow \theta - \eta\,g`,
    step: (theta, g, lr) => [theta[0] - lr * g[0], theta[1] - lr * g[1]],
  },
  momentum: {
    id: "momentum",
    name: "Momentum",
    color: "#22d3ee",
    formula: String.raw`v \leftarrow \beta_1 v + g;\quad \theta \leftarrow \theta - \eta v`,
    step: (theta, g, lr, s) => {
      s.m[0] = s.beta1 * s.m[0] + g[0];
      s.m[1] = s.beta1 * s.m[1] + g[1];
      return [theta[0] - lr * s.m[0], theta[1] - lr * s.m[1]];
    },
  },
  nesterov: {
    id: "nesterov",
    name: "Nesterov",
    color: "#34d399",
    formula: String.raw`v \leftarrow \beta_1 v + g(\theta - \eta\beta_1 v);\;\theta \leftarrow \theta - \eta v`,
    // We use the simplified equivalent form with the gradient already computed at theta.
    // For a faithful Nesterov we'd query grad(theta - eta*beta1*v); the runner below does it.
    step: (theta, g, lr, s) => {
      s.m[0] = s.beta1 * s.m[0] + g[0];
      s.m[1] = s.beta1 * s.m[1] + g[1];
      return [theta[0] - lr * s.m[0], theta[1] - lr * s.m[1]];
    },
  },
  adagrad: {
    id: "adagrad",
    name: "AdaGrad",
    color: "#f472b6",
    formula: String.raw`v \mathrel{+}= g^{2};\;\theta \leftarrow \theta - \eta\,g/\sqrt{v + \varepsilon}`,
    step: (theta, g, lr, s) => {
      s.v[0] += g[0] * g[0];
      s.v[1] += g[1] * g[1];
      return [
        theta[0] - (lr * g[0]) / (Math.sqrt(s.v[0]) + s.eps),
        theta[1] - (lr * g[1]) / (Math.sqrt(s.v[1]) + s.eps),
      ];
    },
  },
  rmsprop: {
    id: "rmsprop",
    name: "RMSProp",
    color: "#fbbf24",
    formula: String.raw`v \leftarrow \beta_2 v + (1-\beta_2) g^{2};\; \theta \leftarrow \theta - \eta\,g/\sqrt{v + \varepsilon}`,
    step: (theta, g, lr, s) => {
      s.v[0] = s.beta2 * s.v[0] + (1 - s.beta2) * g[0] * g[0];
      s.v[1] = s.beta2 * s.v[1] + (1 - s.beta2) * g[1] * g[1];
      return [
        theta[0] - (lr * g[0]) / (Math.sqrt(s.v[0]) + s.eps),
        theta[1] - (lr * g[1]) / (Math.sqrt(s.v[1]) + s.eps),
      ];
    },
  },
  adam: {
    id: "adam",
    name: "Adam",
    color: "#818cf8",
    formula: String.raw`m \leftarrow \beta_1 m + (1{-}\beta_1)g;\; v \leftarrow \beta_2 v + (1{-}\beta_2)g^{2}\\\hat m = m/(1{-}\beta_1^{t}),\; \hat v = v/(1{-}\beta_2^{t})\\\theta \leftarrow \theta - \eta\,\hat m / (\sqrt{\hat v} + \varepsilon)`,
    step: (theta, g, lr, s) => {
      s.t += 1;
      s.m[0] = s.beta1 * s.m[0] + (1 - s.beta1) * g[0];
      s.m[1] = s.beta1 * s.m[1] + (1 - s.beta1) * g[1];
      s.v[0] = s.beta2 * s.v[0] + (1 - s.beta2) * g[0] * g[0];
      s.v[1] = s.beta2 * s.v[1] + (1 - s.beta2) * g[1] * g[1];
      const mh0 = s.m[0] / (1 - Math.pow(s.beta1, s.t));
      const mh1 = s.m[1] / (1 - Math.pow(s.beta1, s.t));
      const vh0 = s.v[0] / (1 - Math.pow(s.beta2, s.t));
      const vh1 = s.v[1] / (1 - Math.pow(s.beta2, s.t));
      return [
        theta[0] - (lr * mh0) / (Math.sqrt(vh0) + s.eps),
        theta[1] - (lr * mh1) / (Math.sqrt(vh1) + s.eps),
      ];
    },
  },
  adamw: {
    id: "adamw",
    name: "AdamW",
    color: "#a855f7",
    formula: String.raw`\text{Adam step}\;+\;\text{decoupled WD:}\\\theta \leftarrow \theta - \eta\!\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} + \lambda\,\theta\right)`,
    step: (theta, g, lr, s) => {
      s.t += 1;
      s.m[0] = s.beta1 * s.m[0] + (1 - s.beta1) * g[0];
      s.m[1] = s.beta1 * s.m[1] + (1 - s.beta1) * g[1];
      s.v[0] = s.beta2 * s.v[0] + (1 - s.beta2) * g[0] * g[0];
      s.v[1] = s.beta2 * s.v[1] + (1 - s.beta2) * g[1] * g[1];
      const mh0 = s.m[0] / (1 - Math.pow(s.beta1, s.t));
      const mh1 = s.m[1] / (1 - Math.pow(s.beta1, s.t));
      const vh0 = s.v[0] / (1 - Math.pow(s.beta2, s.t));
      const vh1 = s.v[1] / (1 - Math.pow(s.beta2, s.t));
      return [
        theta[0] - lr * (mh0 / (Math.sqrt(vh0) + s.eps) + s.wd * theta[0]),
        theta[1] - lr * (mh1 / (Math.sqrt(vh1) + s.eps) + s.wd * theta[1]),
      ];
    },
  },
  lion: {
    id: "lion",
    name: "Lion",
    color: "#fb7185",
    formula: String.raw`c = \beta_1 m + (1{-}\beta_1) g;\quad \theta \leftarrow \theta - \eta\,(\operatorname{sign}(c) + \lambda\,\theta)\\m \leftarrow \beta_2 m + (1{-}\beta_2) g`,
    step: (theta, g, lr, s) => {
      // interpolated direction (uses momentum BEFORE update)
      const c0 = s.beta1 * s.m[0] + (1 - s.beta1) * g[0];
      const c1 = s.beta1 * s.m[1] + (1 - s.beta1) * g[1];
      const updated: Vec2 = [
        theta[0] - lr * (Math.sign(c0) + s.wd * theta[0]),
        theta[1] - lr * (Math.sign(c1) + s.wd * theta[1]),
      ];
      // momentum update (after parameter update)
      s.m[0] = s.beta2 * s.m[0] + (1 - s.beta2) * g[0];
      s.m[1] = s.beta2 * s.m[1] + (1 - s.beta2) * g[1];
      return updated;
    },
  },
};

export const OPTIMIZER_LIST: OptimizerSpec[] = [
  OPTIMIZERS.sgd,
  OPTIMIZERS.momentum,
  OPTIMIZERS.nesterov,
  OPTIMIZERS.adagrad,
  OPTIMIZERS.rmsprop,
  OPTIMIZERS.adam,
  OPTIMIZERS.adamw,
  OPTIMIZERS.lion,
];

/* =============================== runner =============================== */

export interface RunOptions {
  loss: LossSpec;
  start: Vec2;
  lr: number;
  steps: number;
  beta1: number;
  beta2: number;
  eps: number;
  /** Decoupled weight-decay coefficient. AdamW and Lion use it; others ignore. */
  wd?: number;
}

export interface Trajectory {
  optId: OptimizerId;
  color: string;
  name: string;
  points: Vec2[];
  losses: number[];
}

export function runOptimizer(opt: OptimizerSpec, options: RunOptions): Trajectory {
  const { loss, start, lr, steps, beta1, beta2, eps, wd = 0 } = options;
  const state = makeState(beta1, beta2, eps, wd);
  let theta: Vec2 = [start[0], start[1]];
  const points: Vec2[] = [[theta[0], theta[1]]];
  const losses: number[] = [loss.f(theta)];

  for (let i = 0; i < steps; i++) {
    let g: Vec2;
    if (opt.id === "nesterov") {
      const look: Vec2 = [theta[0] - lr * beta1 * state.m[0], theta[1] - lr * beta1 * state.m[1]];
      g = loss.grad(look);
    } else {
      g = loss.grad(theta);
    }
    // Guard against numerical blowup.
    if (!Number.isFinite(g[0]) || !Number.isFinite(g[1])) break;
    theta = opt.step(theta, g, lr, state);
    if (!Number.isFinite(theta[0]) || !Number.isFinite(theta[1])) break;
    points.push([theta[0], theta[1]]);
    losses.push(loss.f(theta));
  }

  return { optId: opt.id, color: opt.color, name: opt.name, points, losses };
}
