/* =============================== mixture of experts =============================== */

/** Stable softmax. */
function softmax(x: number[]): number[] {
  const m = Math.max(...x);
  const e = x.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/**
 * Router scores: for each token (d-dim) compute a logit per expert via a
 * provided gating matrix W_g (n_experts × d). Returns the softmax
 * probability over experts for every token.
 */
export function routerScores(tokens: number[][], gates: number[][]): number[][] {
  const nExperts = gates.length;
  return tokens.map((tok) => {
    const logits = new Array(nExperts);
    for (let e = 0; e < nExperts; e++) {
      let s = 0;
      for (let i = 0; i < tok.length; i++) s += tok[i] * gates[e][i];
      logits[e] = s;
    }
    return softmax(logits);
  });
}

/**
 * Top-k gating per token. Zeros all but the top-k expert probabilities and
 * renormalizes those k to sum to 1 (Mixtral / Switch Transformer style).
 */
export function topKGate(scores: number[], k: number): number[] {
  const out = new Array(scores.length).fill(0);
  const indices = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]).slice(0, k);
  let s = 0;
  for (const i of indices) {
    out[i] = scores[i];
    s += scores[i];
  }
  if (s > 0) for (const i of indices) out[i] /= s;
  return out;
}

/** Per-expert utilization vector (= mean gate weight across tokens). */
export function expertUtilization(gatedScores: number[][]): number[] {
  const nExperts = gatedScores[0]?.length ?? 0;
  const util = new Array(nExperts).fill(0);
  for (const row of gatedScores) {
    for (let e = 0; e < nExperts; e++) util[e] += row[e];
  }
  const T = Math.max(gatedScores.length, 1);
  return util.map((v) => v / T);
}

/**
 * Switch Transformer load-balance auxiliary loss:
 *   L_aux = α · N · Σ_e f_e · P_e
 * where f_e = fraction of tokens whose top-1 expert is e,
 *       P_e = mean router probability for expert e.
 */
export function loadBalanceLoss(
  scoresFull: number[][],
  gatedScores: number[][],
  alpha = 0.01
): number {
  const nExperts = scoresFull[0]?.length ?? 0;
  const T = Math.max(scoresFull.length, 1);
  const f = new Array(nExperts).fill(0);
  const P = new Array(nExperts).fill(0);
  for (let t = 0; t < T; t++) {
    let bestE = 0;
    let bestS = -Infinity;
    for (let e = 0; e < nExperts; e++) {
      P[e] += scoresFull[t][e];
      if (gatedScores[t][e] > bestS) {
        bestS = gatedScores[t][e];
        bestE = e;
      }
    }
    f[bestE] += 1;
  }
  for (let e = 0; e < nExperts; e++) {
    f[e] /= T;
    P[e] /= T;
  }
  let loss = 0;
  for (let e = 0; e < nExperts; e++) loss += f[e] * P[e];
  return alpha * nExperts * loss;
}

/* =============================== params calculator =============================== */

export interface MoEConfig {
  dModel: number;
  dFF: number; // inner FFN dim, typically 4·dModel
  nExperts: number;
  topK: number;
  nLayers: number;
}

/** Dense FFN parameters per layer: 2 · d_model · d_ff */
export function denseFFNParams(cfg: MoEConfig): number {
  return 2 * cfg.dModel * cfg.dFF;
}

/** Total MoE parameters per layer (sum across experts). */
export function totalMoEParams(cfg: MoEConfig): number {
  return cfg.nExperts * denseFFNParams(cfg);
}

/** Active MoE parameters per token (top-k experts). */
export function activeMoEParams(cfg: MoEConfig): number {
  return cfg.topK * denseFFNParams(cfg);
}

/** Activation ratio (active / total). */
export function activationRatio(cfg: MoEConfig): number {
  return cfg.topK / cfg.nExperts;
}

/* =============================== presets =============================== */

export interface MoEModelPreset {
  name: string;
  totalParams: string; // human-readable
  activeParams: string;
  nExperts: number;
  topK: number;
  notes: string;
}

export const MOE_PRESETS: MoEModelPreset[] = [
  { name: "Switch Transformer", totalParams: "1.6T", activeParams: "26B", nExperts: 2048, topK: 1, notes: "Google, single-expert routing." },
  { name: "Mixtral 8×7B", totalParams: "47B", activeParams: "13B", nExperts: 8, topK: 2, notes: "Mistral, top-2 gating." },
  { name: "Mixtral 8×22B", totalParams: "141B", activeParams: "39B", nExperts: 8, topK: 2, notes: "Mistral, scaled-up Mixtral." },
  { name: "Qwen-MoE 14B-A2.7B", totalParams: "14B", activeParams: "2.7B", nExperts: 60, topK: 4, notes: "Alibaba, fine-grained experts." },
  { name: "DeepSeek-V3", totalParams: "671B", activeParams: "37B", nExperts: 256, topK: 8, notes: "Plus 1 shared expert; auxiliary-loss-free routing." },
  { name: "Grok-1", totalParams: "314B", activeParams: "~78B", nExperts: 8, topK: 2, notes: "xAI." },
];

/* =============================== tiny gate / token seeds =============================== */

/** Seeded LCG for deterministic embeddings / gate matrices. */
export function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

/** Build a (n × d) matrix in [-1, 1] with seed. */
export function randomMatrix(n: number, d: number, seed: number): number[][] {
  const rng = lcg(seed);
  return Array.from({ length: n }, () => Array.from({ length: d }, () => rng() * 2 - 1));
}
