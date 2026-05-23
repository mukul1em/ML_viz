// Lightweight, deterministic mock data used by the GPT visualizer.
// We don't actually run a transformer — we generate seeded patterns that
// *look* like real activations / attention so users can build intuition.

export const SAMPLE_PROMPT = "The cat sat on the";

// Pretend BPE tokenizer: split on whitespace, prepend a space marker except
// for the first token. (Toy approximation — real BPE merges subwords.)
export function fakeTokenize(text: string): { token: string; id: number }[] {
  if (!text.trim()) return [];
  const words = text.split(/(\s+)/).filter((s) => s.length > 0);
  const tokens: { token: string; id: number }[] = [];
  for (const w of words) {
    if (/^\s+$/.test(w)) continue;
    tokens.push({ token: w, id: hashStringToId(w) });
  }
  return tokens.slice(0, 16);
}

function hashStringToId(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) % 50257; // GPT-2 vocab size
}

// Deterministic pseudo-random in [-1, 1]
export function seeded(seed: number, dim: number): number {
  const x = Math.sin(seed * 12.9898 + dim * 78.233) * 43758.5453;
  return 2 * (x - Math.floor(x)) - 1;
}

// Synthetic token embedding row (length d) for a given token id.
export function tokenEmbed(id: number, d = 24): number[] {
  return Array.from({ length: d }, (_, i) => seeded(id + 1, i + 1));
}

// Sinusoidal positional embedding (transformer's original recipe).
export function positionalEmbed(pos: number, d = 24): number[] {
  return Array.from({ length: d }, (_, i) => {
    const k = Math.floor(i / 2);
    const freq = 1 / Math.pow(10000, (2 * k) / d);
    return i % 2 === 0 ? Math.sin(pos * freq) : Math.cos(pos * freq);
  });
}

// Synthetic attention pattern: lower-triangular, with structured peaks so
// it visually resembles a real attention head.
export function fakeAttention(
  n: number,
  headSeed = 0
): number[][] {
  const rows: number[][] = [];
  for (let i = 0; i < n; i++) {
    const logits = new Array(n).fill(-Infinity);
    for (let j = 0; j <= i; j++) {
      // base score + structured biases per head
      const diag = i === j ? 1.2 : 0;
      const prev = i - j === 1 ? 1.1 : 0;
      const subjectish = j === 0 ? 0.4 : 0;
      const stuffy = Math.abs(Math.sin(i * 1.7 + j * 0.6 + headSeed * 1.3));
      const noise = 0.5 * seeded(headSeed * 31 + i * 7 + j, 11);
      let s = 0.2 + 0.6 * stuffy + noise;
      if (headSeed % 4 === 0) s += diag * 1.5 + prev * 0.6;
      else if (headSeed % 4 === 1) s += prev * 1.6 + diag * 0.4;
      else if (headSeed % 4 === 2) s += subjectish * 1.8 + diag * 0.6;
      else s += diag * 0.8 + Math.max(0, 0.8 - 0.2 * (i - j));
      logits[j] = s;
    }
    const max = Math.max(...logits.filter((x) => Number.isFinite(x)));
    const exps = logits.map((x) => (Number.isFinite(x) ? Math.exp(x - max) : 0));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    rows.push(exps.map((e) => e / sum));
  }
  return rows;
}

// Fake next-token distribution sorted descending; used for the sampling viz.
export function fakeNextTokenDist(
  prompt: string,
  temperature = 1
): { token: string; prob: number }[] {
  const candidates = [
    "mat",
    "couch",
    "floor",
    "table",
    "bed",
    "rug",
    "chair",
    "shelf",
    "ground",
    "windowsill",
  ];
  const base = candidates.map((tok, i) => ({
    tok,
    logit: 4.5 - i * 0.55 + 0.25 * seeded(hashSimple(prompt) + i, 9),
  }));
  const T = Math.max(temperature, 1e-3);
  const scaled = base.map((b) => b.logit / T);
  const m = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - m));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return base
    .map((b, i) => ({ token: b.tok, prob: exps[i] / sum }))
    .sort((a, b) => b.prob - a.prob);
}

function hashSimple(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return h;
}

export const GPT_CONFIG_PRESETS = {
  "GPT-2 small": { layers: 12, d_model: 768, heads: 12, ffn: 3072 },
  "GPT-2 medium": { layers: 24, d_model: 1024, heads: 16, ffn: 4096 },
  "GPT-2 large": { layers: 36, d_model: 1280, heads: 20, ffn: 5120 },
  "GPT-3 175B": { layers: 96, d_model: 12288, heads: 96, ffn: 49152 },
};
