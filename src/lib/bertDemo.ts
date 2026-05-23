// BERT-flavored mock data for the visualizer.

export interface BertToken {
  token: string;
  id: number;
  segment: 0 | 1;
  isSpecial: boolean; // [CLS], [SEP], [MASK]
  isContinuation: boolean; // ## prefix
  maskedFor?: string; // if masked, original token
}

export const SAMPLE_TEXT_A = "the cat sat on the mat";
export const SAMPLE_TEXT_B = "it slept peacefully";

const WORDPIECE_HINT: Record<string, string[]> = {
  // toy "merges" so the user sees ## continuation tokens.
  peacefully: ["peace", "##fully"],
  unbelievable: ["un", "##believ", "##able"],
  embedding: ["em", "##bed", "##ding"],
  transformer: ["trans", "##former"],
};

function hashStringToId(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) % 30522; // BERT vocab size
}

function pieceTokens(word: string): string[] {
  const hint = WORDPIECE_HINT[word.toLowerCase()];
  if (hint) return hint;
  // fallback: split words longer than 7 chars to fake subwording
  if (word.length > 7) {
    const mid = Math.ceil(word.length / 2);
    return [word.slice(0, mid), "##" + word.slice(mid)];
  }
  return [word];
}

function tokenId(token: string): number {
  if (token === "[CLS]") return 101;
  if (token === "[SEP]") return 102;
  if (token === "[MASK]") return 103;
  if (token === "[PAD]") return 0;
  if (token === "[UNK]") return 100;
  return hashStringToId(token);
}

export function bertTokenize(
  textA: string,
  textB?: string,
  options?: { maskIndices?: number[] }
): BertToken[] {
  const out: BertToken[] = [];
  out.push({
    token: "[CLS]",
    id: 101,
    segment: 0,
    isSpecial: true,
    isContinuation: false,
  });
  for (const w of textA.split(/\s+/).filter(Boolean)) {
    for (const p of pieceTokens(w)) {
      out.push({
        token: p,
        id: tokenId(p),
        segment: 0,
        isSpecial: false,
        isContinuation: p.startsWith("##"),
      });
    }
  }
  out.push({
    token: "[SEP]",
    id: 102,
    segment: 0,
    isSpecial: true,
    isContinuation: false,
  });
  if (textB && textB.trim()) {
    for (const w of textB.split(/\s+/).filter(Boolean)) {
      for (const p of pieceTokens(w)) {
        out.push({
          token: p,
          id: tokenId(p),
          segment: 1,
          isSpecial: false,
          isContinuation: p.startsWith("##"),
        });
      }
    }
    out.push({
      token: "[SEP]",
      id: 102,
      segment: 1,
      isSpecial: true,
      isContinuation: false,
    });
  }

  if (options?.maskIndices) {
    for (const i of options.maskIndices) {
      if (i >= 0 && i < out.length && !out[i].isSpecial) {
        out[i] = {
          ...out[i],
          maskedFor: out[i].token,
          token: "[MASK]",
          id: 103,
          isSpecial: true,
        };
      }
    }
  }
  return out.slice(0, 24);
}

function seeded(seed: number, dim: number): number {
  const x = Math.sin(seed * 12.9898 + dim * 78.233) * 43758.5453;
  return 2 * (x - Math.floor(x)) - 1;
}

export function bertTokenEmbed(id: number, d = 24): number[] {
  return Array.from({ length: d }, (_, i) => seeded(id + 1, i + 1));
}

export function bertPositionEmbed(pos: number, d = 24): number[] {
  // BERT uses learned positional embeddings, but for visualization
  // we synthesize them as a deterministic vector per position.
  return Array.from({ length: d }, (_, i) => seeded(7919 + pos * 13, i + 3) * 0.7);
}

export function bertSegmentEmbed(seg: number, d = 24): number[] {
  // Just two learned vectors (segment A = 0, segment B = 1).
  return Array.from({ length: d }, (_, i) =>
    seg === 0 ? 0.55 * Math.cos((i + 1) * 0.5) : 0.55 * Math.sin((i + 1) * 0.7)
  );
}

/**
 * Bidirectional attention: full N×N matrix (no causal mask), softmaxed per row.
 * Some structure so it looks plausible: stronger diagonal + a couple of focal points.
 */
export function bertAttention(n: number, headSeed = 0): number[][] {
  const rows: number[][] = [];
  for (let i = 0; i < n; i++) {
    const logits: number[] = [];
    for (let j = 0; j < n; j++) {
      const diag = i === j ? 1.0 : 0;
      const near = Math.abs(i - j) === 1 ? 0.7 : 0;
      const cls = j === 0 ? 0.55 : 0;
      const stuffy = Math.abs(Math.sin(i * 1.3 + j * 0.7 + headSeed * 1.1));
      const noise = 0.4 * seeded(headSeed * 31 + i * 7 + j, 11);
      let s = 0.2 + 0.5 * stuffy + noise;
      if (headSeed % 4 === 0) s += diag * 1.3 + near * 0.6;
      else if (headSeed % 4 === 1) s += near * 1.4 + cls * 0.5;
      else if (headSeed % 4 === 2) s += cls * 1.6 + diag * 0.4;
      else s += stuffy * 0.6 + diag * 0.5;
      logits.push(s);
    }
    const max = Math.max(...logits);
    const exps = logits.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    rows.push(exps.map((e) => e / sum));
  }
  return rows;
}

/** Top-k MLM predictions for the masked position. */
export function mlmPredict(
  context: string,
  original?: string
): { token: string; prob: number; isOriginal?: boolean }[] {
  const candidates: Record<string, string[]> = {
    cat: ["cat", "dog", "kitten", "fox", "puppy", "animal", "creature", "pet"],
    sat: ["sat", "lay", "slept", "stood", "perched", "rested", "sits", "lies"],
    mat: ["mat", "couch", "floor", "bed", "rug", "ground", "chair", "table"],
    sunny: ["sunny", "rainy", "cloudy", "bright", "warm", "lovely", "perfect", "quiet"],
  };
  const pool =
    candidates[original?.toLowerCase() ?? ""] ??
    ["the", "a", "of", "and", "in", "to", "for", "with"];
  const hash = context.length + (original?.length ?? 0);
  const logits = pool.map((_tok, i) => {
    const base = i === 0 ? 3.0 : 2.2 - i * 0.35;
    return base + 0.2 * seeded(hash + i, 17);
  });
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return pool
    .map((token, i) => ({
      token,
      prob: exps[i] / sum,
      isOriginal: original ? token.toLowerCase() === original.toLowerCase() : false,
    }))
    .sort((a, b) => b.prob - a.prob);
}

/** Deterministic NSP probability — based on string similarity. */
export function nspPredict(a: string, b: string): { isNext: number; notNext: number } {
  if (!b.trim()) return { isNext: 0.5, notNext: 0.5 };
  const wordsA = new Set(a.toLowerCase().split(/\s+/).filter(Boolean));
  const wordsB = new Set(b.toLowerCase().split(/\s+/).filter(Boolean));
  let overlap = 0;
  for (const w of wordsB) if (wordsA.has(w)) overlap++;
  const lenScore = Math.min(1, Math.abs(a.length - b.length) / 40);
  const score = overlap / Math.max(1, Math.min(wordsA.size, wordsB.size)) - 0.4 * lenScore;
  const p = 1 / (1 + Math.exp(-(score * 3 + 0.4)));
  return { isNext: p, notNext: 1 - p };
}

export const BERT_CONFIG_PRESETS = {
  "BERT-base": { layers: 12, d_model: 768, heads: 12, ffn: 3072, vocab: 30522, ctx: 512 },
  "BERT-large": { layers: 24, d_model: 1024, heads: 16, ffn: 4096, vocab: 30522, ctx: 512 },
  DistilBERT: { layers: 6, d_model: 768, heads: 12, ffn: 3072, vocab: 30522, ctx: 512 },
  "RoBERTa-base": { layers: 12, d_model: 768, heads: 12, ffn: 3072, vocab: 50265, ctx: 512 },
};
