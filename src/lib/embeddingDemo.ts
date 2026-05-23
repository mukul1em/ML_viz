// Synthesize stable 16-D embeddings for each corpus item. Category gives the
// dominant signal (so things in the same cluster have high cosine similarity),
// 2D position adds a smaller continuous variation, and per-word noise adds
// the rest. Everything is deterministic and normalized to unit length.

import {
  corpus,
  type EmbeddingCategory,
  type EmbeddingItem,
} from "../data/embeddingCorpus";

export const EMBED_DIM = 16;

/** Roughly-orthogonal fingerprint per category (length = EMBED_DIM). */
const CATEGORY_FINGERPRINT: Record<EmbeddingCategory, number[]> = {
  animals:   [ 1.0,  0.2, -0.3, -0.4,  0.5,  0.1, -0.2,  0.6, -0.5,  0.3,  0.0, -0.1,  0.4, -0.2,  0.1,  0.3],
  vehicles:  [-0.3,  0.9,  0.4, -0.5,  0.2,  0.7, -0.1,  0.0,  0.3, -0.4,  0.5, -0.6,  0.1,  0.2, -0.3,  0.0],
  food:      [ 0.2, -0.4,  0.9,  0.3, -0.3,  0.5,  0.4, -0.2,  0.1, -0.1, -0.5,  0.2,  0.6, -0.3,  0.4, -0.2],
  emotions:  [-0.4,  0.1, -0.2,  0.9,  0.4, -0.3, -0.5,  0.4,  0.2,  0.6, -0.1, -0.3,  0.0,  0.3, -0.4,  0.5],
  tech:      [ 0.3, -0.5,  0.1, -0.2,  0.9, -0.4,  0.6,  0.1, -0.3,  0.4,  0.3,  0.2, -0.5,  0.1,  0.4, -0.1],
  sports:    [-0.2,  0.4, -0.5,  0.1, -0.3,  0.9,  0.2, -0.4,  0.5, -0.2,  0.4, -0.1,  0.3,  0.5, -0.2, -0.4],
  // Royalty is closer to "people / human roles" — overlaps lightly with emotions
  royalty:   [ 0.1, -0.2,  0.4, -0.1, -0.2,  0.3,  0.9,  0.5, -0.4,  0.2, -0.3,  0.6,  0.1, -0.4,  0.5, -0.2],
};

function hashString(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function seeded(seed: number, k: number): number {
  const x = Math.sin(seed * 0.0001 * 12.9898 + k * 78.233) * 43758.5453;
  return 2 * (x - Math.floor(x)) - 1;
}

function normalize(v: number[]): number[] {
  let s = 0;
  for (const x of v) s += x * x;
  const norm = Math.sqrt(s) || 1;
  return v.map((x) => x / norm);
}

export function embedItem(item: EmbeddingItem): number[] {
  const fp = CATEGORY_FINGERPRINT[item.category];
  const seed = hashString(item.text);

  const out = new Array(EMBED_DIM).fill(0);
  // Category fingerprint dominates
  for (let i = 0; i < EMBED_DIM; i++) out[i] = fp[i] * 0.65;
  // 2D position adds a smooth continuous variation
  out[0] += (item.x - 400) / 600;
  out[1] += (item.y - 300) / 500;
  // Per-word seeded noise
  for (let i = 0; i < EMBED_DIM; i++) {
    out[i] += seeded(seed, i) * 0.22;
  }

  // Royalty-specific: enforce the king/queen/man/woman analogy in vector space.
  // We add a "gender" component on dim 14 and a "royalty" component on dim 15
  // so the parallelogram closes cleanly in those two dims.
  if (item.category === "royalty") {
    const isFemale = ["queen", "woman", "princess"].includes(item.text);
    const isRoyal = ["king", "queen", "prince", "princess"].includes(item.text);
    out[14] = isFemale ? 0.85 : -0.85;
    out[15] = isRoyal ? 0.85 : -0.6;
  }

  return normalize(out);
}

/** Pre-compute and cache the embedding matrix. */
const _embedCache: number[][] = corpus.map(embedItem);

export function getEmbedding(idx: number): number[] {
  return _embedCache[idx];
}

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function magnitude(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

/** Cosine similarity in [-1, 1]. Inputs do not need to be normalized. */
export function cosine(a: number[], b: number[]): number {
  const denom = magnitude(a) * magnitude(b);
  if (denom === 0) return 0;
  return dot(a, b) / denom;
}

export interface Neighbor {
  item: EmbeddingItem;
  similarity: number;
}

/** Top-k most similar items to a given item (excluding itself). */
export function nearestNeighbors(
  queryIdx: number,
  k = 5,
  filter?: (it: EmbeddingItem) => boolean
): Neighbor[] {
  const q = _embedCache[queryIdx];
  const results: Neighbor[] = [];
  for (let i = 0; i < corpus.length; i++) {
    if (i === queryIdx) continue;
    if (filter && !filter(corpus[i])) continue;
    results.push({ item: corpus[i], similarity: cosine(q, _embedCache[i]) });
  }
  results.sort((a, b) => b.similarity - a.similarity);
  return results.slice(0, k);
}

/** Vector arithmetic: a − b + c, returning top-k items closest to result. */
export function analogy(
  aIdx: number,
  bIdx: number,
  cIdx: number,
  k = 4
): Neighbor[] {
  const a = _embedCache[aIdx];
  const b = _embedCache[bIdx];
  const c = _embedCache[cIdx];
  const target = a.map((x, i) => x - b[i] + c[i]);
  const excluded = new Set([aIdx, bIdx, cIdx]);
  const results: Neighbor[] = corpus
    .map((item, i) => ({ item, similarity: cosine(target, _embedCache[i]) }))
    .filter((_, i) => !excluded.has(i));
  results.sort((a, b) => b.similarity - a.similarity);
  return results.slice(0, k);
}
