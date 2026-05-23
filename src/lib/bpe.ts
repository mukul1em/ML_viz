/* =============================== byte-pair encoding =============================== */

export interface MergeOp {
  /** The pair being merged, e.g. ["t", "h"]. */
  a: string;
  b: string;
  /** The resulting token, "th". */
  merged: string;
  /** The merge's rank — lower fires earlier in `encode`. */
  rank: number;
  /** Frequency of the pair at the moment the merge was chosen. */
  count: number;
}

/** Count adjacent token pairs across the whole word list. */
function pairCounts(words: string[][], wordFreq: number[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (let w = 0; w < words.length; w++) {
    const word = words[w];
    const freq = wordFreq[w];
    for (let i = 0; i < word.length - 1; i++) {
      const key = `${word[i]}\u0001${word[i + 1]}`;
      counts.set(key, (counts.get(key) ?? 0) + freq);
    }
  }
  return counts;
}

/** Apply a single merge to a word: replace every adjacent (a,b) with (a+b). */
function applyMerge(word: string[], a: string, b: string): string[] {
  if (word.length < 2) return word;
  const out: string[] = [];
  let i = 0;
  while (i < word.length) {
    if (i < word.length - 1 && word[i] === a && word[i + 1] === b) {
      out.push(a + b);
      i += 2;
    } else {
      out.push(word[i]);
      i += 1;
    }
  }
  return out;
}

/** Compress duplicate words into (word, freq) pairs for speed. */
function compressCorpus(text: string): { words: string[][]; freq: number[] } {
  const groups = new Map<string, number>();
  for (const w of text.toLowerCase().split(/\s+/).filter(Boolean)) {
    groups.set(w, (groups.get(w) ?? 0) + 1);
  }
  const words: string[][] = [];
  const freq: number[] = [];
  for (const [w, f] of groups) {
    words.push(Array.from(w));
    freq.push(f);
  }
  return { words, freq };
}

/**
 * Learn the first `maxMerges` BPE merges from `corpus`.
 * Returns the ordered list of merge operations and the final vocabulary.
 */
export function learnBPE(corpus: string, maxMerges: number): {
  merges: MergeOp[];
  vocab: string[];
  /** Snapshot of corpus after each step (for animation). Index 0 = initial chars. */
  snapshots: string[][][];
} {
  const { words, freq } = compressCorpus(corpus);
  let current = words.map((w) => w.slice());

  const merges: MergeOp[] = [];
  const vocab = new Set<string>();
  for (const w of current) for (const c of w) vocab.add(c);
  const snapshots: string[][][] = [current.map((w) => w.slice())];

  for (let step = 0; step < maxMerges; step++) {
    const counts = pairCounts(current, freq);
    if (counts.size === 0) break;

    let bestKey = "";
    let bestCount = 0;
    for (const [key, c] of counts) {
      if (c > bestCount) {
        bestCount = c;
        bestKey = key;
      }
    }
    if (bestCount < 2) break;

    const [a, b] = bestKey.split("\u0001");
    const merged = a + b;
    merges.push({ a, b, merged, rank: step, count: bestCount });
    vocab.add(merged);

    current = current.map((w) => applyMerge(w, a, b));
    snapshots.push(current.map((w) => w.slice()));
  }

  return { merges, vocab: Array.from(vocab), snapshots };
}

/**
 * Encode an input string using a list of merges (in rank order). Splits on
 * whitespace and applies merges to each word until none can fire.
 */
export function encode(text: string, merges: MergeOp[]): { tokens: string[]; perWord: string[][] } {
  const mergeMap = new Map<string, number>();
  for (const m of merges) mergeMap.set(`${m.a}\u0001${m.b}`, m.rank);

  const perWord: string[][] = [];
  const tokens: string[] = [];

  for (const word of text.toLowerCase().split(/\s+/).filter(Boolean)) {
    let chunks: string[] = Array.from(word);
    while (chunks.length > 1) {
      let bestPair = -1;
      let bestRank = Infinity;
      for (let i = 0; i < chunks.length - 1; i++) {
        const r = mergeMap.get(`${chunks[i]}\u0001${chunks[i + 1]}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestPair = i;
        }
      }
      if (bestPair < 0) break;
      const a = chunks[bestPair];
      const b = chunks[bestPair + 1];
      chunks = [...chunks.slice(0, bestPair), a + b, ...chunks.slice(bestPair + 2)];
    }
    perWord.push(chunks);
    tokens.push(...chunks);
  }

  return { tokens, perWord };
}

/** Token / word / char count comparison for a given input. */
export function countComparison(text: string, merges: MergeOp[]): {
  chars: number;
  words: number;
  tokens: number;
} {
  const cleaned = text.toLowerCase();
  const chars = Array.from(cleaned.replace(/\s+/g, "")).length;
  const words = cleaned.split(/\s+/).filter(Boolean).length;
  const tokens = encode(text, merges).tokens.length;
  return { chars, words, tokens };
}

/* =============================== sample corpus =============================== */

export const SAMPLE_CORPUS = [
  "the quick brown fox jumps over the lazy dog",
  "the cat sat on the mat",
  "the dog sat on the mat",
  "a quick brown dog ran",
  "the lazy cat sat",
  "the quick fox ran over the dog",
  "the dog and the cat sat",
  "the lazy fox jumps over the cat",
  "a cat ran quickly",
  "the brown cat and the brown dog",
].join("\n");
