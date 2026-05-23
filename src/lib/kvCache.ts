/* =============================== kv cache math =============================== */

export interface KVCacheConfig {
  nHeadsKv: number;
  dHead: number;
  nLayers: number;
  batch: number;
  dtypeBytes: number;
}

/** Bytes for the cache at a given context length. */
export function cacheBytesAt(cfg: KVCacheConfig, seqLen: number): number {
  return 2 * cfg.batch * cfg.nHeadsKv * cfg.dHead * seqLen * cfg.nLayers * cfg.dtypeBytes;
}

/**
 * Per-step attention FLOPs *with* a cache: only the latest Q row interacts
 * with the full K/V matrices.   2 (mul + add) · n_heads · seq_len · d_head
 */
export function flopsWithCache(cfg: KVCacheConfig, seqLen: number): number {
  return 2 * cfg.nHeadsKv * seqLen * cfg.dHead;
}

/**
 * Per-step FLOPs *without* a cache (recompute the full prefix QK^T and softmax·V):
 * 2 · 2 · n_heads · seq_len^2 · d_head      (QK^T + softmax·V over the full L×L matrix)
 */
export function flopsRecompute(cfg: KVCacheConfig, seqLen: number): number {
  return 4 * cfg.nHeadsKv * seqLen * seqLen * cfg.dHead;
}

/** Cumulative FLOPs to decode `seqLen` tokens with vs without cache. */
export function cumulativeFlops(cfg: KVCacheConfig, seqLen: number, withCache: boolean): number {
  let total = 0;
  for (let l = 1; l <= seqLen; l++) {
    total += withCache ? flopsWithCache(cfg, l) : flopsRecompute(cfg, l);
  }
  return total;
}

/* =============================== paged attention =============================== */

export interface PagedBlock {
  blockId: number;
  tokens: number; // # tokens currently stored in this block (0..blockSize)
  reqId: number;
}

/** vLLM-style: divide each request's KV cache into fixed-size blocks. */
export function paginate(seqLens: number[], blockSize: number): PagedBlock[][] {
  return seqLens.map((L, reqId) => {
    const numFull = Math.floor(L / blockSize);
    const rem = L - numFull * blockSize;
    const blocks: PagedBlock[] = [];
    for (let b = 0; b < numFull; b++) blocks.push({ blockId: b, tokens: blockSize, reqId });
    if (rem > 0) blocks.push({ blockId: numFull, tokens: rem, reqId });
    return blocks;
  });
}

/** Total physical block count across all requests. */
export function totalBlocks(seqLens: number[], blockSize: number): number {
  return seqLens.reduce((s, L) => s + Math.ceil(L / blockSize), 0);
}

/** Memory wasted vs a perfectly-packed cache (fragmentation overhead). */
export function fragmentation(seqLens: number[], blockSize: number): { wasted: number; ratio: number } {
  const used = seqLens.reduce((s, L) => s + L, 0);
  const allocated = totalBlocks(seqLens, blockSize) * blockSize;
  return { wasted: allocated - used, ratio: allocated > 0 ? (allocated - used) / allocated : 0 };
}

/* =============================== format helpers =============================== */

export function formatBytes(n: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u += 1;
  }
  return `${v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)} ${units[u]}`;
}

export function formatFlops(n: number): string {
  const units = ["FLOP", "kFLOP", "MFLOP", "GFLOP", "TFLOP", "PFLOP"];
  let v = n;
  let u = 0;
  while (v >= 1000 && u < units.length - 1) {
    v /= 1000;
    u += 1;
  }
  return `${v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)} ${units[u]}`;
}
