import Donut from "./Donut";
import { OP_COLOR, OP_LABEL, type OpType } from "./GPTGraphCanvas";

interface Props {
  modelName: string;
  numLayers: number;
  dModel: number;
  numHeads: number;
  ffnDim: number;
  vocabSize?: number;
  contextLength?: number;
}

function computeOpCounts(numLayers: number): Record<OpType, number> {
  const counts: Record<OpType, number> = {
    io: 0,
    embedding: 0,
    norm: 0,
    linear: 0,
    attention: 0,
    softmax: 0,
    activation: 0,
    add: 0,
    mul: 0,
  };
  counts.io = 2; // input, output
  counts.embedding = 2; // token + positional
  for (let i = 0; i < numLayers; i++) {
    counts.norm += 2;
    counts.linear += 4 + 2; // Q,K,V,proj + FFN W1,W2
    counts.softmax += 1; // attention softmax
    counts.add += 2; // 2 residuals
    counts.activation += 1; // GELU
    counts.mul += 2; // QK and AV matmuls
  }
  counts.norm += 1; // final LN
  counts.linear += 1; // LM head
  counts.softmax += 1; // final sampling softmax
  return counts;
}

function approxParams({
  numLayers,
  dModel,
  ffnDim,
  vocabSize,
  contextLength,
}: {
  numLayers: number;
  dModel: number;
  ffnDim: number;
  vocabSize: number;
  contextLength: number;
}) {
  const perLayer =
    4 * dModel * dModel + // Q,K,V, proj
    2 * dModel * ffnDim + // FFN W1, W2
    4 * dModel; // 2 LN params (γ,β) × 2
  const emb = vocabSize * dModel + contextLength * dModel;
  const finalLN = 2 * dModel;
  const lmHead = 0; // weight-tied
  return numLayers * perLayer + emb + finalLN + lmHead;
}

function humanCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return `${n}`;
}

export default function GPTSidebar({
  modelName,
  numLayers,
  dModel,
  numHeads,
  ffnDim,
  vocabSize = 50257,
  contextLength = 1024,
}: Props) {
  const counts = computeOpCounts(numLayers);
  const totalOps = Object.values(counts).reduce((a, b) => a + b, 0);

  const opData = (Object.keys(counts) as OpType[])
    .filter((k) => counts[k] > 0)
    .map((k) => ({
      label: OP_LABEL[k],
      key: k,
      value: counts[k],
      color: OP_COLOR[k],
      pct: (counts[k] / totalOps) * 100,
    }))
    .sort((a, b) => b.value - a.value);

  const params = approxParams({
    numLayers,
    dModel,
    ffnDim,
    vocabSize,
    contextLength,
  });

  const precData = [
    { label: "FP16 weights", value: 60, color: "#7c5cff" },
    { label: "FP32 activations", value: 30, color: "#22d3ee" },
    { label: "INT8 cache", value: 10, color: "#fb923c" },
  ];

  return (
    <div className="flex flex-col gap-4 w-full">
      {/* Model summary */}
      <div className="rounded-2xl bg-ink-900/70 border border-ink-800 p-4">
        <div className="text-sm font-semibold text-white">{modelName}</div>
        <div className="text-[11px] text-ink-400 mt-0.5 leading-relaxed">
          {totalOps} nodes · {opData.length} op types · {vocabSize.toLocaleString()}-token vocab
          <br />
          Context length {contextLength.toLocaleString()} · ≈ {humanCount(params)} parameters
        </div>
        <div className="flex flex-wrap gap-1.5 mt-3">
          <span className="px-2 py-0.5 rounded-md bg-accent/15 border border-accent/40 text-[10px] uppercase tracking-wider text-accent-soft font-semibold">
            Decoder-only
          </span>
          <span className="px-2 py-0.5 rounded-md bg-cyan-500/15 border border-cyan-500/40 text-[10px] uppercase tracking-wider text-cyan-300 font-semibold">
            Pre-norm
          </span>
          <span className="px-2 py-0.5 rounded-md bg-pink-500/15 border border-pink-500/40 text-[10px] uppercase tracking-wider text-pink-300 font-semibold">
            Autoregressive
          </span>
        </div>
      </div>

      {/* Operation types */}
      <div className="rounded-2xl bg-ink-900/70 border border-ink-800 p-4">
        <div className="flex items-center justify-between mb-2">
          <div>
            <div className="text-sm font-semibold text-white">
              Operation types
            </div>
            <div className="text-[11px] text-ink-400">{totalOps} operations</div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <Donut
            data={opData}
            size={130}
            strokeWidth={20}
            centerLabel={`${opData.length}`}
            centerSub="types"
          />
          <div className="flex-1 grid grid-cols-1 gap-1.5 text-[11px] min-w-0">
            {opData.slice(0, 6).map((d) => (
              <div key={d.key} className="flex items-center gap-2 min-w-0">
                <span
                  className="w-2.5 h-2.5 rounded-sm shrink-0"
                  style={{ background: d.color }}
                />
                <span className="text-ink-200 truncate flex-1">{d.label}</span>
                <span className="font-mono text-ink-400 tabular-nums">
                  {d.pct.toFixed(1)}%
                </span>
              </div>
            ))}
            {opData.length > 6 && (
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-sm bg-ink-500 shrink-0" />
                <span className="text-ink-300 flex-1">Other</span>
                <span className="font-mono text-ink-400 tabular-nums">
                  {opData
                    .slice(6)
                    .reduce((a, b) => a + b.pct, 0)
                    .toFixed(1)}
                  %
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Architecture stats */}
      <div className="rounded-2xl bg-ink-900/70 border border-ink-800 p-4">
        <div className="text-sm font-semibold text-white mb-2">
          Architecture
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Stat label="layers" value={numLayers.toString()} />
          <Stat label="d_model" value={dModel.toLocaleString()} />
          <Stat label="heads" value={numHeads.toString()} />
          <Stat label="d_head" value={(dModel / numHeads).toString()} />
          <Stat label="ffn" value={ffnDim.toLocaleString()} />
          <Stat label="vocab" value={vocabSize.toLocaleString()} />
        </div>
      </div>

      {/* Precision */}
      <div className="rounded-2xl bg-ink-900/70 border border-ink-800 p-4">
        <div className="text-sm font-semibold text-white">
          Precision breakdown
        </div>
        <div className="text-[11px] text-ink-400">Typical inference mix</div>
        <div className="flex items-center gap-4 mt-2">
          <Donut
            data={precData}
            size={120}
            strokeWidth={20}
            centerLabel="FP16"
            centerSub="primary"
          />
          <div className="flex-1 grid grid-cols-1 gap-1.5 text-[11px]">
            {precData.map((d) => (
              <div key={d.label} className="flex items-center gap-2">
                <span
                  className="w-2.5 h-2.5 rounded-sm"
                  style={{ background: d.color }}
                />
                <span className="text-ink-200 flex-1">{d.label}</span>
                <span className="font-mono text-ink-400 tabular-nums">
                  {d.value}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-ink-950/70 border border-ink-800 px-2.5 py-1.5">
      <div className="text-[9px] uppercase tracking-wider text-ink-400">
        {label}
      </div>
      <div className="font-mono text-sm text-white">{value}</div>
    </div>
  );
}
