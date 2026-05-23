import { useMemo } from "react";
import type { BertComponentId } from "../../data/bertComponents";
import { OP_COLOR, type OpType } from "../gpt/GPTGraphCanvas";

export type BertGranularity = "block" | "op";

interface GraphNode {
  id: string;
  label: string;
  sub?: string;
  x: number;
  y: number;
  w: number;
  h: number;
  op: OpType;
  componentId?: BertComponentId;
}

interface GraphEdge {
  from: string;
  to: string;
  fromSide?: "right";
  toSide?: "right";
}

interface GroupBox {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  multiplier?: string;
  color?: string;
}

const NODE_W = 180;
const NODE_H = 48;
const NODE_H_TALL = 60;
const CANVAS_W = 640;
const CENTER_X = CANVAS_W / 2;

interface BuildResult {
  nodes: GraphNode[];
  edges: GraphEdge[];
  groups: GroupBox[];
  width: number;
  height: number;
}

export function buildBertGraph(
  granularity: BertGranularity,
  numLayers: number
): BuildResult {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const groups: GroupBox[] = [];
  let y = 16;
  const push = (n: Omit<GraphNode, "x" | "y"> & { x?: number; y?: number }) => {
    const node = {
      x: n.x ?? CENTER_X - (n.w ?? NODE_W) / 2,
      y: n.y ?? y,
      ...n,
    } as GraphNode;
    nodes.push(node);
    return node;
  };

  push({
    id: "input",
    label: "input + [CLS]/[SEP]",
    sub: "sentence pair",
    w: NODE_W,
    h: NODE_H_TALL,
    op: "io",
    componentId: "input",
  });
  y += NODE_H_TALL + 24;

  push({
    id: "tokenization",
    label: "WordPiece",
    sub: "## continuation",
    w: NODE_W,
    h: NODE_H_TALL,
    op: "embedding",
    componentId: "tokenization",
  });
  y += NODE_H_TALL + 28;

  // Three parallel embeddings
  const embY = y;
  const colW = (CANVAS_W - 80) / 3;
  push({
    id: "tokenEmb",
    label: "Token Embedding",
    sub: "W_E ∈ ℝᵛˣᵈ",
    x: 40,
    y: embY,
    w: colW,
    h: NODE_H_TALL,
    op: "embedding",
    componentId: "tokenEmbedding",
  });
  push({
    id: "posEmb",
    label: "Position Embedding",
    sub: "learned · max 512",
    x: 40 + colW,
    y: embY,
    w: colW,
    h: NODE_H_TALL,
    op: "embedding",
    componentId: "positionalEmbedding",
  });
  push({
    id: "segEmb",
    label: "Segment Embedding",
    sub: "A / B (size 2)",
    x: 40 + 2 * colW,
    y: embY,
    w: colW,
    h: NODE_H_TALL,
    op: "embedding",
    componentId: "segmentEmbedding",
  });
  y = embY + NODE_H_TALL + 28;

  push({
    id: "embAdd",
    label: "+ sum (3 embeddings)",
    w: 200,
    h: 40,
    op: "add",
    componentId: "embeddingSum",
  });
  y += 40 + 22;

  push({
    id: "embLN",
    label: "LayerNorm + Dropout",
    w: NODE_W,
    h: NODE_H,
    op: "norm",
    componentId: "embedLayerNorm",
  });
  y += NODE_H + 28;

  // Encoder block group
  const groupTop = y;

  if (granularity === "block") {
    push({
      id: "attn",
      label: "Bidirectional Attention",
      sub: "multi-head · ALL-to-ALL",
      w: NODE_W + 50,
      h: NODE_H_TALL,
      op: "attention",
      componentId: "selfAttention",
    });
    y += NODE_H_TALL + 22;
  } else {
    // OP granularity
    const small = 110;
    const yQ = y;
    push({
      id: "qProj",
      label: "Linear (Q)",
      x: 40,
      y: yQ,
      w: small,
      h: 38,
      op: "linear",
      componentId: "selfAttention",
    });
    push({
      id: "kProj",
      label: "Linear (K)",
      x: CENTER_X - small / 2,
      y: yQ,
      w: small,
      h: 38,
      op: "linear",
      componentId: "selfAttention",
    });
    push({
      id: "vProj",
      label: "Linear (V)",
      x: CANVAS_W - 40 - small,
      y: yQ,
      w: small,
      h: 38,
      op: "linear",
      componentId: "selfAttention",
    });
    y = yQ + 38 + 16;
    push({
      id: "qk",
      label: "QKᵀ / √d_k",
      w: 150,
      h: 38,
      op: "mul",
      componentId: "selfAttention",
    });
    y += 38 + 12;
    push({
      id: "softmax",
      label: "Softmax (no mask)",
      sub: "row-wise · bidirectional",
      w: NODE_W + 30,
      h: NODE_H_TALL,
      op: "softmax",
      componentId: "selfAttention",
    });
    y += NODE_H_TALL + 12;
    push({
      id: "av",
      label: "matmul · V",
      w: 130,
      h: 38,
      op: "mul",
      componentId: "selfAttention",
    });
    y += 38 + 12;
    push({
      id: "attnProj",
      label: "Linear (proj)",
      w: 150,
      h: 38,
      op: "linear",
      componentId: "selfAttention",
    });
    y += 38 + 18;
    edges.push({ from: "embLN", to: "qProj" });
    edges.push({ from: "embLN", to: "kProj" });
    edges.push({ from: "embLN", to: "vProj" });
    edges.push({ from: "qProj", to: "qk" });
    edges.push({ from: "kProj", to: "qk" });
    edges.push({ from: "qk", to: "softmax" });
    edges.push({ from: "vProj", to: "av" });
    edges.push({ from: "softmax", to: "av" });
    edges.push({ from: "av", to: "attnProj" });
  }

  push({
    id: "add1",
    label: "add (residual)",
    w: 140,
    h: 40,
    op: "add",
    componentId: "residual1",
  });
  y += 40 + 18;

  push({
    id: "ln1",
    label: "LayerNorm  (POST)",
    w: NODE_W,
    h: NODE_H,
    op: "norm",
    componentId: "layerNorm1",
  });
  y += NODE_H + 22;

  if (granularity === "block") {
    push({
      id: "ffn",
      label: "Feed-Forward",
      sub: "4× · GELU",
      w: NODE_W + 30,
      h: NODE_H_TALL,
      op: "linear",
      componentId: "ffn",
    });
    y += NODE_H_TALL + 22;
  } else {
    push({
      id: "ffnW1",
      label: "Linear (4d expand)",
      w: 180,
      h: 38,
      op: "linear",
      componentId: "ffn",
    });
    y += 38 + 12;
    push({
      id: "gelu",
      label: "GELU",
      w: 100,
      h: 36,
      op: "activation",
      componentId: "ffn",
    });
    y += 36 + 12;
    push({
      id: "ffnW2",
      label: "Linear (project)",
      w: 180,
      h: 38,
      op: "linear",
      componentId: "ffn",
    });
    y += 38 + 18;
    edges.push({ from: "ln1", to: "ffnW1" });
    edges.push({ from: "ffnW1", to: "gelu" });
    edges.push({ from: "gelu", to: "ffnW2" });
  }

  push({
    id: "add2",
    label: "add (residual)",
    w: 140,
    h: 40,
    op: "add",
    componentId: "residual2",
  });
  y += 40 + 18;

  push({
    id: "ln2",
    label: "LayerNorm  (POST)",
    w: NODE_W,
    h: NODE_H,
    op: "norm",
    componentId: "layerNorm2",
  });
  y += NODE_H + 16;

  const groupBottom = y;
  groups.push({
    id: "encoder",
    x: 24,
    y: groupTop - 12,
    w: CANVAS_W - 48,
    h: groupBottom - groupTop + 4,
    label: "Encoder block",
    multiplier: `× ${numLayers}`,
    color: "#22d3ee",
  });
  y += 24;

  // Heads branching: [CLS] → Pooler → NSP, sequence → MLM
  const headTop = y;
  // Branch labels
  push({
    id: "clsBranch",
    label: "[CLS] vector",
    sub: "(sequence-level)",
    x: 30,
    y: headTop,
    w: 200,
    h: NODE_H_TALL,
    op: "embedding",
    componentId: "pooler",
  });
  push({
    id: "tokensBranch",
    label: "All token vectors",
    sub: "(token-level)",
    x: CANVAS_W - 30 - 200,
    y: headTop,
    w: 200,
    h: NODE_H_TALL,
    op: "embedding",
    componentId: "mlmHead",
  });
  y = headTop + NODE_H_TALL + 22;

  push({
    id: "pooler",
    label: "Pooler · tanh",
    sub: "Linear → tanh",
    x: 30 + 100 - NODE_W / 2,
    y,
    w: NODE_W,
    h: NODE_H_TALL,
    op: "activation",
    componentId: "pooler",
  });
  push({
    id: "mlmHead",
    label: "MLM Head",
    sub: "Linear → GELU → LN → Wᵀ",
    x: CANVAS_W - 30 - 200 + 100 - NODE_W / 2,
    y,
    w: NODE_W,
    h: NODE_H_TALL,
    op: "linear",
    componentId: "mlmHead",
  });
  y += NODE_H_TALL + 22;

  push({
    id: "nspHead",
    label: "NSP Head",
    sub: "Linear → 2 classes",
    x: 30 + 100 - NODE_W / 2,
    y,
    w: NODE_W,
    h: NODE_H_TALL,
    op: "softmax",
    componentId: "nspHead",
  });
  push({
    id: "mlmOut",
    label: "Token logits",
    sub: `softmax over V`,
    x: CANVAS_W - 30 - 200 + 100 - NODE_W / 2,
    y,
    w: NODE_W,
    h: NODE_H_TALL,
    op: "softmax",
    componentId: "mlmHead",
  });
  y += NODE_H_TALL + 16;

  // Sequential edges
  const seq = (() => {
    if (granularity === "block") {
      return [
        ["input", "tokenization"],
        ["tokenization", "tokenEmb"],
        ["tokenization", "posEmb"],
        ["tokenization", "segEmb"],
        ["tokenEmb", "embAdd"],
        ["posEmb", "embAdd"],
        ["segEmb", "embAdd"],
        ["embAdd", "embLN"],
        ["embLN", "attn"],
        ["attn", "add1"],
        ["add1", "ln1"],
        ["ln1", "ffn"],
        ["ffn", "add2"],
        ["add2", "ln2"],
        ["ln2", "clsBranch"],
        ["ln2", "tokensBranch"],
        ["clsBranch", "pooler"],
        ["tokensBranch", "mlmHead"],
        ["pooler", "nspHead"],
        ["mlmHead", "mlmOut"],
      ];
    }
    return [
      ["input", "tokenization"],
      ["tokenization", "tokenEmb"],
      ["tokenization", "posEmb"],
      ["tokenization", "segEmb"],
      ["tokenEmb", "embAdd"],
      ["posEmb", "embAdd"],
      ["segEmb", "embAdd"],
      ["embAdd", "embLN"],
      ["attnProj", "add1"],
      ["add1", "ln1"],
      ["ffnW2", "add2"],
      ["add2", "ln2"],
      ["ln2", "clsBranch"],
      ["ln2", "tokensBranch"],
      ["clsBranch", "pooler"],
      ["tokensBranch", "mlmHead"],
      ["pooler", "nspHead"],
      ["mlmHead", "mlmOut"],
    ];
  })();
  for (const [a, b] of seq) edges.push({ from: a, to: b });

  // Residual side-arcs
  edges.push({ from: "embLN", to: "add1", fromSide: "right", toSide: "right" });
  edges.push({ from: "ln1", to: "add2", fromSide: "right", toSide: "right" });

  return { nodes, edges, groups, width: CANVAS_W, height: y + 24 };
}

interface Props {
  granularity: BertGranularity;
  numLayers: number;
  selected: BertComponentId;
  onSelect: (id: BertComponentId) => void;
}

export default function BertGraphCanvas({
  granularity,
  numLayers,
  selected,
  onSelect,
}: Props) {
  const { nodes, edges, groups, width, height } = useMemo(
    () => buildBertGraph(granularity, numLayers),
    [granularity, numLayers]
  );
  const nodeMap = useMemo(() => {
    const m = new Map<string, GraphNode>();
    for (const n of nodes) m.set(n.id, n);
    return m;
  }, [nodes]);

  return (
    <div className="relative w-full h-full overflow-auto bg-[radial-gradient(circle_at_1px_1px,rgba(255,255,255,0.05)_1px,transparent_0)] [background-size:18px_18px]">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        className="block mx-auto"
        style={{ maxWidth: "100%" }}
      >
        <defs>
          <marker
            id="bert-arrow"
            markerWidth="10"
            markerHeight="10"
            refX="6"
            refY="3"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M0,0 L0,6 L7,3 z" fill="#5b6478" />
          </marker>
          <marker
            id="bert-arrow-res"
            markerWidth="10"
            markerHeight="10"
            refX="6"
            refY="3"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M0,0 L0,6 L7,3 z" fill="#a48bff" />
          </marker>
        </defs>

        {/* Group boxes */}
        {groups.map((g) => {
          const c = g.color ?? "#22d3ee";
          return (
            <g key={g.id}>
              <rect
                x={g.x}
                y={g.y}
                width={g.w}
                height={g.h}
                rx={14}
                fill={`${c}0c`}
                stroke={`${c}99`}
                strokeWidth={1.5}
                strokeDasharray="5 5"
              />
              <g>
                <rect
                  x={g.x + 14}
                  y={g.y - 10}
                  width={108}
                  height={20}
                  rx={6}
                  fill={c}
                />
                <text
                  x={g.x + 14 + 54}
                  y={g.y + 4}
                  textAnchor="middle"
                  fontSize={11}
                  fontWeight={700}
                  fill="#062029"
                >
                  {g.label}
                </text>
              </g>
              {g.multiplier && (
                <text
                  x={g.x + g.w + 14}
                  y={g.y + g.h / 2 + 5}
                  fill={c}
                  fontSize={13}
                  fontWeight={700}
                  fontFamily="JetBrains Mono, monospace"
                >
                  {g.multiplier}
                </text>
              )}
            </g>
          );
        })}

        {edges.map((e, i) => {
          const a = nodeMap.get(e.from);
          const b = nodeMap.get(e.to);
          if (!a || !b) return null;
          return (
            <EdgePath
              key={`${e.from}-${e.to}-${i}`}
              a={a}
              b={b}
              fromSide={e.fromSide}
              toSide={e.toSide}
            />
          );
        })}

        {nodes.map((n) => (
          <Node
            key={n.id}
            n={n}
            selected={n.componentId === selected}
            onClick={() => n.componentId && onSelect(n.componentId)}
          />
        ))}
      </svg>
    </div>
  );
}

function Node({
  n,
  selected,
  onClick,
}: {
  n: GraphNode;
  selected: boolean;
  onClick: () => void;
}) {
  const color = OP_COLOR[n.op];
  return (
    <g
      transform={`translate(${n.x}, ${n.y})`}
      className="cursor-pointer"
      onClick={onClick}
    >
      <rect
        width={n.w}
        height={n.h}
        rx={10}
        fill={selected ? `${color}40` : `${color}1f`}
        stroke={selected ? "#ffffff" : `${color}88`}
        strokeWidth={selected ? 2 : 1.4}
        style={{
          filter: selected
            ? `drop-shadow(0 0 14px ${color}80)`
            : `drop-shadow(0 4px 14px rgba(0,0,0,0.35))`,
        }}
      />
      <text
        x={n.w / 2}
        y={n.sub ? n.h / 2 - 4 : n.h / 2 + 4}
        textAnchor="middle"
        fill="#ffffff"
        fontSize={12.5}
        fontWeight={600}
        pointerEvents="none"
      >
        {n.label}
      </text>
      {n.sub && (
        <text
          x={n.w / 2}
          y={n.h - 14}
          textAnchor="middle"
          fill="rgba(255,255,255,0.6)"
          fontSize={10.5}
          fontFamily="JetBrains Mono, monospace"
          pointerEvents="none"
        >
          {n.sub}
        </text>
      )}
      <circle cx={10} cy={10} r={3.5} fill={color} pointerEvents="none" />
    </g>
  );
}

function EdgePath({
  a,
  b,
  fromSide,
  toSide,
}: {
  a: GraphNode;
  b: GraphNode;
  fromSide?: "right";
  toSide?: "right";
}) {
  const isResidual = fromSide === "right" && toSide === "right";
  if (isResidual) {
    const ax = a.x + a.w;
    const ay = a.y + a.h / 2;
    const bx = b.x + b.w;
    const by = b.y + b.h / 2;
    const outX = Math.max(ax, bx) + 36;
    return (
      <path
        d={`M ${ax} ${ay} L ${outX} ${ay} L ${outX} ${by} L ${bx} ${by}`}
        stroke="#a48bff"
        strokeWidth={1.4}
        fill="none"
        strokeDasharray="3 4"
        markerEnd="url(#bert-arrow-res)"
        opacity={0.7}
      />
    );
  }
  const startX = a.x + a.w / 2;
  const startY = a.y + a.h;
  const endX = b.x + b.w / 2;
  const endY = b.y;
  const midY = (startY + endY) / 2;
  const d =
    Math.abs(startX - endX) < 1
      ? `M ${startX} ${startY} L ${endX} ${endY}`
      : `M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`;
  return (
    <path
      d={d}
      stroke="#5b6478"
      strokeWidth={1.4}
      fill="none"
      markerEnd="url(#bert-arrow)"
    />
  );
}
