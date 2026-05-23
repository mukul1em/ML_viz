import { useMemo } from "react";
import type { GPTComponentId } from "../../data/gptComponents";

export type Granularity = "block" | "op";

export type OpType =
  | "io"
  | "embedding"
  | "norm"
  | "linear"
  | "attention"
  | "softmax"
  | "activation"
  | "add"
  | "mul";

export const OP_COLOR: Record<OpType, string> = {
  io: "#ef4444",
  embedding: "#fb923c",
  norm: "#38bdf8",
  linear: "#34d399",
  attention: "#22d3ee",
  softmax: "#a78bfa",
  activation: "#fbbf24",
  add: "#f472b6",
  mul: "#facc15",
};

export const OP_LABEL: Record<OpType, string> = {
  io: "I/O",
  embedding: "Embedding",
  norm: "LayerNorm",
  linear: "Linear",
  attention: "Attention",
  softmax: "Softmax",
  activation: "GELU",
  add: "Add",
  mul: "Mul",
};

export interface GraphNode {
  id: string;
  label: string;
  sub?: string;
  x: number;
  y: number;
  w: number;
  h: number;
  op: OpType;
  componentId?: GPTComponentId;
}

export interface GraphEdge {
  from: string;
  to: string;
  toSide?: "top" | "bottom" | "left" | "right";
  fromSide?: "top" | "bottom" | "left" | "right";
}

export interface GroupBox {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  multiplier?: string;
  multiplierY?: number;
}

interface BuildResult {
  nodes: GraphNode[];
  edges: GraphEdge[];
  groups: GroupBox[];
  width: number;
  height: number;
}

const NODE_W = 180;
const NODE_W_SM = 130;
const NODE_H = 48;
const NODE_H_TALL = 60;
const CANVAS_W = 620;
const CENTER_X = CANVAS_W / 2;

function nodeCenter(n: GraphNode) {
  return { x: n.x + n.w / 2, y: n.y + n.h / 2 };
}

export function buildGPTGraph(
  granularity: Granularity,
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

  // input
  push({
    id: "input",
    label: "input",
    w: NODE_W_SM,
    h: NODE_H,
    op: "io",
    componentId: "input",
  });
  y += NODE_H + 30;

  // token + positional embeddings (parallel)
  const tok = push({
    id: "tokenEmb",
    label: "Token Embedding",
    sub: "W_E ∈ ℝᵛˣᵈ",
    x: CENTER_X - NODE_W - 20,
    w: NODE_W,
    h: NODE_H_TALL,
    op: "embedding",
    componentId: "tokenEmbedding",
  });
  const pos = push({
    id: "posEmb",
    label: "Positional Embedding",
    sub: "W_P ∈ ℝᴸˣᵈ",
    x: CENTER_X + 20,
    w: NODE_W,
    h: NODE_H_TALL,
    op: "embedding",
    componentId: "positionalEmbedding",
  });
  y = Math.max(tok.y + tok.h, pos.y + pos.h) + 28;

  // add
  push({
    id: "embAdd",
    label: "add",
    w: 100,
    h: 40,
    op: "add",
    componentId: "embeddingSum",
  });
  y += 40 + 30;

  // Decoder cycle group
  const groupTop = y;

  push({
    id: "ln1",
    label: "LayerNorm",
    w: NODE_W,
    h: NODE_H,
    op: "norm",
    componentId: "layerNorm1",
  });
  y += NODE_H + 24;

  if (granularity === "block") {
    push({
      id: "attn",
      label: "CausalSelfAttention",
      sub: `heads: causal · masked`,
      w: NODE_W + 30,
      h: NODE_H_TALL,
      op: "attention",
      componentId: "selfAttention",
    });
    y += NODE_H_TALL + 24;
  } else {
    // OP granularity: expand attention into Q/K/V, softmax, matmul, proj
    const left = CENTER_X - NODE_W - 8;
    const mid = CENTER_X - NODE_W / 2;
    const right = CENTER_X + 8;
    const small = 110;
    const qy = y;
    push({
      id: "qProj",
      label: "Linear (Q)",
      x: left,
      w: small,
      h: 38,
      op: "linear",
      componentId: "selfAttention",
    });
    push({
      id: "kProj",
      label: "Linear (K)",
      x: mid + (NODE_W / 2 - small / 2),
      w: small,
      h: 38,
      op: "linear",
      componentId: "selfAttention",
    });
    push({
      id: "vProj",
      label: "Linear (V)",
      x: right + NODE_W - small,
      w: small,
      h: 38,
      op: "linear",
      componentId: "selfAttention",
    });
    y = qy + 38 + 18;
    push({
      id: "qk",
      label: "QKᵀ / √d_k",
      w: 150,
      h: 38,
      op: "mul",
      componentId: "selfAttention",
    });
    y += 38 + 14;
    push({
      id: "mask",
      label: "Causal Mask + Softmax",
      sub: "row-wise normalization",
      w: NODE_W + 30,
      h: NODE_H_TALL,
      op: "softmax",
      componentId: "selfAttention",
    });
    y += NODE_H_TALL + 14;
    push({
      id: "av",
      label: "matmul · V",
      w: 130,
      h: 38,
      op: "mul",
      componentId: "selfAttention",
    });
    y += 38 + 14;
    push({
      id: "attnProj",
      label: "Linear (proj)",
      w: 150,
      h: 38,
      op: "linear",
      componentId: "selfAttention",
    });
    y += 38 + 18;

    edges.push({ from: "ln1", to: "qProj" });
    edges.push({ from: "ln1", to: "kProj" });
    edges.push({ from: "ln1", to: "vProj" });
    edges.push({ from: "qProj", to: "qk" });
    edges.push({ from: "kProj", to: "qk" });
    edges.push({ from: "qk", to: "mask" });
    edges.push({ from: "vProj", to: "av" });
    edges.push({ from: "mask", to: "av" });
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
  y += 40 + 24;

  push({
    id: "ln2",
    label: "LayerNorm",
    w: NODE_W,
    h: NODE_H,
    op: "norm",
    componentId: "layerNorm2",
  });
  y += NODE_H + 24;

  if (granularity === "block") {
    push({
      id: "ffn",
      label: "FeedForward",
      sub: "expansion: 4× · GELU",
      w: NODE_W + 30,
      h: NODE_H_TALL,
      op: "linear",
      componentId: "ffn",
    });
    y += NODE_H_TALL + 24;
  } else {
    push({
      id: "ffnW1",
      label: "Linear (4d expand)",
      w: 180,
      h: 38,
      op: "linear",
      componentId: "ffn",
    });
    y += 38 + 14;
    push({
      id: "gelu",
      label: "GELU",
      w: 100,
      h: 36,
      op: "activation",
      componentId: "ffn",
    });
    y += 36 + 14;
    push({
      id: "ffnW2",
      label: "Linear (project)",
      w: 180,
      h: 38,
      op: "linear",
      componentId: "ffn",
    });
    y += 38 + 18;
    edges.push({ from: "ln2", to: "ffnW1" });
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

  const groupBottom = y;
  groups.push({
    id: "decoder",
    x: 28,
    y: groupTop - 12,
    w: CANVAS_W - 56,
    h: groupBottom - groupTop + 6,
    label: "Decoder block",
    multiplier: `× ${numLayers}`,
    multiplierY: (groupTop + groupBottom) / 2,
  });
  y += 24;

  push({
    id: "finalLn",
    label: "Final LayerNorm",
    w: NODE_W,
    h: NODE_H,
    op: "norm",
    componentId: "finalLayerNorm",
  });
  y += NODE_H + 22;

  push({
    id: "lmHead",
    label: "LM Head (Linear)",
    sub: "→ ℝⱽ (logits)",
    w: NODE_W,
    h: NODE_H_TALL,
    op: "linear",
    componentId: "lmHead",
  });
  y += NODE_H_TALL + 22;

  push({
    id: "softmax",
    label: "softmax",
    w: NODE_W_SM,
    h: NODE_H,
    op: "softmax",
    componentId: "sampling",
  });
  y += NODE_H + 22;

  push({
    id: "output",
    label: "output",
    w: NODE_W_SM,
    h: NODE_H,
    op: "io",
    componentId: "sampling",
  });
  y += NODE_H;

  // Sequential edges (the ones not specified inside OP granularity)
  const seqOrder = (() => {
    if (granularity === "block") {
      return [
        ["input", "tokenEmb"],
        ["input", "posEmb"],
        ["tokenEmb", "embAdd"],
        ["posEmb", "embAdd"],
        ["embAdd", "ln1"],
        ["ln1", "attn"],
        ["attn", "add1"],
        ["add1", "ln2"],
        ["ln2", "ffn"],
        ["ffn", "add2"],
        ["add2", "finalLn"],
        ["finalLn", "lmHead"],
        ["lmHead", "softmax"],
        ["softmax", "output"],
      ];
    }
    return [
      ["input", "tokenEmb"],
      ["input", "posEmb"],
      ["tokenEmb", "embAdd"],
      ["posEmb", "embAdd"],
      ["embAdd", "ln1"],
      ["attnProj", "add1"],
      ["add1", "ln2"],
      ["ffnW2", "add2"],
      ["add2", "finalLn"],
      ["finalLn", "lmHead"],
      ["lmHead", "softmax"],
      ["softmax", "output"],
    ];
  })();
  for (const [a, b] of seqOrder) edges.push({ from: a, to: b });

  // Residual side-arrows (input → add)
  edges.push({ from: "embAdd", to: "add1", fromSide: "right", toSide: "right" });
  edges.push({ from: "add1", to: "add2", fromSide: "right", toSide: "right" });

  return { nodes, edges, groups, width: CANVAS_W, height: y + 24 };
}

interface Props {
  granularity: Granularity;
  numLayers: number;
  selected: GPTComponentId;
  onSelect: (id: GPTComponentId) => void;
}

export default function GPTGraphCanvas({
  granularity,
  numLayers,
  selected,
  onSelect,
}: Props) {
  const { nodes, edges, groups, width, height } = useMemo(
    () => buildGPTGraph(granularity, numLayers),
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
            id="arrow-edge"
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
            id="arrow-edge-residual"
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
        {groups.map((g) => (
          <g key={g.id}>
            <rect
              x={g.x}
              y={g.y}
              width={g.w}
              height={g.h}
              rx={14}
              fill="rgba(251, 146, 60, 0.05)"
              stroke="rgba(251, 146, 60, 0.55)"
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
                fill="#fb923c"
              />
              <text
                x={g.x + 14 + 54}
                y={g.y + 4}
                textAnchor="middle"
                fontSize={11}
                fontWeight={700}
                fill="#1f1006"
              >
                {g.label}
              </text>
            </g>
            {g.multiplier && (
              <text
                x={g.x + g.w + 14}
                y={(g.multiplierY ?? g.y + g.h / 2) + 5}
                fill="#fb923c"
                fontSize={13}
                fontWeight={700}
                fontFamily="JetBrains Mono, monospace"
              >
                {g.multiplier}
              </text>
            )}
          </g>
        ))}

        {/* Edges (drawn before nodes so they sit underneath) */}
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

        {/* Nodes */}
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
        fontFamily="Inter, sans-serif"
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
      {/* Small op-type pill */}
      <g pointerEvents="none">
        <circle cx={10} cy={10} r={3.5} fill={color} />
      </g>
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
  fromSide?: "top" | "bottom" | "left" | "right";
  toSide?: "top" | "bottom" | "left" | "right";
}) {
  const isResidual = fromSide === "right" && toSide === "right";

  if (isResidual) {
    // Side-arc: bracket-shaped residual
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
        markerEnd="url(#arrow-edge-residual)"
        opacity={0.7}
      />
    );
  }

  const ac = nodeCenter(a);
  const bc = nodeCenter(b);
  const startX = ac.x;
  const startY = a.y + a.h;
  const endX = bc.x;
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
      markerEnd="url(#arrow-edge)"
    />
  );
}
