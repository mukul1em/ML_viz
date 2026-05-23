import { useMemo } from "react";
import type { QwenComponentId } from "../../data/qwenComponents";

export type Granularity = "block" | "op";

type OpType =
  | "io"
  | "embedding"
  | "rmsnorm"
  | "linear"
  | "rope"
  | "gqa"
  | "softmax"
  | "silu"
  | "mul"
  | "add";

const OP_COLOR: Record<OpType, string> = {
  io: "#ef4444",
  embedding: "#fb923c",
  rmsnorm: "#22d3ee",
  linear: "#34d399",
  rope: "#ec4899",
  gqa: "#facc15",
  softmax: "#a78bfa",
  silu: "#60a5fa",
  mul: "#fbbf24",
  add: "#f472b6",
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
  componentId?: QwenComponentId;
  emphasized?: boolean;
}

export interface GraphEdge {
  from: string;
  to: string;
  fromSide?: "top" | "bottom" | "left" | "right";
  toSide?: "top" | "bottom" | "left" | "right";
  curve?: number;
}

export interface GroupBox {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  color: string;
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

const NODE_W = 200;
const NODE_W_SM = 140;
const NODE_H = 48;
const NODE_H_TALL = 60;
const CANVAS_W = 660;
const CENTER_X = CANVAS_W / 2;

function nodeCenter(n: GraphNode) {
  return { x: n.x + n.w / 2, y: n.y + n.h / 2 };
}

export function buildQwenGraph(
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
  y += NODE_H + 26;

  // tokenization
  push({
    id: "tok",
    label: "Tokenizer · BPE",
    sub: "vocab ≈ 152k",
    w: NODE_W,
    h: NODE_H_TALL,
    op: "embedding",
    componentId: "tokenization",
  });
  y += NODE_H_TALL + 22;

  // token embedding (NO positional!)
  push({
    id: "tokEmb",
    label: "Token Embedding",
    sub: "W_E ∈ ℝᵛˣᵈ · no pos emb",
    w: NODE_W,
    h: NODE_H_TALL,
    op: "embedding",
    componentId: "tokenEmbedding",
  });
  y += NODE_H_TALL + 30;

  /* ---------------------------- Decoder block ---------------------------- */
  const groupTop = y;

  push({
    id: "rms1",
    label: "RMSNorm",
    sub: "no centering · no bias",
    w: NODE_W,
    h: NODE_H_TALL,
    op: "rmsnorm",
    componentId: "rmsNorm1",
    emphasized: true,
  });
  y += NODE_H_TALL + 22;

  if (granularity === "block") {
    push({
      id: "attn",
      label: "Self-Attention",
      sub: "GQA · RoPE · causal",
      w: NODE_W + 30,
      h: NODE_H_TALL,
      op: "rope",
      componentId: "selfAttention",
      emphasized: true,
    });
    y += NODE_H_TALL + 22;
  } else {
    // OP mode: expand attention
    const left = CENTER_X - NODE_W - 6;
    const mid = CENTER_X - NODE_W / 2;
    const right = CENTER_X + 6;
    const small = 110;

    const projY = y;
    push({
      id: "qProj",
      label: "Linear · Q",
      sub: "→ q_heads",
      x: left,
      w: small,
      h: 44,
      op: "linear",
      componentId: "qkvProj",
    });
    push({
      id: "kProj",
      label: "Linear · K",
      sub: "→ kv_heads",
      x: mid + (NODE_W / 2 - small / 2),
      w: small,
      h: 44,
      op: "linear",
      componentId: "qkvProj",
    });
    push({
      id: "vProj",
      label: "Linear · V",
      sub: "→ kv_heads",
      x: right + NODE_W - small,
      w: small,
      h: 44,
      op: "linear",
      componentId: "qkvProj",
    });
    y = projY + 44 + 16;

    push({
      id: "rope",
      label: "RoPE · rotate(Q, K)",
      sub: "position via 2-D rotation",
      w: NODE_W + 30,
      h: NODE_H_TALL,
      op: "rope",
      componentId: "rope",
      emphasized: true,
    });
    y += NODE_H_TALL + 16;

    push({
      id: "gqa",
      label: "GQA · share K/V across heads",
      sub: "q_heads share kv_heads",
      w: NODE_W + 60,
      h: NODE_H_TALL,
      op: "gqa",
      componentId: "gqa",
      emphasized: true,
    });
    y += NODE_H_TALL + 16;

    push({
      id: "sdpa",
      label: "softmax(QKᵀ/√d_k) · V",
      sub: "causal mask",
      w: NODE_W + 30,
      h: NODE_H_TALL,
      op: "softmax",
      componentId: "scaledDotProduct",
    });
    y += NODE_H_TALL + 16;

    push({
      id: "attnOut",
      label: "Linear · out",
      w: NODE_W,
      h: 44,
      op: "linear",
      componentId: "attnOut",
    });
    y += 44 + 16;

    edges.push({ from: "rms1", to: "qProj" });
    edges.push({ from: "rms1", to: "kProj" });
    edges.push({ from: "rms1", to: "vProj" });
    edges.push({ from: "qProj", to: "rope" });
    edges.push({ from: "kProj", to: "rope" });
    edges.push({ from: "rope", to: "gqa" });
    edges.push({ from: "vProj", to: "gqa" });
    edges.push({ from: "gqa", to: "sdpa" });
    edges.push({ from: "sdpa", to: "attnOut" });
  }

  push({
    id: "add1",
    label: "add (residual)",
    w: 140,
    h: 40,
    op: "add",
    componentId: "residual1",
  });
  y += 40 + 22;

  push({
    id: "rms2",
    label: "RMSNorm",
    w: NODE_W,
    h: NODE_H,
    op: "rmsnorm",
    componentId: "rmsNorm2",
    emphasized: true,
  });
  y += NODE_H + 22;

  if (granularity === "block") {
    push({
      id: "swiglu",
      label: "SwiGLU FFN",
      sub: "down(silu(gate(x)) ⊙ up(x))",
      w: NODE_W + 30,
      h: NODE_H_TALL,
      op: "silu",
      componentId: "swiglu",
      emphasized: true,
    });
    y += NODE_H_TALL + 22;
  } else {
    const small = 130;
    const left = CENTER_X - small - 12;
    const right = CENTER_X + 12;
    const gateUpY = y;

    push({
      id: "gate",
      label: "Linear · gate",
      x: left,
      w: small,
      h: 44,
      op: "linear",
      componentId: "gateUp",
    });
    push({
      id: "up",
      label: "Linear · up",
      x: right,
      w: small,
      h: 44,
      op: "linear",
      componentId: "gateUp",
    });
    y = gateUpY + 44 + 16;

    push({
      id: "silu",
      label: "SiLU(gate) ⊙ up",
      sub: "element-wise product",
      w: NODE_W + 30,
      h: NODE_H_TALL,
      op: "silu",
      componentId: "silu",
      emphasized: true,
    });
    y += NODE_H_TALL + 16;

    push({
      id: "down",
      label: "Linear · down",
      sub: "→ d_model",
      w: NODE_W,
      h: 44,
      op: "linear",
      componentId: "downProj",
    });
    y += 44 + 16;

    edges.push({ from: "rms2", to: "gate" });
    edges.push({ from: "rms2", to: "up" });
    edges.push({ from: "gate", to: "silu" });
    edges.push({ from: "up", to: "silu" });
    edges.push({ from: "silu", to: "down" });
  }

  push({
    id: "add2",
    label: "add (residual)",
    w: 140,
    h: 40,
    op: "add",
    componentId: "residual2",
  });
  y += 40 + 14;

  const groupBottom = y;
  groups.push({
    id: "decoder",
    x: 24,
    y: groupTop - 12,
    w: CANVAS_W - 48,
    h: groupBottom - groupTop + 4,
    label: "Qwen decoder block",
    color: "#7c5cff",
    multiplier: `× ${numLayers}`,
    multiplierY: (groupTop + groupBottom) / 2,
  });
  y += 26;

  // Final RMSNorm + LM Head + sampling
  push({
    id: "finalRms",
    label: "Final RMSNorm",
    w: NODE_W,
    h: NODE_H,
    op: "rmsnorm",
    componentId: "finalRmsNorm",
  });
  y += NODE_H + 20;

  push({
    id: "lmHead",
    label: "LM Head (Linear)",
    sub: "tied to W_E (small Qwen)",
    w: NODE_W,
    h: NODE_H_TALL,
    op: "linear",
    componentId: "lmHead",
  });
  y += NODE_H_TALL + 20;

  push({
    id: "smx",
    label: "softmax + sample",
    w: NODE_W_SM,
    h: NODE_H,
    op: "softmax",
    componentId: "sampling",
  });
  y += NODE_H + 20;

  push({
    id: "out",
    label: "next token",
    w: NODE_W_SM,
    h: NODE_H,
    op: "io",
    componentId: "sampling",
  });
  y += NODE_H;

  // Sequential edges
  const seq =
    granularity === "block"
      ? [
          ["input", "tok"],
          ["tok", "tokEmb"],
          ["tokEmb", "rms1"],
          ["rms1", "attn"],
          ["attn", "add1"],
          ["add1", "rms2"],
          ["rms2", "swiglu"],
          ["swiglu", "add2"],
          ["add2", "finalRms"],
          ["finalRms", "lmHead"],
          ["lmHead", "smx"],
          ["smx", "out"],
        ]
      : [
          ["input", "tok"],
          ["tok", "tokEmb"],
          ["tokEmb", "rms1"],
          ["attnOut", "add1"],
          ["add1", "rms2"],
          ["down", "add2"],
          ["add2", "finalRms"],
          ["finalRms", "lmHead"],
          ["lmHead", "smx"],
          ["smx", "out"],
        ];
  for (const [a, b] of seq) edges.push({ from: a, to: b });

  // Residual side-arrows
  edges.push({ from: "tokEmb", to: "add1", fromSide: "right", toSide: "right" });
  edges.push({ from: "add1", to: "add2", fromSide: "right", toSide: "right" });

  return { nodes, edges, groups, width: CANVAS_W, height: y + 24 };
}

interface Props {
  granularity: Granularity;
  numLayers: number;
  selected: QwenComponentId;
  onSelect: (id: QwenComponentId) => void;
}

export default function QwenGraphCanvas({
  granularity,
  numLayers,
  selected,
  onSelect,
}: Props) {
  const { nodes, edges, groups, width, height } = useMemo(
    () => buildQwenGraph(granularity, numLayers),
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
          <marker id="qw-arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L7,3 z" fill="#5b6478" />
          </marker>
          <marker id="qw-arrow-res" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
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
              fill={`${g.color}10`}
              stroke={`${g.color}88`}
              strokeWidth={1.5}
              strokeDasharray="5 5"
            />
            <g>
              <rect
                x={g.x + 14}
                y={g.y - 10}
                width={150}
                height={20}
                rx={6}
                fill={g.color}
              />
              <text
                x={g.x + 14 + 75}
                y={g.y + 4}
                textAnchor="middle"
                fontSize={11}
                fontWeight={700}
                fill="#0a0b10"
              >
                {g.label}
              </text>
            </g>
            {g.multiplier && (
              <text
                x={g.x + g.w + 14}
                y={(g.multiplierY ?? g.y + g.h / 2) + 5}
                fill={g.color}
                fontSize={13}
                fontWeight={700}
                fontFamily="JetBrains Mono, monospace"
              >
                {g.multiplier}
              </text>
            )}
          </g>
        ))}

        {/* Edges */}
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
  const isEmph = !!n.emphasized;
  return (
    <g
      transform={`translate(${n.x}, ${n.y})`}
      className="cursor-pointer"
      onClick={onClick}
    >
      {isEmph && !selected && (
        <rect
          x={-3}
          y={-3}
          width={n.w + 6}
          height={n.h + 6}
          rx={12}
          fill="none"
          stroke={`${color}55`}
          strokeWidth={1.2}
          strokeDasharray="2 3"
        />
      )}
      <rect
        width={n.w}
        height={n.h}
        rx={10}
        fill={selected ? `${color}40` : `${color}1f`}
        stroke={selected ? "#ffffff" : `${color}aa`}
        strokeWidth={selected ? 2 : 1.4}
        style={{
          filter: selected
            ? `drop-shadow(0 0 14px ${color}99)`
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
    const ax = a.x + a.w;
    const ay = a.y + a.h / 2;
    const bx = b.x + b.w;
    const by = b.y + b.h / 2;
    const outX = Math.max(ax, bx) + 38;
    return (
      <path
        d={`M ${ax} ${ay} L ${outX} ${ay} L ${outX} ${by} L ${bx} ${by}`}
        stroke="#a48bff"
        strokeWidth={1.4}
        fill="none"
        strokeDasharray="3 4"
        markerEnd="url(#qw-arrow-res)"
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
      markerEnd="url(#qw-arrow)"
    />
  );
}
