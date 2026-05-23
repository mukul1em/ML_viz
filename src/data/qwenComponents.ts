export type QwenComponentId =
  | "input"
  | "tokenization"
  | "tokenEmbedding"
  | "rmsNorm1"
  | "selfAttention"
  | "qkvProj"
  | "rope"
  | "gqa"
  | "scaledDotProduct"
  | "attnOut"
  | "residual1"
  | "rmsNorm2"
  | "swiglu"
  | "gateUp"
  | "silu"
  | "downProj"
  | "residual2"
  | "finalRmsNorm"
  | "lmHead"
  | "sampling";

export interface QwenComponent {
  id: QwenComponentId;
  short: string;
  full: string;
  tagline: string;
  group:
    | "input"
    | "embedding"
    | "block"
    | "attention"
    | "ffn"
    | "output";
  inBlock?: boolean;
  /** True if this is a Qwen-distinctive feature worth highlighting. */
  signature?: boolean;
}

export const qwenComponents: Record<QwenComponentId, QwenComponent> = {
  input: {
    id: "input",
    short: "Input text",
    full: "Input text",
    tagline: "The prompt the model continues.",
    group: "input",
  },
  tokenization: {
    id: "tokenization",
    short: "Tokenizer (BPE)",
    full: "Tokenizer",
    tagline:
      "Byte-level BPE with a 152k vocab — much larger than GPT-2's 50k.",
    group: "input",
  },
  tokenEmbedding: {
    id: "tokenEmbedding",
    short: "Token embedding",
    full: "Token embedding",
    tagline:
      "Each token ID → learned vector. NO positional embedding here — RoPE handles that inside attention.",
    group: "embedding",
  },
  rmsNorm1: {
    id: "rmsNorm1",
    short: "RMSNorm",
    full: "RMSNorm (pre-attention)",
    tagline: "Lightweight LayerNorm replacement — no centering, no bias.",
    group: "block",
    inBlock: true,
    signature: true,
  },
  selfAttention: {
    id: "selfAttention",
    short: "Causal Attention",
    full: "Self-Attention (causal)",
    tagline: "Q · Kᵀ / √d_k → mask → softmax → · V. Each token attends back, not forward.",
    group: "attention",
    inBlock: true,
  },
  qkvProj: {
    id: "qkvProj",
    short: "Q, K, V projections",
    full: "Q / K / V linear projections",
    tagline: "Three linear maps from the residual stream. Q has more heads than K & V (GQA).",
    group: "attention",
    inBlock: true,
  },
  rope: {
    id: "rope",
    short: "RoPE",
    full: "Rotary Position Embeddings",
    tagline:
      "Inject position information by *rotating* pairs of dimensions in Q and K — no learned position vectors.",
    group: "attention",
    inBlock: true,
    signature: true,
  },
  gqa: {
    id: "gqa",
    short: "Grouped-Query Attn",
    full: "Grouped-Query Attention",
    tagline:
      "Multiple Q heads share one K/V head. Cuts the KV cache cost dramatically at inference.",
    group: "attention",
    inBlock: true,
    signature: true,
  },
  scaledDotProduct: {
    id: "scaledDotProduct",
    short: "softmax(QKᵀ/√d_k) · V",
    full: "Scaled dot-product attention",
    tagline: "The classic attention kernel applied per head, with causal mask.",
    group: "attention",
    inBlock: true,
  },
  attnOut: {
    id: "attnOut",
    short: "Linear (out)",
    full: "Attention output projection",
    tagline: "Concatenate heads → linear → back to d_model.",
    group: "attention",
    inBlock: true,
  },
  residual1: {
    id: "residual1",
    short: "+ Residual",
    full: "Residual after attention",
    tagline: "Add the pre-attention activations back in.",
    group: "block",
    inBlock: true,
  },
  rmsNorm2: {
    id: "rmsNorm2",
    short: "RMSNorm",
    full: "RMSNorm (pre-FFN)",
    tagline: "Same RMSNorm again, before the SwiGLU FFN.",
    group: "block",
    inBlock: true,
    signature: true,
  },
  swiglu: {
    id: "swiglu",
    short: "SwiGLU FFN",
    full: "SwiGLU Feed-Forward",
    tagline:
      "Gated FFN: down(  silu(gate(x))  ⊙  up(x)  ). Three linears, not two.",
    group: "ffn",
    inBlock: true,
    signature: true,
  },
  gateUp: {
    id: "gateUp",
    short: "gate, up",
    full: "Gate & Up projections",
    tagline:
      "Two parallel linear maps from d_model → ffn_dim. One becomes the gate, the other the value.",
    group: "ffn",
    inBlock: true,
  },
  silu: {
    id: "silu",
    short: "SiLU · ⊙",
    full: "SiLU + element-wise multiply",
    tagline:
      "Apply SiLU(x) = x · σ(x) to the gate, then element-wise multiply with `up`.",
    group: "ffn",
    inBlock: true,
  },
  downProj: {
    id: "downProj",
    short: "Linear (down)",
    full: "Down projection",
    tagline: "Linear map ffn_dim → d_model. No bias.",
    group: "ffn",
    inBlock: true,
  },
  residual2: {
    id: "residual2",
    short: "+ Residual",
    full: "Residual after FFN",
    tagline: "Skip connection across the FFN.",
    group: "block",
    inBlock: true,
  },
  finalRmsNorm: {
    id: "finalRmsNorm",
    short: "Final RMSNorm",
    full: "Final RMSNorm",
    tagline: "Normalize the residual stream one last time.",
    group: "output",
  },
  lmHead: {
    id: "lmHead",
    short: "LM head",
    full: "Language-Modeling head",
    tagline:
      "Linear projection to the vocabulary. Often *tied* to the input embedding matrix.",
    group: "output",
  },
  sampling: {
    id: "sampling",
    short: "softmax + sample",
    full: "Softmax + sampling",
    tagline: "Logits → probs → next token.",
    group: "output",
  },
};
