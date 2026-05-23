import type { QwenComponentId, QwenComponent } from "./qwenComponents";

// LLaMA uses the same set of architectural components as Qwen (RMSNorm, RoPE,
// GQA, SwiGLU, etc.), so we reuse the typed key union.
export type LlamaComponentId = QwenComponentId;
export type LlamaComponent = QwenComponent;

export const llamaComponents: Record<LlamaComponentId, LlamaComponent> = {
  input: {
    id: "input",
    short: "Input text",
    full: "Input text",
    tagline: "The prompt the model continues — open-weights, runnable anywhere.",
    group: "input",
  },
  tokenization: {
    id: "tokenization",
    short: "Tokenizer (tiktoken BPE)",
    full: "Tokenizer",
    tagline:
      "LLaMA-3 ships a tiktoken-style BPE with a 128k vocab; LLaMA-1 / 2 used SentencePiece 32k.",
    group: "input",
  },
  tokenEmbedding: {
    id: "tokenEmbedding",
    short: "Token embedding",
    full: "Token embedding",
    tagline:
      "Each token ID → learned vector. NO positional embedding here — RoPE handles position inside attention.",
    group: "embedding",
  },
  rmsNorm1: {
    id: "rmsNorm1",
    short: "RMSNorm",
    full: "RMSNorm (pre-attention)",
    tagline:
      "Lightweight LayerNorm — divide by RMS, no centering, no bias. LLaMA mainstreamed it.",
    group: "block",
    inBlock: true,
    signature: true,
  },
  selfAttention: {
    id: "selfAttention",
    short: "Causal Attention",
    full: "Self-Attention (causal)",
    tagline:
      "Q · Kᵀ / √d_k → causal mask → softmax → · V. No bias on any linear projection.",
    group: "attention",
    inBlock: true,
  },
  qkvProj: {
    id: "qkvProj",
    short: "Q, K, V projections",
    full: "Q / K / V linear projections",
    tagline:
      "Three bias-free linear maps from the residual stream. Q has more heads than K & V (GQA).",
    group: "attention",
    inBlock: true,
  },
  rope: {
    id: "rope",
    short: "RoPE (θ=500k)",
    full: "Rotary Position Embeddings",
    tagline:
      "Rotate pairs of Q/K dimensions by a per-position angle. LLaMA-3.1 bumps θ to 500k for 128k context.",
    group: "attention",
    inBlock: true,
    signature: true,
  },
  gqa: {
    id: "gqa",
    short: "Grouped-Query Attn",
    full: "Grouped-Query Attention",
    tagline:
      "First widely-deployed in LLaMA-2-70B. Multiple Q heads share one K/V head ⇒ 4–8× smaller KV cache.",
    group: "attention",
    inBlock: true,
    signature: true,
  },
  scaledDotProduct: {
    id: "scaledDotProduct",
    short: "softmax(QKᵀ/√d_k) · V",
    full: "Scaled dot-product attention",
    tagline: "The classical attention kernel per head, with causal mask.",
    group: "attention",
    inBlock: true,
  },
  attnOut: {
    id: "attnOut",
    short: "Linear (out)",
    full: "Attention output projection",
    tagline: "Concatenate heads → bias-free linear → back to d_model.",
    group: "attention",
    inBlock: true,
  },
  residual1: {
    id: "residual1",
    short: "+ Residual",
    full: "Residual after attention",
    tagline: "Add the pre-attention activations back in (pre-norm Transformer).",
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
      "Gated FFN — down(  SiLU(gate(x))  ⊙  up(x)  ). Three linears, no bias, no GELU.",
    group: "ffn",
    inBlock: true,
    signature: true,
  },
  gateUp: {
    id: "gateUp",
    short: "gate, up",
    full: "Gate & Up projections",
    tagline:
      "Two parallel bias-free linears d_model → ffn_dim. One becomes the gate, one the value.",
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
    tagline: "Bias-free linear map ffn_dim → d_model.",
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
    tagline: "Normalize the residual stream one last time before the LM head.",
    group: "output",
  },
  lmHead: {
    id: "lmHead",
    short: "LM head",
    full: "Language-Modeling head",
    tagline:
      "Linear projection to the 128k vocab. Tied to the input embedding on 1B / 3B; untied on 8B and above.",
    group: "output",
  },
  sampling: {
    id: "sampling",
    short: "softmax + sample",
    full: "Softmax + sampling",
    tagline: "Logits → probs → next token (temperature, top-p, top-k as usual).",
    group: "output",
  },
};
