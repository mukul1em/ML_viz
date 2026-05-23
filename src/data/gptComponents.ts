export type GPTComponentId =
  | "input"
  | "tokenization"
  | "tokenEmbedding"
  | "positionalEmbedding"
  | "embeddingSum"
  | "layerNorm1"
  | "selfAttention"
  | "multiHead"
  | "residual1"
  | "layerNorm2"
  | "ffn"
  | "residual2"
  | "finalLayerNorm"
  | "lmHead"
  | "sampling";

export interface GPTComponent {
  id: GPTComponentId;
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
}

export const gptComponents: Record<GPTComponentId, GPTComponent> = {
  input: {
    id: "input",
    short: "Input text",
    full: "Input text",
    tagline: "The prompt to continue.",
    group: "input",
  },
  tokenization: {
    id: "tokenization",
    short: "Tokenizer (BPE)",
    full: "Tokenization",
    tagline: "Text → subword token IDs.",
    group: "input",
  },
  tokenEmbedding: {
    id: "tokenEmbedding",
    short: "Token embedding",
    full: "Token embedding",
    tagline: "Each token ID → learned vector.",
    group: "embedding",
  },
  positionalEmbedding: {
    id: "positionalEmbedding",
    short: "Positional embedding",
    full: "Positional embedding",
    tagline: "Inject position information.",
    group: "embedding",
  },
  embeddingSum: {
    id: "embeddingSum",
    short: "Embedding sum",
    full: "Token + positional embedding",
    tagline: "Element-wise addition.",
    group: "embedding",
  },
  layerNorm1: {
    id: "layerNorm1",
    short: "LayerNorm",
    full: "Layer Normalization (pre-attention)",
    tagline: "Normalize features per token.",
    group: "block",
    inBlock: true,
  },
  selfAttention: {
    id: "selfAttention",
    short: "Causal Self-Attention",
    full: "Masked Self-Attention",
    tagline: "Each token attends to past tokens.",
    group: "attention",
    inBlock: true,
  },
  multiHead: {
    id: "multiHead",
    short: "Multi-Head split",
    full: "Multi-Head Attention",
    tagline: "Parallel attention heads.",
    group: "attention",
    inBlock: true,
  },
  residual1: {
    id: "residual1",
    short: "+ Residual",
    full: "Residual connection (after attention)",
    tagline: "Add the input back in.",
    group: "block",
    inBlock: true,
  },
  layerNorm2: {
    id: "layerNorm2",
    short: "LayerNorm",
    full: "Layer Normalization (pre-FFN)",
    tagline: "Normalize before MLP.",
    group: "block",
    inBlock: true,
  },
  ffn: {
    id: "ffn",
    short: "Feed-Forward (MLP)",
    full: "Feed-Forward Network",
    tagline: "Per-token MLP, 4× hidden expansion.",
    group: "ffn",
    inBlock: true,
  },
  residual2: {
    id: "residual2",
    short: "+ Residual",
    full: "Residual connection (after FFN)",
    tagline: "Add input back in.",
    group: "block",
    inBlock: true,
  },
  finalLayerNorm: {
    id: "finalLayerNorm",
    short: "Final LayerNorm",
    full: "Final LayerNorm",
    tagline: "Normalize the final hidden states.",
    group: "output",
  },
  lmHead: {
    id: "lmHead",
    short: "LM head",
    full: "Language Modeling head",
    tagline: "Project hidden → vocabulary logits.",
    group: "output",
  },
  sampling: {
    id: "sampling",
    short: "Softmax + sampling",
    full: "Softmax + next-token sampling",
    tagline: "Logits → probabilities → next token.",
    group: "output",
  },
};

export const blockComponentOrder: GPTComponentId[] = [
  "layerNorm1",
  "selfAttention",
  "multiHead",
  "residual1",
  "layerNorm2",
  "ffn",
  "residual2",
];
