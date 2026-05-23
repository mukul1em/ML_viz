export type BertComponentId =
  | "input"
  | "tokenization"
  | "tokenEmbedding"
  | "positionalEmbedding"
  | "segmentEmbedding"
  | "embeddingSum"
  | "embedLayerNorm"
  | "selfAttention"
  | "multiHead"
  | "residual1"
  | "layerNorm1"
  | "ffn"
  | "residual2"
  | "layerNorm2"
  | "pooler"
  | "nspHead"
  | "mlmHead";

export interface BertComponent {
  id: BertComponentId;
  short: string;
  full: string;
  tagline: string;
  group: "input" | "embedding" | "block" | "attention" | "ffn" | "head";
}

export const bertComponents: Record<BertComponentId, BertComponent> = {
  input: {
    id: "input",
    short: "Input pair",
    full: "Input text + [CLS]/[SEP]",
    tagline: "A sentence or sentence pair with special tokens.",
    group: "input",
  },
  tokenization: {
    id: "tokenization",
    short: "WordPiece",
    full: "WordPiece tokenization",
    tagline: "Subword tokens with ## prefix for continuations.",
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
    short: "Position embedding",
    full: "Positional embedding (learned)",
    tagline: "Learned position embeddings, not sinusoidal.",
    group: "embedding",
  },
  segmentEmbedding: {
    id: "segmentEmbedding",
    short: "Segment embedding",
    full: "Segment embedding (A/B)",
    tagline: "Identifies which sentence each token belongs to.",
    group: "embedding",
  },
  embeddingSum: {
    id: "embeddingSum",
    short: "+ Sum (3 embeddings)",
    full: "Token + Position + Segment",
    tagline: "Element-wise sum of all three embeddings.",
    group: "embedding",
  },
  embedLayerNorm: {
    id: "embedLayerNorm",
    short: "LayerNorm + Dropout",
    full: "Embedding LayerNorm + Dropout",
    tagline: "Normalize and regularize before encoder.",
    group: "embedding",
  },
  selfAttention: {
    id: "selfAttention",
    short: "Bidirectional Attention",
    full: "Bidirectional Self-Attention",
    tagline: "Every token attends to every other token — no mask.",
    group: "attention",
  },
  multiHead: {
    id: "multiHead",
    short: "Multi-Head split",
    full: "Multi-Head Attention",
    tagline: "Parallel attention heads.",
    group: "attention",
  },
  residual1: {
    id: "residual1",
    short: "+ Residual",
    full: "Residual connection (post-attention)",
    tagline: "Add input back in.",
    group: "block",
  },
  layerNorm1: {
    id: "layerNorm1",
    short: "LayerNorm (post-norm)",
    full: "LayerNorm — post-norm",
    tagline: "Applied AFTER the residual add.",
    group: "block",
  },
  ffn: {
    id: "ffn",
    short: "Feed-Forward (GELU)",
    full: "Feed-Forward Network",
    tagline: "4× expansion, GELU activation.",
    group: "ffn",
  },
  residual2: {
    id: "residual2",
    short: "+ Residual",
    full: "Residual connection (post-FFN)",
    tagline: "Add input back in.",
    group: "block",
  },
  layerNorm2: {
    id: "layerNorm2",
    short: "LayerNorm (post-norm)",
    full: "LayerNorm — post-norm",
    tagline: "Applied AFTER the FFN add.",
    group: "block",
  },
  pooler: {
    id: "pooler",
    short: "Pooler",
    full: "[CLS] Pooler (tanh)",
    tagline: "Project [CLS] vector for sequence-level tasks.",
    group: "head",
  },
  nspHead: {
    id: "nspHead",
    short: "NSP head",
    full: "Next-Sentence Prediction head",
    tagline: "Binary classifier from pooled [CLS].",
    group: "head",
  },
  mlmHead: {
    id: "mlmHead",
    short: "MLM head",
    full: "Masked Language Modeling head",
    tagline: "Predict masked tokens from final hidden states.",
    group: "head",
  },
};
