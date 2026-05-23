export interface EmbeddingModelSpec {
  name: string;
  family: string;
  dims: number;
  dimsAlt?: number;
  ctxTokens: number;
  trainObjective: string;
  pooling: "CLS" | "mean" | "last-token" | "weighted";
  openSource: boolean;
  strength: string;
  releasedYear: number;
  /** Brand accent — used for the gradient on the card */
  accent: string;
}

export const embeddingModels: EmbeddingModelSpec[] = [
  {
    name: "text-embedding-3-large",
    family: "OpenAI",
    dims: 3072,
    dimsAlt: 256,
    ctxTokens: 8191,
    trainObjective: "Contrastive (text pairs, large-scale)",
    pooling: "weighted",
    openSource: false,
    strength: "Top-tier MTEB scores, Matryoshka — truncate dims for storage.",
    releasedYear: 2024,
    accent: "#10b981",
  },
  {
    name: "text-embedding-3-small",
    family: "OpenAI",
    dims: 1536,
    dimsAlt: 256,
    ctxTokens: 8191,
    trainObjective: "Contrastive (text pairs)",
    pooling: "weighted",
    openSource: false,
    strength: "Cheap, fast, still excellent. Default for most RAG stacks.",
    releasedYear: 2024,
    accent: "#34d399",
  },
  {
    name: "bge-large-en-v1.5",
    family: "BAAI",
    dims: 1024,
    ctxTokens: 512,
    trainObjective: "Multi-stage contrastive + retrieval finetune",
    pooling: "CLS",
    openSource: true,
    strength: "Strong open-source baseline; consistently in top MTEB ranks.",
    releasedYear: 2023,
    accent: "#22d3ee",
  },
  {
    name: "e5-large-v2",
    family: "Microsoft",
    dims: 1024,
    ctxTokens: 512,
    trainObjective: "Weakly-supervised contrastive (text pairs from web)",
    pooling: "mean",
    openSource: true,
    strength: "Trained at scale on noisy pairs; great asymmetric retrieval.",
    releasedYear: 2022,
    accent: "#38bdf8",
  },
  {
    name: "all-MiniLM-L6-v2",
    family: "Sentence-BERT",
    dims: 384,
    ctxTokens: 256,
    trainObjective: "Knowledge distillation + NLI/STS contrastive",
    pooling: "mean",
    openSource: true,
    strength: "Tiny, fast, runs on CPU. The go-to baseline since 2021.",
    releasedYear: 2021,
    accent: "#a78bfa",
  },
  {
    name: "nomic-embed-text-v1.5",
    family: "Nomic",
    dims: 768,
    dimsAlt: 64,
    ctxTokens: 8192,
    trainObjective: "Multi-stage contrastive (open data + open weights)",
    pooling: "mean",
    openSource: true,
    strength: "Fully open: data, code, weights. Matryoshka + long context.",
    releasedYear: 2024,
    accent: "#f472b6",
  },
];
