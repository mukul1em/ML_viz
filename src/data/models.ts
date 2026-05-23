import {
  BookOpen,
  Bot,
  Compass,
  Cpu,
  Feather,
  Layers,
  Sparkles,
  type LucideIcon,
} from "lucide-react";

export type ModelFamily =
  | "decoder"
  | "encoder"
  | "encoder-decoder"
  | "embedding";

export interface ModelEntry {
  slug: string;
  name: string;
  family: ModelFamily;
  blurb: string;
  icon: LucideIcon;
  available: boolean;
  path?: string;
  accent: string;
  tagline?: string;
}

export const familyLabel: Record<ModelFamily, string> = {
  decoder: "Decoder-only (causal LM)",
  encoder: "Encoder-only",
  "encoder-decoder": "Encoder–Decoder",
  embedding: "Embedding Models",
};

export const models: ModelEntry[] = [
  {
    slug: "gpt",
    name: "GPT",
    family: "decoder",
    blurb:
      "Decoder-only transformer: tokenize → embed → N transformer blocks → LM head → next-token distribution. Click any component to dive in.",
    icon: Bot,
    available: true,
    path: "/llm/gpt",
    accent: "from-violet-500/30 to-cyan-400/20",
    tagline: "Generative Pre-trained Transformer",
  },
  {
    slug: "bert",
    name: "BERT",
    family: "encoder",
    blurb:
      "Bidirectional encoder. Masked language modeling + next-sentence prediction.",
    icon: BookOpen,
    available: true,
    path: "/llm/bert",
    accent: "from-emerald-400/30 to-cyan-400/20",
    tagline: "Bidirectional Encoder Representations",
  },
  {
    slug: "embeddings",
    name: "Embedding Models",
    family: "embedding",
    blurb:
      "Text → vector. Sentence-BERT, BGE, E5, OpenAI text-embedding-3. Language becomes geometry — and that geometry powers semantic search, RAG, and clustering.",
    icon: Compass,
    available: true,
    path: "/llm/embeddings",
    accent: "from-cyan-400/30 to-fuchsia-400/20",
    tagline: "Language as vectors",
  },
  {
    slug: "qwen",
    name: "Qwen",
    family: "decoder",
    blurb:
      "Alibaba's modern decoder-only family. RMSNorm, RoPE, Grouped-Query Attention, SwiGLU — every notable post-GPT trick in one stack.",
    icon: Feather,
    available: true,
    path: "/llm/qwen",
    accent: "from-violet-500/30 to-pink-500/20",
    tagline: "Decoder-only · RoPE · GQA · SwiGLU",
  },
  {
    slug: "llama",
    name: "LLaMA",
    family: "decoder",
    blurb:
      "Meta's open-source decoder-only family. The reference recipe — RMSNorm, RoPE, GQA, SwiGLU, no bias on linears — that the entire modern LLM ecosystem now copies.",
    icon: Cpu,
    available: true,
    path: "/llm/llama",
    accent: "from-orange-400/30 to-pink-500/20",
    tagline: "Decoder-only · open weights · no-bias linears",
  },
  {
    slug: "t5",
    name: "T5",
    family: "encoder-decoder",
    blurb: "Text-to-text encoder–decoder with relative position biases.",
    icon: Layers,
    available: false,
    accent: "from-amber-400/30 to-rose-500/20",
    tagline: "Text-to-Text Transfer Transformer",
  },
  {
    slug: "mixtral",
    name: "Mixture-of-Experts",
    family: "decoder",
    blurb: "Sparse routing across expert FFNs (Mixtral-style).",
    icon: Sparkles,
    available: false,
    accent: "from-indigo-400/30 to-violet-500/20",
    tagline: "Sparse expert routing",
  },
];
