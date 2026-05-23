import {
  Activity,
  Boxes,
  Compass,
  Database,
  Dices,
  GitBranch,
  KeyRound,
  Layers,
  Ruler,
  Scale,
  Shrink,
  Sigma,
  Spline,
  TrendingDown,
  Type,
  Waves,
  Workflow,
  Zap,
  type LucideIcon,
} from "lucide-react";

export type FunctionCategory =
  | "activation"
  | "attention"
  | "loss"
  | "optimizer"
  | "normalization"
  | "decoding"
  | "fine-tuning";

export interface FunctionEntry {
  slug: string;
  name: string;
  category: FunctionCategory;
  blurb: string;
  icon: LucideIcon;
  available: boolean;
  path?: string;
  accent: string;
}

export const categoryLabel: Record<FunctionCategory, string> = {
  activation: "Activation",
  attention: "Attention",
  loss: "Loss",
  optimizer: "Optimizer",
  normalization: "Normalization",
  decoding: "Decoding",
  "fine-tuning": "Fine-tuning",
};

export const functions: FunctionEntry[] = [
  {
    slug: "softmax",
    name: "Softmax",
    category: "activation",
    blurb:
      "Turn a vector of logits into a probability distribution. Explore temperature, sharpness, and stability.",
    icon: Sigma,
    available: true,
    path: "/softmax",
    accent: "from-violet-500/30 to-cyan-400/20",
  },
  {
    slug: "qkv",
    name: "Q, K, V Attention",
    category: "attention",
    blurb:
      "The heart of every transformer. Watch how Query, Key, and Value cooperate to route information across tokens.",
    icon: KeyRound,
    available: true,
    path: "/qkv",
    accent: "from-pink-500/30 to-cyan-400/20",
  },
  {
    slug: "sigmoid",
    name: "Sigmoid",
    category: "activation",
    blurb: "σ(x) = 1 / (1 + e⁻ˣ). Range (0, 1). σ′ = σ(1−σ).",
    icon: Spline,
    available: true,
    path: "/sigmoid",
    accent: "from-pink-500/30 to-violet-500/20",
  },
  {
    slug: "relu",
    name: "ReLU & friends",
    category: "activation",
    blurb: "f(x) = max(0, x). Sparse, piecewise-linear. Plus Leaky · ELU · GELU · SiLU · Mish.",
    icon: Zap,
    available: true,
    path: "/relu",
    accent: "from-amber-400/30 to-rose-500/20",
  },
  {
    slug: "tanh",
    name: "Tanh",
    category: "activation",
    blurb: "tanh(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ). Zero-centered, range (−1, 1). tanh′ = 1 − tanh².",
    icon: Waves,
    available: true,
    path: "/tanh",
    accent: "from-cyan-400/30 to-teal-400/20",
  },
  {
    slug: "cross-entropy",
    name: "Cross-Entropy",
    category: "loss",
    blurb: "H(p, q) = −Σ p log q. NLL of categorical / Bernoulli. ∇ = q − y when paired with softmax/σ.",
    icon: TrendingDown,
    available: true,
    path: "/cross-entropy",
    accent: "from-rose-500/30 to-amber-400/20",
  },
  {
    slug: "mse",
    name: "MSE / L2 Loss",
    category: "loss",
    blurb:
      "L = (ŷ − y)². Gaussian-MLE regression. Strong gradient pull and famous outlier sensitivity.",
    icon: Ruler,
    available: true,
    path: "/mse",
    accent: "from-rose-500/30 to-violet-500/20",
  },
  {
    slug: "loss-family",
    name: "Losses & friends",
    category: "loss",
    blurb:
      "MAE · Huber · Log-cosh · Hinge · Focal — the robust-loss toolbox for regression, SVMs, and detection.",
    icon: Layers,
    available: true,
    path: "/loss-family",
    accent: "from-cyan-400/30 to-rose-500/20",
  },
  {
    slug: "kl-divergence",
    name: "KL Divergence",
    category: "loss",
    blurb:
      "D_KL(p ‖ q) = Σ p log(p/q). Forward vs reverse, JS divergence, the VAE / distillation / RLHF workhorse.",
    icon: Scale,
    available: true,
    path: "/kl-divergence",
    accent: "from-violet-500/30 to-amber-400/20",
  },
  {
    slug: "gradient-descent",
    name: "Gradient Descent",
    category: "optimizer",
    blurb: "θ ← θ − η∇L(θ). SGD · Momentum · Nesterov · AdaGrad · RMSProp · Adam · AdamW · Lion on a live loss landscape.",
    icon: GitBranch,
    available: true,
    path: "/gradient-descent",
    accent: "from-emerald-400/30 to-cyan-400/20",
  },
  {
    slug: "batchnorm",
    name: "Batch Norm",
    category: "normalization",
    blurb: "x̂ = (x − μ_B)/√(σ²_B + ε), y = γ x̂ + β. Plus LayerNorm · InstanceNorm · GroupNorm · RMSNorm.",
    icon: Activity,
    available: true,
    path: "/batchnorm",
    accent: "from-indigo-400/30 to-violet-500/20",
  },
  {
    slug: "positional-encodings",
    name: "Positional Encodings",
    category: "attention",
    blurb:
      "Sinusoidal · Learned · RoPE · ALiBi side-by-side. Length extrapolation, rotation, and bias visualized.",
    icon: Compass,
    available: true,
    path: "/positional-encodings",
    accent: "from-cyan-400/30 to-violet-500/20",
  },
  {
    slug: "attention-variants",
    name: "Attention Variants",
    category: "attention",
    blurb:
      "MHA → MQA → GQA → MLA. Heads, KV-cache memory, and the tradeoffs that power LLaMA · Mistral · DeepSeek.",
    icon: Workflow,
    available: true,
    path: "/attention-variants",
    accent: "from-pink-500/30 to-cyan-400/20",
  },
  {
    slug: "moe",
    name: "Mixture of Experts",
    category: "attention",
    blurb:
      "Router + top-k gating across N expert FFNs. Active vs total parameters, load balancing, Mixtral · DeepSeek · Qwen-MoE.",
    icon: Boxes,
    available: true,
    path: "/moe",
    accent: "from-amber-400/30 to-violet-500/20",
  },
  {
    slug: "sampling",
    name: "Sampling Strategies",
    category: "decoding",
    blurb:
      "Temperature · top-k · top-p · min-p · repetition penalty · beam search. The knobs every chat UI exposes.",
    icon: Dices,
    available: true,
    path: "/sampling",
    accent: "from-violet-500/30 to-cyan-400/20",
  },
  {
    slug: "kv-cache",
    name: "KV Cache",
    category: "decoding",
    blurb:
      "Why autoregressive inference is linear, not quadratic. K/V grow one row per token; paged attention manages it.",
    icon: Database,
    available: true,
    path: "/kv-cache",
    accent: "from-cyan-400/30 to-emerald-400/20",
  },
  {
    slug: "bpe",
    name: "BPE Tokenization",
    category: "decoding",
    blurb:
      "Byte-pair encoding: greedy merges that turn a corpus into a compact vocabulary. The pipeline behind tiktoken & SentencePiece.",
    icon: Type,
    available: true,
    path: "/bpe",
    accent: "from-emerald-400/30 to-cyan-400/20",
  },
  {
    slug: "lora",
    name: "LoRA",
    category: "fine-tuning",
    blurb:
      "W + BA with rank r ≪ min(d, k). Massive parameter savings, swappable adapters, the dominant LLM fine-tune.",
    icon: Shrink,
    available: true,
    path: "/lora",
    accent: "from-pink-500/30 to-violet-500/20",
  },
];
