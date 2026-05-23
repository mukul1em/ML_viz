import {
  Activity,
  GitBranch,
  KeyRound,
  Sigma,
  Spline,
  TrendingDown,
  Waves,
  Zap,
  type LucideIcon,
} from "lucide-react";

export type FunctionCategory =
  | "activation"
  | "attention"
  | "loss"
  | "optimizer"
  | "normalization";

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
    blurb: "Squash any real number into (0, 1). Classic logistic activation.",
    icon: Spline,
    available: false,
    accent: "from-pink-500/30 to-violet-500/20",
  },
  {
    slug: "relu",
    name: "ReLU & friends",
    category: "activation",
    blurb: "ReLU, Leaky ReLU, GELU, SiLU side-by-side.",
    icon: Zap,
    available: false,
    accent: "from-amber-400/30 to-rose-500/20",
  },
  {
    slug: "tanh",
    name: "Tanh",
    category: "activation",
    blurb: "Zero-centered sibling of sigmoid.",
    icon: Waves,
    available: false,
    accent: "from-cyan-400/30 to-teal-400/20",
  },
  {
    slug: "cross-entropy",
    name: "Cross-Entropy",
    category: "loss",
    blurb: "The natural partner of softmax. See how loss reacts to confidence.",
    icon: TrendingDown,
    available: false,
    accent: "from-rose-500/30 to-amber-400/20",
  },
  {
    slug: "gradient-descent",
    name: "Gradient Descent",
    category: "optimizer",
    blurb: "Watch SGD, Momentum, and Adam descend a loss landscape.",
    icon: GitBranch,
    available: false,
    accent: "from-emerald-400/30 to-cyan-400/20",
  },
  {
    slug: "batchnorm",
    name: "Batch Norm",
    category: "normalization",
    blurb: "Visualize how batch normalization reshapes activations.",
    icon: Activity,
    available: false,
    accent: "from-indigo-400/30 to-violet-500/20",
  },
];
