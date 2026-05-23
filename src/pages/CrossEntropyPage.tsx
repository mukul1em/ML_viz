import { TrendingDown } from "lucide-react";
import CrossEntropyViz from "../components/CrossEntropyViz";

export default function CrossEntropyPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Loss</span>
          <span className="chip">NLL</span>
          <span className="chip">Information-theoretic</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-rose-500 to-amber-400 items-center justify-center shadow-glow">
            <TrendingDown className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              H(p, q)
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              Cross-entropy. The <span className="text-rose-300">negative log-likelihood</span> of
              categorical or Bernoulli predictions —{" "}
              <span className="text-violet-300">H(p) + D<sub>KL</sub>(p ‖ q)</span> — and the
              gradient companion of softmax / sigmoid.
            </p>
          </div>
        </div>
      </header>

      <CrossEntropyViz />
    </div>
  );
}
