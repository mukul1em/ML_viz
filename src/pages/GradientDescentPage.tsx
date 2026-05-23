import { GitBranch } from "lucide-react";
import GradientDescentViz from "../components/GradientDescentViz";

export default function GradientDescentPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Optimizer</span>
          <span className="chip">First-order</span>
          <span className="chip">Iterative</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-400 to-cyan-400 items-center justify-center shadow-glow">
            <GitBranch className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              θ ← θ − η ∇L
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              Gradient descent and its descendants —{" "}
              <span className="text-violet-300">SGD</span> ·{" "}
              <span className="text-cyan-300">Momentum</span> ·{" "}
              <span className="text-emerald-300">Nesterov</span> ·{" "}
              <span className="text-pink-300">AdaGrad</span> ·{" "}
              <span className="text-amber-300">RMSProp</span> ·{" "}
              <span className="text-indigo-300">Adam</span> — racing on the same
              loss surface.
            </p>
          </div>
        </div>
      </header>

      <GradientDescentViz />
    </div>
  );
}
