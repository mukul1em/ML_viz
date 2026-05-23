import { Zap } from "lucide-react";
import ReLUViz from "../components/ReLUViz";

export default function ReLUPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Activation</span>
          <span className="chip">Hidden-layer</span>
          <span className="chip">Piecewise-linear</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-amber-400 to-rose-500 items-center justify-center shadow-glow">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              ReLU & friends
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              <span className="text-violet-300">max(0, x)</span> — and its smoothed,
              leaky, and gated descendants:{" "}
              <span className="text-cyan-300">Leaky</span>,{" "}
              <span className="text-emerald-300">ELU</span>,{" "}
              <span className="text-pink-300">GELU</span>,{" "}
              <span className="text-amber-300">SiLU</span>,{" "}
              <span className="text-indigo-300">Mish</span>,{" "}
              <span className="text-rose-300">Softplus</span>.
            </p>
          </div>
        </div>
      </header>

      <ReLUViz />
    </div>
  );
}
