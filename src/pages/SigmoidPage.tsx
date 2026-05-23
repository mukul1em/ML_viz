import { Spline } from "lucide-react";
import SigmoidViz from "../components/SigmoidViz";

export default function SigmoidPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Activation</span>
          <span className="chip">Binary</span>
          <span className="chip">Smooth</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-pink-500 to-violet-500 items-center justify-center shadow-glow">
            <Spline className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              σ(x)
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              The logistic function. <span className="text-violet-300">ℝ → (0, 1)</span>,
              monotonic, smooth, with the elegant identity{" "}
              <span className="text-cyan-300">σ′ = σ(1−σ)</span>.
            </p>
          </div>
        </div>
      </header>

      <SigmoidViz />
    </div>
  );
}
