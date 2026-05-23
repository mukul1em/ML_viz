import { Waves } from "lucide-react";
import TanhViz from "../components/TanhViz";

export default function TanhPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Activation</span>
          <span className="chip">Zero-centered</span>
          <span className="chip">RNN</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan-400 to-teal-400 items-center justify-center shadow-glow">
            <Waves className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              tanh(x)
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              The hyperbolic tangent. <span className="text-cyan-300">ℝ → (−1, 1)</span>,
              zero-centered, with peak slope{" "}
              <span className="text-emerald-300">tanh′(0) = 1</span>{" "}
              — 4× the sigmoid&apos;s.
            </p>
          </div>
        </div>
      </header>

      <TanhViz />
    </div>
  );
}
