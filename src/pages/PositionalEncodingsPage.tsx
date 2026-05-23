import { Compass } from "lucide-react";
import PositionalEncodingsViz from "../components/PositionalEncodingsViz";

export default function PositionalEncodingsPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Attention</span>
          <span className="chip">Position-aware</span>
          <span className="chip">LLM</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan-400 to-violet-500 items-center justify-center shadow-glow">
            <Compass className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              PE(pos)
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              How transformers learn that order matters —{" "}
              <span className="text-cyan-300">sinusoidal</span> ·{" "}
              <span className="text-violet-300">learned</span> ·{" "}
              <span className="text-pink-300">RoPE</span> ·{" "}
              <span className="text-amber-300">ALiBi</span> side-by-side, with
              length-extrapolation behavior visible at a glance.
            </p>
          </div>
        </div>
      </header>

      <PositionalEncodingsViz />
    </div>
  );
}
