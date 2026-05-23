import { Dices } from "lucide-react";
import SamplingViz from "../components/SamplingViz";

export default function SamplingPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Decoding</span>
          <span className="chip">Inference-time</span>
          <span className="chip">LLM</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-violet-500 to-cyan-400 items-center justify-center shadow-glow">
            <Dices className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              p(token | context)
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              The decoding toolbox —{" "}
              <span className="text-violet-300">temperature</span>,{" "}
              <span className="text-cyan-300">top-k</span>,{" "}
              <span className="text-cyan-300">top-p</span>,{" "}
              <span className="text-cyan-300">min-p</span>,{" "}
              <span className="text-amber-300">repetition penalty</span>, and{" "}
              <span className="text-emerald-300">beam search</span> — that turns
              logits into the actual token a chat model emits.
            </p>
          </div>
        </div>
      </header>

      <SamplingViz />
    </div>
  );
}
