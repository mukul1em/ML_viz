import { Boxes } from "lucide-react";
import MoEViz from "../components/MoEViz";

export default function MoEPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Attention</span>
          <span className="chip">Sparse</span>
          <span className="chip">LLM</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-amber-400 to-violet-500 items-center justify-center shadow-glow">
            <Boxes className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              y = Σ_{"e ∈ topₖ(g)"} g̃_e E_e(x)
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              <span className="text-violet-300">Sparse experts</span>: a router
              picks the top-k FFNs per token, so total parameters scale to{" "}
              <span className="text-cyan-300">trillions</span> while per-token
              compute stays at the cost of a single dense FFN —{" "}
              <span className="text-pink-300">Mixtral</span>,{" "}
              <span className="text-pink-300">DeepSeek-V3</span>,{" "}
              <span className="text-pink-300">Switch Transformer</span>.
            </p>
          </div>
        </div>
      </header>

      <MoEViz />
    </div>
  );
}
