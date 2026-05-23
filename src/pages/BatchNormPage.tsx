import { Activity } from "lucide-react";
import BatchNormViz from "../components/BatchNormViz";

export default function BatchNormPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Normalization</span>
          <span className="chip">Per-channel</span>
          <span className="chip">Train ≠ Eval</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-indigo-400 to-violet-500 items-center justify-center shadow-glow">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              y = γ x̂ + β
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              <span className="text-indigo-300">Batch Normalization</span> and its
              cousins —{" "}
              <span className="text-violet-300">LayerNorm</span> ·{" "}
              <span className="text-emerald-300">GroupNorm</span> ·{" "}
              <span className="text-pink-300">InstanceNorm</span> ·{" "}
              <span className="text-amber-300">RMSNorm</span>. Same recipe, different
              axis of aggregation.
            </p>
          </div>
        </div>
      </header>

      <BatchNormViz />
    </div>
  );
}
