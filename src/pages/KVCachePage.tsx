import { Database } from "lucide-react";
import KVCacheViz from "../components/KVCacheViz";

export default function KVCachePage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Decoding</span>
          <span className="chip">Inference</span>
          <span className="chip">LLM</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan-400 to-emerald-400 items-center justify-center shadow-glow">
            <Database className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              o_t = softmax(q_t K_{"≤"}t^T) V_{"≤"}t
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              Why autoregressive inference is{" "}
              <span className="text-emerald-300">linear</span>, not quadratic.
              K and V grow one row per token; the prefix is cached. Paged
              attention then turns the GPU into a slot-allocator.
            </p>
          </div>
        </div>
      </header>

      <KVCacheViz />
    </div>
  );
}
