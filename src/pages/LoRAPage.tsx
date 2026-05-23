import { Shrink } from "lucide-react";
import LoRAViz from "../components/LoRAViz";

export default function LoRAPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Fine-tuning</span>
          <span className="chip">PEFT</span>
          <span className="chip">LLM</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-pink-500 to-violet-500 items-center justify-center shadow-glow">
            <Shrink className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              W' = W + (α/r) · BA
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              Freeze the base. Learn a tiny <span className="text-cyan-300">B·A</span>{" "}
              with rank <span className="text-violet-300">r ≪ d, k</span>.
              Trainable params drop{" "}
              <span className="text-pink-300">100×–10000×</span> — the trick
              that turned fine-tuning a 70B model from a datacenter job into a
              workstation job.
            </p>
          </div>
        </div>
      </header>

      <LoRAViz />
    </div>
  );
}
