import { Workflow } from "lucide-react";
import AttentionVariantsViz from "../components/AttentionVariantsViz";

export default function AttentionVariantsPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Attention</span>
          <span className="chip">KV cache</span>
          <span className="chip">LLM</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-pink-500 to-cyan-400 items-center justify-center shadow-glow">
            <Workflow className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              MHA → GQA → MQA → MLA
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              Same attention formula, four ways to share K and V across heads —
              and an{" "}
              <span className="text-pink-300">8×–16× swing in inference memory</span>{" "}
              depending on which one you pick. The headline architecture choice
              every modern LLM has to make.
            </p>
          </div>
        </div>
      </header>

      <AttentionVariantsViz />
    </div>
  );
}
