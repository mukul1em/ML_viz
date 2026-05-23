import { Type } from "lucide-react";
import BPEViz from "../components/BPEViz";

export default function BPEPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Decoding</span>
          <span className="chip">Tokenizer</span>
          <span className="chip">LLM</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-400 to-cyan-400 items-center justify-center shadow-glow">
            <Type className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              (a*, b*) = argmax count(a, b)
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              Greedy merges that turn a corpus into a sub-word vocabulary —
              the foundation under{" "}
              <span className="text-amber-300">tiktoken</span> /{" "}
              <span className="text-amber-300">SentencePiece</span>, every
              GPT, every LLaMA, every embedding lookup.
            </p>
          </div>
        </div>
      </header>

      <BPEViz />
    </div>
  );
}
