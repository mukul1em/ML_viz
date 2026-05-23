import { KeyRound } from "lucide-react";
import QKVViz from "../components/QKVViz";

export default function QKVPage() {
  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Attention</span>
          <span className="chip">Mechanism</span>
          <span className="chip">Foundational</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-pink-500 to-cyan-400 items-center justify-center shadow-[0_10px_30px_-10px_rgba(244,114,182,0.6)]">
            <KeyRound className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight">
              Q, K, V — Attention from the ground up
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1.5 leading-relaxed">
              The single most important idea in modern deep learning. Every
              token plays three roles — Query, Key, Value — and the way they
              cooperate is what makes transformers tick. Tweak the inputs and
              walk through the math step by step.
            </p>
          </div>
        </div>
      </header>

      <QKVViz />
    </div>
  );
}
