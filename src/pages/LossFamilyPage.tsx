import { Layers } from "lucide-react";
import LossFamilyViz from "../components/LossFamilyViz";

export default function LossFamilyPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Loss</span>
          <span className="chip">Robust</span>
          <span className="chip">Margin · Focal</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan-400 to-rose-500 items-center justify-center shadow-glow">
            <Layers className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              Losses &amp; friends
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              The robust-loss toolbox —{" "}
              <span className="text-cyan-300">MAE</span> ·{" "}
              <span className="text-emerald-300">Huber</span> ·{" "}
              <span className="text-amber-300">Log-cosh</span> ·{" "}
              <span className="text-violet-300">Hinge</span> ·{" "}
              <span className="text-rose-300">Focal</span> — and what each one
              assumes about your noise.
            </p>
          </div>
        </div>
      </header>

      <LossFamilyViz />
    </div>
  );
}
