import { Scale } from "lucide-react";
import KLDivergenceViz from "../components/KLDivergenceViz";

export default function KLDivergencePage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Loss</span>
          <span className="chip">Information-theoretic</span>
          <span className="chip">Asymmetric</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-violet-500 to-amber-400 items-center justify-center shadow-glow">
            <Scale className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              D<sub>KL</sub>(p ‖ q)
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              Kullback–Leibler divergence. <span className="text-emerald-300">Forward</span> covers
              the mass, <span className="text-amber-300">reverse</span> chases the mode, and{" "}
              <span className="text-pink-300">JS</span> is the symmetric in-between. The same
              quantity behind <span className="text-violet-300">VAE</span>,{" "}
              <span className="text-violet-300">RLHF</span>, and{" "}
              <span className="text-violet-300">distillation</span>.
            </p>
          </div>
        </div>
      </header>

      <KLDivergenceViz />
    </div>
  );
}
