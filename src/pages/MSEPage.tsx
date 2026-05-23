import { Ruler } from "lucide-react";
import MSEViz from "../components/MSEViz";

export default function MSEPage() {
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Loss</span>
          <span className="chip">Regression</span>
          <span className="chip">L2</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-rose-500 to-violet-500 items-center justify-center shadow-glow">
            <Ruler className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight font-mono">
              ½‖ŷ − y‖²
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1 leading-relaxed font-mono text-sm">
              Mean squared error — the <span className="text-rose-300">Gaussian-MLE</span> loss
              for regression. Strictly convex, with the clean gradient{" "}
              <span className="text-amber-300">2(ŷ − y)</span> and a famous{" "}
              <span className="text-emerald-300">outlier weakness</span>.
            </p>
          </div>
        </div>
      </header>

      <MSEViz />
    </div>
  );
}
