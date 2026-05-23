import { Sigma } from "lucide-react";
import SoftmaxViz from "../components/SoftmaxViz";

export default function SoftmaxPage() {
  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Activation</span>
          <span className="chip">Multi-class</span>
          <span className="chip">Differentiable</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-accent to-teal items-center justify-center shadow-glow">
            <Sigma className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight">
              Softmax
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1.5 leading-relaxed">
              Softmax converts an arbitrary vector of real-valued{" "}
              <em>logits</em> into a probability distribution over{" "}
              <em>n</em> classes. Tweak each logit, change the temperature, and
              watch the distribution morph in real time.
            </p>
          </div>
        </div>
      </header>

      <SoftmaxViz />
    </div>
  );
}
