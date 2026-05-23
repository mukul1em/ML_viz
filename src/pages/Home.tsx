import { Link } from "react-router-dom";
import { ArrowRight, Lock, Sparkles } from "lucide-react";
import { categoryLabel, functions, type FunctionCategory } from "../data/functions";

export default function Home() {
  const grouped = functions.reduce<Record<FunctionCategory, typeof functions>>(
    (acc, fn) => {
      (acc[fn.category] ||= []).push(fn);
      return acc;
    },
    {
      activation: [],
      attention: [],
      loss: [],
      optimizer: [],
      normalization: [],
      decoding: [],
      "fine-tuning": [],
    }
  );

  return (
    <div className="flex flex-col gap-10">
      <section className="flex flex-col gap-4">
        <div className="chip w-fit">
          <Sparkles className="w-3.5 h-3.5 text-accent-soft" />
          <span>An interactive lab for ML & DL intuition</span>
        </div>
        <h1 className="text-4xl lg:text-5xl font-extrabold tracking-tight leading-[1.1] max-w-3xl">
          Visualize the functions that power{" "}
          <span className="bg-gradient-to-r from-accent to-teal bg-clip-text text-transparent">
            modern machine learning
          </span>
          .
        </h1>
        <p className="text-ink-300 max-w-2xl text-lg leading-relaxed">
          Drag sliders, change temperatures, randomize inputs — and see how
          classical activations, losses, and optimizers behave. Built for
          students, engineers, and anyone who learns by playing.
        </p>
        <div className="flex flex-wrap gap-3 mt-2">
          <Link to="/softmax" className="btn-primary">
            Start with Softmax <ArrowRight className="w-4 h-4" />
          </Link>
          <a
            href="#catalog"
            className="btn"
            onClick={(e) => {
              const el = document.getElementById("catalog");
              if (el) {
                e.preventDefault();
                el.scrollIntoView({ behavior: "smooth" });
              }
            }}
          >
            Browse catalog
          </a>
        </div>
      </section>

      <section id="catalog" className="flex flex-col gap-8 scroll-mt-10">
        {(Object.keys(grouped) as FunctionCategory[]).map((cat) => {
          const items = grouped[cat];
          if (!items.length) return null;
          return (
            <div key={cat} className="flex flex-col gap-3">
              <div className="flex items-baseline justify-between">
                <h2 className="text-lg font-bold tracking-tight">
                  {categoryLabel[cat]}
                </h2>
                <div className="text-xs text-ink-400">
                  {items.filter((i) => i.available).length} available ·{" "}
                  {items.filter((i) => !i.available).length} coming soon
                </div>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {items.map((fn) => {
                  const Icon = fn.icon;
                  const card = (
                    <div
                      className={[
                        "relative card p-5 h-full overflow-hidden group transition",
                        fn.available
                          ? "hover:border-accent/40 hover:-translate-y-0.5"
                          : "opacity-70",
                      ].join(" ")}
                    >
                      <div
                        className={`absolute inset-0 opacity-30 bg-gradient-to-br ${fn.accent}`}
                      />
                      <div className="relative flex flex-col gap-3 h-full">
                        <div className="flex items-center justify-between">
                          <div className="w-10 h-10 rounded-xl bg-ink-900/70 border border-ink-700 flex items-center justify-center">
                            <Icon className="w-5 h-5 text-accent-soft" />
                          </div>
                          {fn.available ? (
                            <span className="chip text-[10px] uppercase">
                              Ready
                            </span>
                          ) : (
                            <span className="chip text-[10px] uppercase">
                              <Lock className="w-3 h-3" />
                              Soon
                            </span>
                          )}
                        </div>
                        <div className="font-bold text-lg">{fn.name}</div>
                        <p className="text-sm text-ink-300 leading-relaxed flex-1">
                          {fn.blurb}
                        </p>
                        {fn.available && fn.path && (
                          <div className="flex items-center gap-1.5 text-accent-soft text-sm font-medium pt-1">
                            Open visualization
                            <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
                          </div>
                        )}
                      </div>
                    </div>
                  );
                  return fn.available && fn.path ? (
                    <Link key={fn.slug} to={fn.path}>
                      {card}
                    </Link>
                  ) : (
                    <div key={fn.slug}>{card}</div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </section>
    </div>
  );
}
