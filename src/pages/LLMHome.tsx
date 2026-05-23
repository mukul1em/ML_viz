import { Link } from "react-router-dom";
import { ArrowRight, Lock, Sparkles } from "lucide-react";
import { familyLabel, models, type ModelFamily } from "../data/models";

export default function LLMHome() {
  const grouped = models.reduce<Record<ModelFamily, typeof models>>(
    (acc, m) => {
      (acc[m.family] ||= []).push(m);
      return acc;
    },
    { decoder: [], encoder: [], "encoder-decoder": [], embedding: [] }
  );

  return (
    <div className="flex flex-col gap-10">
      <section className="flex flex-col gap-4">
        <div className="chip w-fit">
          <Sparkles className="w-3.5 h-3.5 text-accent-soft" />
          <span>LLM architecture explorer</span>
        </div>
        <h1 className="text-4xl lg:text-5xl font-extrabold tracking-tight leading-[1.1] max-w-3xl">
          Dissect the architectures behind{" "}
          <span className="bg-gradient-to-r from-accent to-teal bg-clip-text text-transparent">
            modern language models
          </span>
          .
        </h1>
        <p className="text-ink-300 max-w-2xl text-lg leading-relaxed">
          Click any block of a model and see what it does, the math behind it,
          and an interactive visual. Starting with the canonical decoder-only
          transformer — <strong>GPT</strong>.
        </p>
        <div className="flex flex-wrap gap-3 mt-2">
          <Link to="/llm/gpt" className="btn-primary">
            Open GPT visualization <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>

      <section className="flex flex-col gap-8">
        {(Object.keys(grouped) as ModelFamily[]).map((fam) => {
          const items = grouped[fam];
          if (!items.length) return null;
          return (
            <div key={fam} className="flex flex-col gap-3">
              <div className="flex items-baseline justify-between">
                <h2 className="text-lg font-bold tracking-tight">
                  {familyLabel[fam]}
                </h2>
                <div className="text-xs text-ink-400">
                  {items.filter((i) => i.available).length} available ·{" "}
                  {items.filter((i) => !i.available).length} coming soon
                </div>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {items.map((m) => {
                  const Icon = m.icon;
                  const card = (
                    <div
                      className={[
                        "relative card p-5 h-full overflow-hidden group transition",
                        m.available
                          ? "hover:border-accent/40 hover:-translate-y-0.5"
                          : "opacity-70",
                      ].join(" ")}
                    >
                      <div
                        className={`absolute inset-0 opacity-30 bg-gradient-to-br ${m.accent}`}
                      />
                      <div className="relative flex flex-col gap-3 h-full">
                        <div className="flex items-center justify-between">
                          <div className="w-10 h-10 rounded-xl bg-ink-900/70 border border-ink-700 flex items-center justify-center">
                            <Icon className="w-5 h-5 text-accent-soft" />
                          </div>
                          {m.available ? (
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
                        <div>
                          <div className="font-bold text-lg">{m.name}</div>
                          {m.tagline && (
                            <div className="text-[11px] text-ink-400">
                              {m.tagline}
                            </div>
                          )}
                        </div>
                        <p className="text-sm text-ink-300 leading-relaxed flex-1">
                          {m.blurb}
                        </p>
                        {m.available && m.path && (
                          <div className="flex items-center gap-1.5 text-accent-soft text-sm font-medium pt-1">
                            Explore architecture
                            <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
                          </div>
                        )}
                      </div>
                    </div>
                  );
                  return m.available && m.path ? (
                    <Link key={m.slug} to={m.path}>
                      {card}
                    </Link>
                  ) : (
                    <div key={m.slug}>{card}</div>
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
