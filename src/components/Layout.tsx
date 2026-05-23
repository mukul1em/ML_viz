import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Bot,
  Brain,
  Code2,
  FunctionSquare as FunctionIcon,
  Lock,
  Sparkles,
} from "lucide-react";
import {
  categoryLabel,
  functions,
  type FunctionCategory,
} from "../data/functions";
import { familyLabel, models, type ModelFamily } from "../data/models";

type Section = "functions" | "llm";

const TOP_TABS: {
  id: Section;
  label: string;
  to: string;
  icon: typeof Bot;
  matches: (path: string) => boolean;
}[] = [
  {
    id: "functions",
    label: "ML Functions",
    to: "/",
    icon: FunctionIcon,
    matches: (p) => !p.startsWith("/llm"),
  },
  {
    id: "llm",
    label: "LLM Models",
    to: "/llm",
    icon: Bot,
    matches: (p) => p.startsWith("/llm"),
  },
];

export default function Layout() {
  const location = useLocation();
  const activeTab =
    TOP_TABS.find((t) => t.matches(location.pathname))?.id ?? "functions";

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top bar */}
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-ink-950/75 border-b border-ink-800/80">
        <div className="px-4 lg:px-6 py-3 flex items-center gap-4">
          <Link to="/" className="flex items-center gap-3 group shrink-0">
            <div className="relative">
              <div className="absolute inset-0 bg-accent/40 blur-xl rounded-2xl group-hover:bg-accent/60 transition" />
              <div className="relative w-9 h-9 rounded-xl bg-gradient-to-br from-accent to-teal flex items-center justify-center shadow-glow">
                <Brain className="w-5 h-5 text-white" />
              </div>
            </div>
            <div className="leading-tight hidden sm:block">
              <div className="text-base font-bold">ML Viz</div>
              <div className="text-[11px] text-ink-400">
                Interactive function & model lab
              </div>
            </div>
          </Link>

          <nav className="flex items-center gap-1 ml-2 sm:ml-4 rounded-xl bg-ink-900/70 border border-ink-800 p-1">
            {TOP_TABS.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <Link
                  key={tab.id}
                  to={tab.to}
                  className={[
                    "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-all",
                    isActive
                      ? "bg-accent text-white shadow-glow font-semibold"
                      : "text-ink-300 hover:text-white hover:bg-ink-800",
                  ].join(" ")}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{tab.label}</span>
                </Link>
              );
            })}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            <span className="hidden md:inline chip text-[11px]">
              <Sparkles className="w-3 h-3 text-accent-soft" />
              Interactive
            </span>
            <span className="text-ink-400" aria-hidden>
              <Code2 className="w-5 h-5" />
            </span>
          </div>
        </div>
      </header>

      <div className="flex flex-1">
        {/* Sidebar (contextual) */}
        <aside className="hidden lg:flex w-72 shrink-0 flex-col gap-5 px-5 py-6 border-r border-ink-800/80 bg-ink-950/50 backdrop-blur-xl sticky top-[57px] h-[calc(100vh-57px)] overflow-y-auto">
          {activeTab === "functions" ? (
            <FunctionsNav pathname={location.pathname} />
          ) : (
            <LLMNav pathname={location.pathname} />
          )}

          <div className="mt-auto pt-4 border-t border-ink-800/70">
            <div className="card p-3.5 text-xs text-ink-300 leading-relaxed">
              <div className="flex items-center gap-2 mb-1.5 text-ink-100 font-semibold">
                <Sparkles className="w-3.5 h-3.5 text-accent-soft" />
                Tip
              </div>
              {activeTab === "functions"
                ? "Drag sliders, watch the math react. Toggle presets to build intuition fast."
                : "Click any block of the architecture to see its definition and an interactive visual."}
            </div>
          </div>
        </aside>

        {/* Main */}
        <div className="flex-1 min-w-0 flex flex-col">
          <main className="flex-1 px-5 lg:px-10 py-8 lg:py-10 max-w-[1400px] w-full mx-auto">
            <Outlet />
          </main>
          <footer className="px-5 lg:px-10 py-6 text-xs text-ink-400 border-t border-ink-800/60">
            Built for intuition. ML Viz — interactive visualizations for ML, DL,
            and LLMs.
          </footer>
        </div>
      </div>
    </div>
  );
}

function FunctionsNav({ pathname }: { pathname: string }) {
  const grouped = functions.reduce<Record<FunctionCategory, typeof functions>>(
    (acc, fn) => {
      (acc[fn.category] ||= []).push(fn);
      return acc;
    },
    { activation: [], attention: [], loss: [], optimizer: [], normalization: [] }
  );
  return (
    <nav className="flex flex-col gap-5 mt-1">
      {(Object.keys(grouped) as FunctionCategory[]).map((cat) => {
        const items = grouped[cat];
        if (!items.length) return null;
        return (
          <div key={cat}>
            <div className="text-[11px] uppercase tracking-[0.14em] text-ink-400 font-semibold px-2 mb-2">
              {categoryLabel[cat]}
            </div>
            <div className="flex flex-col gap-1">
              {items.map((fn) => {
                const Icon = fn.icon;
                const isActive = fn.path && pathname === fn.path;
                const inner = (
                  <div
                    className={[
                      "flex items-center gap-3 px-2.5 py-2 rounded-xl text-sm transition-all",
                      isActive
                        ? "bg-accent/15 text-white border border-accent/30 shadow-glow"
                        : "text-ink-200 hover:bg-ink-800/60 border border-transparent",
                      !fn.available && "opacity-60 cursor-not-allowed",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span className="flex-1">{fn.name}</span>
                    {!fn.available && (
                      <Lock className="w-3.5 h-3.5 text-ink-400" />
                    )}
                  </div>
                );
                return fn.available && fn.path ? (
                  <NavLink key={fn.slug} to={fn.path}>
                    {inner}
                  </NavLink>
                ) : (
                  <div key={fn.slug} title="Coming soon">
                    {inner}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </nav>
  );
}

function LLMNav({ pathname }: { pathname: string }) {
  const grouped = models.reduce<Record<ModelFamily, typeof models>>(
    (acc, m) => {
      (acc[m.family] ||= []).push(m);
      return acc;
    },
    { decoder: [], encoder: [], "encoder-decoder": [], embedding: [] }
  );
  return (
    <nav className="flex flex-col gap-5 mt-1">
      <div>
        <div className="text-[11px] uppercase tracking-[0.14em] text-ink-400 font-semibold px-2 mb-2">
          Overview
        </div>
        <NavLink to="/llm">
          {({ isActive }) => (
            <div
              className={[
                "flex items-center gap-3 px-2.5 py-2 rounded-xl text-sm transition-all",
                isActive
                  ? "bg-accent/15 text-white border border-accent/30 shadow-glow"
                  : "text-ink-200 hover:bg-ink-800/60 border border-transparent",
              ].join(" ")}
            >
              <Brain className="w-4 h-4 shrink-0" />
              <span className="flex-1">All models</span>
            </div>
          )}
        </NavLink>
      </div>

      {(Object.keys(grouped) as ModelFamily[]).map((fam) => {
        const items = grouped[fam];
        if (!items.length) return null;
        return (
          <div key={fam}>
            <div className="text-[11px] uppercase tracking-[0.14em] text-ink-400 font-semibold px-2 mb-2">
              {familyLabel[fam]}
            </div>
            <div className="flex flex-col gap-1">
              {items.map((m) => {
                const Icon = m.icon;
                const isActive = m.path && pathname === m.path;
                const inner = (
                  <div
                    className={[
                      "flex items-center gap-3 px-2.5 py-2 rounded-xl text-sm transition-all",
                      isActive
                        ? "bg-accent/15 text-white border border-accent/30 shadow-glow"
                        : "text-ink-200 hover:bg-ink-800/60 border border-transparent",
                      !m.available && "opacity-60 cursor-not-allowed",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span className="flex-1">{m.name}</span>
                    {!m.available && (
                      <Lock className="w-3.5 h-3.5 text-ink-400" />
                    )}
                  </div>
                );
                return m.available && m.path ? (
                  <NavLink key={m.slug} to={m.path}>
                    {inner}
                  </NavLink>
                ) : (
                  <div key={m.slug} title="Coming soon">
                    {inner}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </nav>
  );
}
