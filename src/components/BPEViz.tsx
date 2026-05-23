import { useMemo, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  SAMPLE_CORPUS,
  countComparison,
  encode,
  learnBPE,
  type MergeOp,
} from "../lib/bpe";

const C_CHAR = "#475569";
const C_BPE = "#22d3ee";
const C_NEW = "#fbbf24";
const C_HL = "#a78bfa";

export default function BPEViz() {
  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <BPELab />
      <ComparisonTable />

      <PropertiesGrid />
      <UsageGrid />
    </div>
  );
}

/* =============================== top =============================== */

function DefinitionCard() {
  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4">
      <div className="flex flex-col sm:flex-row sm:items-baseline gap-2 sm:gap-3">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold shrink-0">
          Definition
        </span>
        <p className="text-ink-100 text-sm leading-relaxed">
          <span className="text-cyan-300">Byte-Pair Encoding</span> starts with
          single characters and greedily merges the most frequent adjacent pair
          into a new token. Repeat until the vocabulary reaches a target size.
          The result is a fixed sub-word table where common letter sequences
          ("ing", "the", "tion") become single tokens — the algorithm behind{" "}
          <span className="text-amber-300">tiktoken</span> (GPT-2/3/4) and{" "}
          <span className="text-amber-300">SentencePiece</span>.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`(a^*, b^*) \;=\; \arg\max_{(a,b)\in\mathrm{pairs}} \mathrm{count}(a, b)`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`V_{t+1} \;\leftarrow\; V_t \cup \{a^*b^*\}`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\text{encode}(w) \to \text{apply merges by rank}`} />
      </MathBox>
    </section>
  );
}

/* =============================== bpe lab =============================== */

function BPELab() {
  const [corpus, setCorpus] = useState(SAMPLE_CORPUS);
  const [maxMerges, setMaxMerges] = useState(20);
  const [activeStep, setActiveStep] = useState(8);
  const [input, setInput] = useState("the brown cat jumps over the dog");

  const learned = useMemo(() => learnBPE(corpus, maxMerges), [corpus, maxMerges]);
  const activeMerges = useMemo(() => learned.merges.slice(0, Math.min(activeStep, learned.merges.length)), [learned, activeStep]);
  const encoded = useMemo(() => encode(input, activeMerges), [input, activeMerges]);

  const stepSnapshot = learned.snapshots[Math.min(activeStep, learned.snapshots.length - 1)];

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          BPE lab · learn merges, then tokenize
        </span>
        <span className="text-[10px] font-mono text-ink-500">
          step {activeStep} / {learned.merges.length}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="flex flex-col gap-2">
          <label className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
            Training corpus
          </label>
          <textarea
            value={corpus}
            onChange={(e) => setCorpus(e.target.value)}
            spellCheck={false}
            rows={6}
            className="w-full rounded-lg bg-ink-950 border border-ink-800 px-2 py-1.5 font-mono text-[11px] text-ink-100 leading-snug focus:outline-none focus:border-violet-400/60"
          />
        </div>
        <div className="flex flex-col gap-2">
          <label className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
            Test input (highlighted with current tokenizer)
          </label>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            spellCheck={false}
            className="w-full rounded-lg bg-ink-950 border border-ink-800 px-2 py-2 font-mono text-[12px] text-ink-100 focus:outline-none focus:border-violet-400/60"
          />
          <TokenizedDisplay perWord={encoded.perWord} />
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <SliderRow label="max merges (vocab)" value={maxMerges} onChange={(v) => { const n = Math.round(v); setMaxMerges(n); setActiveStep(Math.min(activeStep, n)); }} min={1} max={80} step={1} fmt={(v) => v.toFixed(0)} color={C_BPE} />
        <SliderRow label="step (animate merges)" value={activeStep} onChange={(v) => setActiveStep(Math.min(Math.round(v), learned.merges.length))} min={0} max={Math.max(1, learned.merges.length)} step={1} fmt={(v) => v.toFixed(0)} color={C_NEW} />
      </div>

      <MergeLog merges={learned.merges} activeStep={activeStep} />
      <CorpusSnapshot snapshot={stepSnapshot} latestMerge={learned.merges[activeStep - 1]} />
    </section>
  );
}

function TokenizedDisplay({ perWord }: { perWord: string[][] }) {
  return (
    <div className="rounded-lg bg-ink-950/80 border border-ink-800/60 p-2.5 flex flex-wrap gap-2 min-h-[44px]">
      {perWord.length === 0 ? (
        <span className="text-[11px] text-ink-500 font-mono">type to see tokens…</span>
      ) : (
        perWord.map((word, wi) => (
          <span key={wi} className="flex gap-[1px]">
            {word.map((t, ti) => (
              <span
                key={ti}
                className="px-1 py-0.5 rounded text-[11px] font-mono"
                style={{
                  background: t.length > 1 ? "rgba(34,211,238,0.18)" : "rgba(99,113,138,0.18)",
                  color: t.length > 1 ? "#a5f3fc" : "#cbd5e1",
                }}
              >
                {t}
              </span>
            ))}
          </span>
        ))
      )}
    </div>
  );
}

function MergeLog({ merges, activeStep }: { merges: MergeOp[]; activeStep: number }) {
  if (merges.length === 0) {
    return (
      <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-3 text-[11px] text-ink-500 font-mono">
        no merges yet — increase corpus / max merges
      </div>
    );
  }
  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2.5 flex flex-col gap-1">
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold mb-1">
        Merge schedule (top rank wins on encode)
      </div>
      <div className="flex flex-wrap gap-1.5">
        {merges.map((m, i) => {
          const inactive = i >= activeStep;
          const isLatest = i === activeStep - 1;
          return (
            <span
              key={i}
              className="px-1.5 py-0.5 rounded text-[11px] font-mono inline-flex items-center gap-1"
              style={{
                background: inactive
                  ? "rgba(60,67,89,0.4)"
                  : isLatest
                    ? "rgba(251,191,36,0.22)"
                    : "rgba(34,211,238,0.14)",
                color: inactive ? "#5b6478" : isLatest ? "#fde68a" : "#a5f3fc",
                border: isLatest ? "1px solid rgba(251,191,36,0.6)" : "1px solid transparent",
              }}
            >
              <span className="text-[9px] text-ink-500">{i + 1}.</span>
              {m.a}+{m.b}→<span className="font-semibold">{m.merged}</span>
              <span className="text-[9px] text-ink-500">×{m.count}</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

function CorpusSnapshot({
  snapshot,
  latestMerge,
}: {
  snapshot: string[][];
  latestMerge?: MergeOp;
}) {
  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2.5 flex flex-col gap-1.5">
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
        Corpus state (unique words)
      </div>
      <div className="flex flex-wrap gap-2.5 font-mono text-[11px]">
        {snapshot.map((w, wi) => (
          <span key={wi} className="flex gap-[1px]">
            {w.map((c, ci) => {
              const isMerged = latestMerge && c === latestMerge.merged;
              return (
                <span
                  key={ci}
                  className="px-1 py-0.5 rounded"
                  style={{
                    background: isMerged ? "rgba(251,191,36,0.25)" : c.length > 1 ? "rgba(34,211,238,0.15)" : "rgba(99,113,138,0.15)",
                    color: isMerged ? "#fde68a" : c.length > 1 ? "#a5f3fc" : "#cbd5e1",
                  }}
                >
                  {c}
                </span>
              );
            })}
          </span>
        ))}
      </div>
    </div>
  );
}

/* =============================== comparison table =============================== */

function ComparisonTable() {
  const merges = useMemo(() => learnBPE(SAMPLE_CORPUS, 40).merges, []);
  const samples = [
    "the lazy dog",
    "tokenization",
    "transformer",
    "the quick brown fox jumps over the lazy dog",
    "supercalifragilisticexpialidocious",
  ];

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          char vs word vs BPE token counts
        </span>
        <span className="text-[10px] font-mono text-ink-500">vocab = 40 merges on toy corpus</span>
      </div>

      <div className="overflow-x-auto -mx-2 px-2">
        <table className="w-full text-[12px] font-mono">
          <thead>
            <tr className="text-left text-ink-500 border-b border-ink-800/60">
              <th className="py-1.5 pr-3">Input</th>
              <th className="py-1.5 pr-3 text-right">chars</th>
              <th className="py-1.5 pr-3 text-right">words</th>
              <th className="py-1.5 pr-3 text-right">BPE tokens</th>
              <th className="py-1.5 pr-3 text-right">compression vs chars</th>
            </tr>
          </thead>
          <tbody>
            {samples.map((s) => {
              const c = countComparison(s, merges);
              const ratio = c.chars > 0 ? c.chars / Math.max(c.tokens, 1) : 1;
              return (
                <tr key={s} className="border-b border-ink-900/60">
                  <td className="py-1.5 pr-3 text-ink-200 truncate max-w-[260px]">{s}</td>
                  <td className="py-1.5 pr-3 text-right tabular-nums">{c.chars}</td>
                  <td className="py-1.5 pr-3 text-right tabular-nums">{c.words}</td>
                  <td className="py-1.5 pr-3 text-right tabular-nums" style={{ color: C_BPE }}>{c.tokens}</td>
                  <td className="py-1.5 pr-3 text-right tabular-nums" style={{ color: C_HL }}>{ratio.toFixed(2)}×</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        Real BPE tables (tiktoken `cl100k_base` has ~100,000 merges) compress
        English to ~4 characters per token. Rare words and code use more
        tokens; common phrases ("the", " and") collapse to one. Whitespace
        handling and pre-tokenization differ between{" "}
        <span className="text-amber-300">tiktoken</span> (regex split) and{" "}
        <span className="text-amber-300">SentencePiece</span> (unicode-aware, no
        pre-split).
      </p>
    </section>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="Time to train">
        <InlineMath math={String.raw`O(\#\text{merges}\cdot \text{corpus size})`} />
      </PropCard>
      <PropCard title="Encoding time">
        <InlineMath math={String.raw`O(\#\text{chars} \cdot \log \#\text{merges})`} />
      </PropCard>
      <PropCard title="Vocab size">
        <InlineMath math={String.raw`|V| = |\text{base chars}| + \#\text{merges}`} />
      </PropCard>
      <PropCard title="Compression rate">
        <InlineMath math={String.raw`\sim 4\text{ chars / token in English}`} />
      </PropCard>
      <PropCard title="Greedy training">
        <InlineMath math={String.raw`\text{always merge the most-frequent pair}`} />
      </PropCard>
      <PropCard title="Greedy decoding">
        <InlineMath math={String.raw`\text{apply merge with lowest rank that fires}`} />
      </PropCard>
      <PropCard title="Determinism">
        <InlineMath math={String.raw`\text{same corpus} \Rightarrow \text{same vocab}`} />
      </PropCard>
      <PropCard title="OOV-free">
        <InlineMath math={String.raw`\text{any UTF-8 input is tokenizable}`} />
      </PropCard>
      <PropCard title="Variants">
        <InlineMath math={String.raw`\mathrm{BPE}, \mathrm{WordPiece}, \mathrm{Unigram}`} />
      </PropCard>
    </section>
  );
}

function PropCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-3.5">
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold mb-2">
        {title}
      </div>
      <div className="text-ink-100 text-base">{children}</div>
    </div>
  );
}

/* =============================== usage =============================== */

function UsageGrid() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <UsageCard title="Where used" accent="text-cyan-300">
        <UsageItem head="GPT-2 / 3 / 4" body="tiktoken (BPE on bytes), 50k–100k vocab." />
        <UsageItem head="LLaMA / Mistral / Qwen" body="SentencePiece BPE, ~32k–150k vocab." />
        <UsageItem head="Claude / Gemini" body="proprietary BPE-flavored tokenizers." />
        <UsageItem head="Whisper" body="multilingual BPE for speech tokens." />
        <UsageItem head="CodeLLaMA / DeepSeek-Coder" body="extended BPE with code-friendly merges." />
        <UsageItem head="BERT" body="WordPiece (BPE cousin) — Greedy + ## prefix." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-emerald-300">
        <UsageItem head="No OOV" body="any input decomposes to byte-level fallback." />
        <UsageItem head="Compact vocab" body="32k tokens covers Wikipedia-level English with ~4× compression." />
        <UsageItem head="Frequent → single token" body={<>articles, suffixes, common words become one ID — embedding table is well-used.</>} />
        <UsageItem head="Cross-lingual" body="byte-level BPE handles emoji, code, math, accents." />
        <UsageItem head="Reversible" body="encode ∘ decode = identity." />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem head="Whitespace sensitivity" body="' the' vs 'the' are different tokens; spaces matter." />
        <UsageItem head="Numbers / digits split oddly" body="'1234' often becomes ['12','34'] — bad for arithmetic." />
        <UsageItem head="Greedy ≠ optimal" body="unigram tokenizers (SentencePiece) can outperform BPE." />
        <UsageItem head="Cross-language imbalance" body="non-Latin scripts often 2–3× more tokens for the same text." />
        <UsageItem head="Tokenizer-bound" body="re-tokenizing breaks downstream caching and fine-tunes." />
        <UsageItem head="Glitch tokens" body="rare merges in training data cause prompt-injection edge cases." />
      </UsageCard>
    </section>
  );
}

function UsageCard({
  title,
  accent,
  children,
}: {
  title: string;
  accent: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-2.5">
      <div className={`text-[10px] uppercase tracking-wider font-bold ${accent}`}>
        {title}
      </div>
      {children}
    </div>
  );
}

function UsageItem({ head, body }: { head: string; body: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 border-l-2 border-ink-800 pl-2.5">
      <div className="text-ink-100 text-[12px] font-mono font-semibold">{head}</div>
      <div className="text-ink-400 text-[11.5px] leading-snug">{body}</div>
    </div>
  );
}

/* =============================== shared =============================== */

function SliderRow({
  label,
  value,
  onChange,
  min,
  max,
  step,
  color,
  fmt,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  color?: string;
  fmt?: (v: number) => string;
}) {
  return (
    <div className="flex items-center gap-2 font-mono text-[12px]">
      <span className="w-40 shrink-0 text-ink-400" style={color ? { color } : undefined}>
        {label}
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1"
        style={color ? { accentColor: color } : undefined}
      />
      <span className="w-12 text-right tabular-nums">
        {fmt ? fmt(value) : value.toFixed(2)}
      </span>
    </div>
  );
}

function MathBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-3 flex items-center justify-center">
      {children}
    </div>
  );
}

// reference to keep tooling happy with unused color constants
void C_CHAR;
