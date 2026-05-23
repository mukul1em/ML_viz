import { Feather, Layers, Workflow, Zap } from "lucide-react";
import {
  qwenParamBreakdown,
  type QwenConfig,
} from "../../lib/qwenDemo";

interface Props {
  modelName: string;
  config: QwenConfig;
}

export default function QwenSidebar({ modelName, config }: Props) {
  const breakdown = qwenParamBreakdown(config);
  const headDim = Math.floor(config.d_model / config.q_heads);
  const groupSize = Math.max(1, Math.floor(config.q_heads / config.kv_heads));
  const kvSavings = 1 - config.kv_heads / config.q_heads;

  return (
    <div className="flex flex-col gap-4">
      {/* Headline card */}
      <div className="rounded-xl bg-gradient-to-br from-violet-500/15 via-pink-500/10 to-transparent border border-violet-500/30 p-4">
        <div className="flex items-center gap-2 text-violet-200 mb-1.5">
          <Feather className="w-4 h-4" />
          <span className="text-[11px] uppercase tracking-wider font-bold">Qwen</span>
        </div>
        <div className="text-lg font-bold text-white leading-tight">
          {modelName}
        </div>
        <div className="text-xs text-ink-300 mt-1">
          decoder-only · causal LM
        </div>
      </div>

      {/* Architecture metrics */}
      <Section title="Architecture">
        <Metric
          icon={<Layers className="w-3.5 h-3.5 text-violet-300" />}
          label="Layers"
          value={config.layers.toString()}
        />
        <Metric label="d_model" value={config.d_model.toLocaleString()} />
        <Metric label="ffn_dim" value={config.ffn.toLocaleString()} />
        <Metric label="head_dim" value={headDim.toString()} />
        <Metric label="vocab" value={config.vocab.toLocaleString()} />
      </Section>

      {/* GQA */}
      <Section title="Attention (GQA)">
        <Metric label="Query heads" value={config.q_heads.toString()} accent="#34d399" />
        <Metric label="KV heads" value={config.kv_heads.toString()} accent="#facc15" />
        <Metric
          label="Group size"
          value={`${groupSize} Q : 1 KV`}
          accent="#a78bfa"
        />
        <div className="px-3 py-2 rounded-md bg-amber-400/10 border border-amber-400/40 text-[11px] text-amber-100 leading-relaxed">
          KV cache <strong>{Math.round(kvSavings * 100)}% smaller</strong> than
          equivalent full-MHA. Big inference win.
        </div>
      </Section>

      {/* Signature features */}
      <Section title="Signature features">
        <FeatureRow icon={<Zap className="w-3.5 h-3.5" />} label="RMSNorm" detail="no centering · no bias" color="#22d3ee" />
        <FeatureRow icon={<Workflow className="w-3.5 h-3.5" />} label="RoPE" detail="rotary positions on Q,K" color="#ec4899" />
        <FeatureRow icon={<Layers className="w-3.5 h-3.5" />} label="GQA" detail="shared K/V across Q heads" color="#facc15" />
        <FeatureRow icon={<Zap className="w-3.5 h-3.5" />} label="SwiGLU" detail="gated FFN, 3 linears" color="#7c5cff" />
        <FeatureRow
          icon={<Zap className="w-3.5 h-3.5" />}
          label={config.tied_embeddings ? "Tied embeddings" : "Untied embeddings"}
          detail={config.tied_embeddings ? "W_E = LM head" : "separate output projection"}
          color="#34d399"
        />
      </Section>

      {/* Parameter breakdown */}
      <Section title="Parameters (approx)">
        <div className="text-center mb-1.5">
          <div className="text-[10px] text-ink-400 uppercase tracking-wider">Total</div>
          <div className="text-3xl font-bold text-white font-mono">
            {formatM(breakdown.total)}
          </div>
        </div>
        <ParamBar
          breakdown={[
            { label: "Embedding", value: breakdown.embedding, color: "#fb923c" },
            { label: "Attention", value: breakdown.attention, color: "#34d399" },
            { label: "FFN", value: breakdown.ffn, color: "#7c5cff" },
            { label: "Other", value: breakdown.other, color: "#22d3ee" },
          ]}
        />
        <div className="text-[10px] text-ink-400 mt-1.5">
          One decoder block ≈{" "}
          <span className="font-mono text-ink-200">{formatM(breakdown.perBlock)}</span> ·
          stacked × {config.layers}
        </div>
      </Section>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-3 flex flex-col gap-2">
      <div className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
        {title}
      </div>
      <div className="flex flex-col gap-1.5">{children}</div>
    </div>
  );
}

function Metric({
  label,
  value,
  icon,
  accent,
}: {
  label: string;
  value: string;
  icon?: React.ReactNode;
  accent?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-3 text-xs">
      <span className="flex items-center gap-1.5 text-ink-300">
        {icon}
        {label}
      </span>
      <span
        className="font-mono font-semibold"
        style={{ color: accent ?? "#ffffff" }}
      >
        {value}
      </span>
    </div>
  );
}

function FeatureRow({
  icon,
  label,
  detail,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  detail: string;
  color: string;
}) {
  return (
    <div className="flex items-center gap-2 text-[11.5px]">
      <div
        className="w-6 h-6 rounded-md flex items-center justify-center shrink-0"
        style={{ background: `${color}22`, color, border: `1px solid ${color}55` }}
      >
        {icon}
      </div>
      <div className="flex flex-col leading-tight">
        <span className="font-semibold text-white">{label}</span>
        <span className="text-ink-400 text-[10.5px]">{detail}</span>
      </div>
    </div>
  );
}

function ParamBar({
  breakdown,
}: {
  breakdown: { label: string; value: number; color: string }[];
}) {
  const total = Math.max(
    breakdown.reduce((s, b) => s + b.value, 0),
    0.001
  );
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex h-3 rounded-full overflow-hidden">
        {breakdown.map((b) => (
          <div
            key={b.label}
            title={`${b.label}: ${formatM(b.value)}`}
            style={{
              width: `${(b.value / total) * 100}%`,
              background: b.color,
            }}
          />
        ))}
      </div>
      <div className="grid grid-cols-2 gap-1 text-[10px]">
        {breakdown.map((b) => (
          <div key={b.label} className="flex items-center gap-1.5">
            <span
              className="w-2 h-2 rounded-sm"
              style={{ background: b.color }}
            />
            <span className="text-ink-300">{b.label}</span>
            <span className="ml-auto font-mono text-ink-200">
              {formatM(b.value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatM(m: number): string {
  if (m >= 1000) return `${(m / 1000).toFixed(2)}B`;
  if (m >= 100) return `${m.toFixed(0)}M`;
  if (m >= 10) return `${m.toFixed(1)}M`;
  return `${m.toFixed(2)}M`;
}
