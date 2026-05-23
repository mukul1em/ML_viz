import { useMemo, useState } from "react";
import { Tex as InlineMath, TexBlock as BlockMath } from "./Tex";
import {
  VARIANTS,
  denseParams,
  loraParams,
  loraReconstruct,
  matSub,
  randomGaussian,
  rankSweep,
  frobenius,
  type LoRAConfig,
} from "../lib/lora";

const C_AXIS = "rgba(255,255,255,0.10)";
const C_W = "#a78bfa";
const C_BA = "#22d3ee";
const C_ERR = "#f472b6";
const C_OK = "#34d399";
const C_HOT = "#fbbf24";

export default function LoRAViz() {
  return (
    <div className="flex flex-col gap-5">
      <DefinitionCard />
      <EquationsRow />

      <LoRALab />
      <RankSweepPanel />
      <AdapterSwapDiagram />
      <VariantsTable />

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
          <span className="text-violet-300">LoRA</span> freezes the base weight
          matrix <InlineMath math={String.raw`W \in \mathbb{R}^{d\times k}`} />{" "}
          and learns a low-rank update{" "}
          <InlineMath math={String.raw`\Delta W = BA`} /> with{" "}
          <InlineMath math={String.raw`B \in \mathbb{R}^{d\times r}`} />,{" "}
          <InlineMath math={String.raw`A \in \mathbb{R}^{r\times k}`} />,{" "}
          <InlineMath math={String.raw`r \ll \min(d,k)`} />. Trainable params
          drop from <InlineMath math={String.raw`d\,k`} /> to{" "}
          <InlineMath math={String.raw`(d+k)\,r`} /> — typically <em>100×–10000×</em> fewer.
        </p>
      </div>
    </section>
  );
}

function EquationsRow() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <MathBox>
        <BlockMath math={String.raw`W' = W + \tfrac{\alpha}{r}\,B A`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\#\theta_{\text{LoRA}} = (d+k)\,r,\quad \#\theta_{W} = d\,k`} />
      </MathBox>
      <MathBox>
        <BlockMath math={String.raw`\text{init: } B = 0,\;\; A \sim \mathcal{N}(0, 1/r)`} />
      </MathBox>
    </section>
  );
}

/* =============================== lora lab =============================== */

function LoRALab() {
  const d = 32;
  const k = 32;
  const [rank, setRank] = useState(4);
  const [alpha, setAlpha] = useState(16);
  const [seed, setSeed] = useState(7);
  const [wSeed, setWSeed] = useState(3);

  // Synthetic target W: low-rank plus noise so a small r is a great approximation.
  const W = useMemo(() => {
    const U = randomGaussian(d, 6, 1, wSeed);
    const V = randomGaussian(6, k, 1, wSeed + 313);
    // matmul U·V then add noise
    const base = U.map((row) => {
      const out = new Array(k).fill(0);
      for (let j = 0; j < k; j++) {
        let s = 0;
        for (let r = 0; r < 6; r++) s += row[r] * V[r][j];
        out[j] = s / 2.4;
      }
      return out;
    });
    const noise = randomGaussian(d, k, 0.08, wSeed + 999);
    return base.map((row, i) => row.map((v, j) => v + noise[i][j]));
  }, [wSeed]);

  // LoRA adapters
  const B = useMemo(() => randomGaussian(d, rank, 1, seed), [d, rank, seed]);
  const A = useMemo(() => randomGaussian(rank, k, 1 / Math.sqrt(Math.max(rank, 1)), seed + 7919), [rank, k, seed]);
  const approx = useMemo(() => loraReconstruct(W, A, B, alpha, rank), [W, A, B, alpha, rank]);
  const residual = useMemo(() => matSub(W, approx), [W, approx]);

  const errFrac = useMemo(() => {
    const w = frobenius(W);
    const e = frobenius(residual);
    return w > 0 ? e / w : 0;
  }, [W, residual]);

  const cfg: LoRAConfig = { d, k, r: rank, alpha };
  const dense = denseParams(cfg);
  const lora = loraParams(cfg);

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Lab · W vs BA reconstruction
        </span>
        <div className="flex items-center gap-2">
          <button onClick={() => setWSeed((s) => s + 1)} className="text-[10px] font-mono text-ink-400 hover:text-white border border-ink-700 px-2 py-0.5 rounded">
            re-roll W
          </button>
          <button onClick={() => setSeed((s) => s + 1)} className="text-[10px] font-mono text-ink-400 hover:text-white border border-ink-700 px-2 py-0.5 rounded">
            re-roll adapter
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
        <SliderRow label="rank r" value={rank} onChange={(v) => setRank(Math.round(v))} min={1} max={32} step={1} fmt={(v) => v.toFixed(0)} color={C_BA} />
        <SliderRow label="α (scale)" value={alpha} onChange={(v) => setAlpha(Math.round(v))} min={1} max={64} step={1} fmt={(v) => v.toFixed(0)} color={C_HOT} />
        <div className="flex items-end gap-2 font-mono text-[12px]">
          <Stat label="LoRA params" value={`${lora}`} color={C_BA} hint={`vs ${dense} dense`} />
          <Stat label="ratio" value={`${(lora / dense * 100).toFixed(1)}%`} color={C_OK} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <MatrixCard title="W (frozen base)" matrix={W} color={C_W} />
        <MatrixCard title={`BA · α/r  (rank ${rank})`} matrix={approx} color={C_BA} />
        <MatrixCard title={`W − BA (residual, ‖·‖/‖W‖ = ${(errFrac * 100).toFixed(1)}%)`} matrix={residual} color={C_ERR} />
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        Increase rank to see the residual shrink. At{" "}
        <InlineMath math={String.raw`r=6`} /> (the true latent rank of this
        synthetic W) the approximation is essentially perfect, yet uses only{" "}
        <strong>{(loraParams({ ...cfg, r: 6 }) / dense * 100).toFixed(1)}%</strong> of W's
        params. This is the empirical observation behind LoRA: in practice,
        the <em>update</em> matrix during fine-tuning has very low intrinsic
        rank.
      </p>
    </section>
  );
}

function MatrixCard({ title, matrix, color }: { title: string; matrix: number[][]; color: string }) {
  const W = 220;
  const H = 220;
  const PAD = 8;
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const cw = (W - PAD * 2) / cols;
  const ch = (H - PAD * 2 - 18) / rows;

  let vMax = 0;
  for (const row of matrix) for (const v of row) vMax = Math.max(vMax, Math.abs(v));
  vMax = Math.max(vMax, 1e-9);

  const colorFor = (v: number) => {
    const t = Math.max(-1, Math.min(1, v / vMax));
    if (t >= 0) {
      const r = Math.round(40 + 180 * t);
      const g = Math.round(20 + 110 * t);
      const b = Math.round(90 - 60 * t);
      return `rgb(${r},${g},${b})`;
    } else {
      const a = -t;
      const r = Math.round(20 + 30 * a);
      const g = Math.round(50 + 90 * a);
      const b = Math.round(90 + 130 * a);
      return `rgb(${r},${g},${b})`;
    }
  };

  return (
    <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
      <div className="text-[10px] uppercase tracking-wider font-bold mb-1" style={{ color }}>
        {title}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full block">
        {matrix.map((row, i) =>
          row.map((v, j) => (
            <rect
              key={`${i}-${j}`}
              x={PAD + j * cw}
              y={PAD + 12 + i * ch}
              width={cw + 0.5}
              height={ch + 0.5}
              fill={colorFor(v)}
            />
          ))
        )}
      </svg>
    </div>
  );
}

/* =============================== rank sweep =============================== */

function RankSweepPanel() {
  const d = 32;
  const k = 32;
  const [seed, setSeed] = useState(11);
  const [wSeed, setWSeed] = useState(3);
  const rMax = 32;
  const alpha = 1;

  const W = useMemo(() => {
    const U = randomGaussian(d, 6, 1, wSeed);
    const V = randomGaussian(6, k, 1, wSeed + 313);
    const base = U.map((row) => {
      const out = new Array(k).fill(0);
      for (let j = 0; j < k; j++) {
        let s = 0;
        for (let r = 0; r < 6; r++) s += row[r] * V[r][j];
        out[j] = s / 2.4;
      }
      return out;
    });
    const noise = randomGaussian(d, k, 0.08, wSeed + 999);
    return base.map((row, i) => row.map((v, j) => v + noise[i][j]));
  }, [wSeed]);

  const sweep = useMemo(() => rankSweep(W, rMax, seed, alpha), [W, seed]);

  const W_ = 720;
  const H = 240;
  const PAD = { l: 44, r: 50, t: 14, b: 28 };
  const xMin = 1;
  const xMax = rMax;
  const yMin = 0;
  const yMax = 1.4;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W_ - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  // Second axis: param ratio
  const yMaxRatio = 2.2;
  const sy2 = (v: number) => H - PAD.b - ((v - 0) / (yMaxRatio - 0)) * (H - PAD.t - PAD.b);

  const errPath = sweep.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.r).toFixed(2)} ${sy(p.error).toFixed(2)}`).join(" ");
  const ratioPath = sweep.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.r).toFixed(2)} ${sy2(p.ratio).toFixed(2)}`).join(" ");

  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Rank sweep · reconstruction error vs parameter cost
        </span>
        <div className="flex items-center gap-2">
          <button onClick={() => setWSeed((s) => s + 1)} className="text-[10px] font-mono text-ink-400 hover:text-white border border-ink-700 px-2 py-0.5 rounded">
            re-roll W
          </button>
          <button onClick={() => setSeed((s) => s + 1)} className="text-[10px] font-mono text-ink-400 hover:text-white border border-ink-700 px-2 py-0.5 rounded">
            re-roll seed
          </button>
        </div>
      </div>

      <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-2">
        <svg viewBox={`0 0 ${W_} ${H}`} className="w-full block">
          <Grid sx={sx} sy={sy} yTicks={[0, 0.25, 0.5, 0.75, 1, 1.25]} xTicks={[1, 4, 8, 12, 16, 20, 24, 28, 32]} W={W_} H={H} PAD={PAD} xLabel="rank r" />
          {/* right-axis ticks for ratio */}
          {[0, 0.5, 1, 1.5, 2].map((v) => (
            <g key={v}>
              <line x1={W_ - PAD.r} y1={sy2(v)} x2={W_ - PAD.r + 4} y2={sy2(v)} stroke={C_AXIS} />
              <text x={W_ - PAD.r + 6} y={sy2(v) + 3} fontSize={9} fill={C_OK} fontFamily="JetBrains Mono, monospace">
                {v.toFixed(2)}×
              </text>
            </g>
          ))}
          <path d={errPath} stroke={C_ERR} strokeWidth={2.4} fill="none" />
          <path d={ratioPath} stroke={C_OK} strokeWidth={2.0} fill="none" strokeDasharray="4 3" />
          {/* legend */}
          <g transform={`translate(${PAD.l + 16}, ${PAD.t + 8})`}>
            <rect width="160" height="36" rx={6} fill="rgba(0,0,0,0.45)" />
            <line x1={10} y1={12} x2={26} y2={12} stroke={C_ERR} strokeWidth={2.4} />
            <text x={32} y={15} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">‖W − BA‖ / ‖W‖</text>
            <line x1={10} y1={28} x2={26} y2={28} stroke={C_OK} strokeWidth={2} strokeDasharray="4 3" />
            <text x={32} y={31} fontSize={10} fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">params(LoRA)/params(W)</text>
          </g>
        </svg>
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        Error drops sharply through the true latent rank then plateaus (the
        residual is just noise from here on). The parameter cost grows{" "}
        <em>linearly</em> in r while error decreases <em>convex-monotonically</em>{" "}
        — so very small r gets you most of the way there. Hu et al. (2021) found
        r=4–16 is enough for most LLM tasks even on 175B-param GPT-3.
      </p>
    </section>
  );
}

/* =============================== adapter swap diagram =============================== */

function AdapterSwapDiagram() {
  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          One base · many adapters
        </span>
        <span className="text-[10px] font-mono text-ink-500">hot-swap per request</span>
      </div>

      <div className="rounded-xl bg-ink-950/80 border border-ink-800/60 p-3">
        <svg viewBox="0 0 720 220" className="w-full block">
          {/* base */}
          <rect x={280} y={80} width={160} height={60} rx={10} fill="rgba(167,139,250,0.18)" stroke={C_W} />
          <text x={360} y={108} fontSize={13} fill={C_W} textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontWeight={700}>W (frozen)</text>
          <text x={360} y={126} fontSize={10} fill="#cbd5e1" textAnchor="middle" fontFamily="JetBrains Mono, monospace">7B · 70B · ... params</text>

          {/* adapters */}
          {[
            { x: 20, y: 14, name: "code-llama-r16", color: C_BA },
            { x: 20, y: 100, name: "medical-r8", color: C_BA },
            { x: 20, y: 180, name: "RLHF-r32", color: C_BA },
            { x: 520, y: 14, name: "korean-r16", color: C_HOT },
            { x: 520, y: 100, name: "math-tutor-r8", color: C_HOT },
            { x: 520, y: 180, name: "legal-r32", color: C_HOT },
          ].map((a, i) => (
            <g key={i}>
              <rect x={a.x} y={a.y - 14} width={160} height={28} rx={6} fill={`${a.color}22`} stroke={a.color} />
              <text x={a.x + 80} y={a.y + 4} fontSize={11} fill={a.color} textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontWeight={700}>
                {a.name}
              </text>
              <line x1={a.x < 360 ? a.x + 160 : a.x} y1={a.y} x2={a.x < 360 ? 280 : 440} y2={110} stroke="rgba(255,255,255,0.18)" strokeDasharray="3 3" />
            </g>
          ))}

          <text x={360} y={206} fontSize={10} fill="#7a8094" textAnchor="middle" fontFamily="JetBrains Mono, monospace">
            adapters are ~MB each; the base is GBs and stays in HBM
          </text>
        </svg>
      </div>

      <p className="text-[11px] text-ink-400 leading-snug">
        Because <InlineMath math={String.raw`W' = W + \tfrac{\alpha}{r}BA`} />{" "}
        and BA is tiny, a serving cluster can keep dozens of fine-tunes in
        memory per base model. Routing each request to the right adapter is
        nearly free (just an extra matmul of shape{" "}
        <InlineMath math={String.raw`(d,r)(r,k)`} />). vLLM, S-LoRA, Punica all
        exploit this for multi-tenant LLM serving.
      </p>
    </section>
  );
}

/* =============================== variants table =============================== */

function VariantsTable() {
  return (
    <section className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          The PEFT family · how each variant compresses the update
        </span>
      </div>

      <div className="overflow-x-auto -mx-2 px-2">
        <table className="w-full text-[12px] font-mono">
          <thead>
            <tr className="text-left text-ink-500 border-b border-ink-800/60">
              <th className="py-1.5 pr-3">Method</th>
              <th className="py-1.5 pr-3">Storage cost</th>
              <th className="py-1.5 pr-3">Compute cost</th>
              <th className="py-1.5 pr-3">Notes</th>
            </tr>
          </thead>
          <tbody>
            {VARIANTS.map((v) => (
              <tr key={v.name} className="border-b border-ink-900/60 hover:bg-ink-900/40">
                <td className="py-1.5 pr-3 text-ink-100" style={{ color: C_BA }}>{v.name}</td>
                <td className="py-1.5 pr-3 text-ink-200">{v.storageCost}</td>
                <td className="py-1.5 pr-3 text-ink-200">{v.computeCost}</td>
                <td className="py-1.5 pr-3 text-ink-400 text-[11px]">{v.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* =============================== properties =============================== */

function PropertiesGrid() {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <PropCard title="Trainable params">
        <InlineMath math={String.raw`(d+k)\,r \ll d\,k`} />
      </PropCard>
      <PropCard title="Param ratio">
        <InlineMath math={String.raw`\rho = \frac{(d+k)r}{dk}`} />
      </PropCard>
      <PropCard title="Init convention">
        <InlineMath math={String.raw`B = 0,\;A \sim \mathcal{N}(0, \sigma^2)`} />
      </PropCard>
      <PropCard title="Scaling factor">
        <InlineMath math={String.raw`\Delta W = \tfrac{\alpha}{r}\,BA`} />
      </PropCard>
      <PropCard title="Forward overhead">
        <InlineMath math={String.raw`+\,O(d r + r k)\text{ matmul}`} />
      </PropCard>
      <PropCard title="Merge at inference">
        <InlineMath math={String.raw`W' \leftarrow W + \tfrac{\alpha}{r}BA\;\;(\text{zero overhead})`} />
      </PropCard>
      <PropCard title="Typical rank">
        <InlineMath math={String.raw`r\in\{4, 8, 16, 32\}`} />
      </PropCard>
      <PropCard title="VRAM during training">
        <InlineMath math={String.raw`\ll \text{full fine-tune}\;(\text{only BA grads + optim states})`} />
      </PropCard>
      <PropCard title="Targets">
        <InlineMath math={String.raw`W_Q, W_K, W_V, W_O \;(\text{attention}) + W_{FF}`} />
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
      <UsageCard title="Where used" accent="text-violet-300">
        <UsageItem head="Instruction fine-tuning" body="Alpaca, Vicuna, Open-Hermes all started as LoRA on LLaMA." />
        <UsageItem head="RLHF / DPO" body="apply LoRA on top of a frozen base + reward model — much cheaper than full FT." />
        <UsageItem head="Multi-tenant serving" body="vLLM / S-LoRA / Punica hot-swap thousands of adapters." />
        <UsageItem head="Domain adaptation" body="medicine, code, law, language — one adapter each, same base." />
        <UsageItem head="Diffusion models" body="Stable Diffusion LoRAs (DreamBooth-LoRA, character LoRAs)." />
        <UsageItem head="Audio / vision" body="Whisper, ViT, CLIP all support LoRA fine-tuning." />
      </UsageCard>

      <UsageCard title="Why used" accent="text-cyan-300">
        <UsageItem head="100×–10000× fewer params" body="LLaMA-2-7B → ~6M trainable instead of 7B." />
        <UsageItem head="Fits one GPU" body="combined with QLoRA, 65B models fit on a 48GB card." />
        <UsageItem head="No serving overhead" body={<>at inference, merge into{" "}<InlineMath math={String.raw`W'`} /> with zero extra latency.</>} />
        <UsageItem head="Composable" body="adapters can be added (BA₁ + BA₂) for multi-task." />
        <UsageItem head="Quick iteration" body="train in hours, not days; cheap to keep many checkpoints." />
        <UsageItem head="Empirical intrinsic rank" body="fine-tuning updates have very low rank in practice." />
      </UsageCard>

      <UsageCard title="Limitations" accent="text-amber-300">
        <UsageItem head="Can't add new knowledge well" body="LoRA shifts the distribution; pretraining-scale facts still hard." />
        <UsageItem head="Layer / target choice matters" body="results vary widely by which W matrices you adapt." />
        <UsageItem head="α / r interaction" body="too-large α destabilizes early training." />
        <UsageItem head="Quality gap at low rank" body="r=2 underfits hard tasks; full FT still wins by a few %." />
        <UsageItem head="Catastrophic forgetting" body="merged LoRA can shift the base behavior on unrelated tasks." />
        <UsageItem head="Not free during training" body="forward & backward still go through W (frozen); only optimizer state shrinks." />
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

function Stat({
  label,
  value,
  color,
  hint,
}: {
  label: string;
  value: string;
  color?: string;
  hint?: string;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-wider text-ink-500">{label}</span>
      <span className="text-base font-semibold tabular-nums" style={color ? { color } : undefined}>
        {value}
      </span>
      {hint && <span className="text-[10px] text-ink-500">{hint}</span>}
    </div>
  );
}

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
      <span className="w-28 shrink-0 text-ink-400" style={color ? { color } : undefined}>
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

interface GridProps {
  sx: (v: number) => number;
  sy: (v: number) => number;
  yTicks: number[];
  xTicks: number[];
  W: number;
  H: number;
  PAD: { l: number; r: number; t: number; b: number };
  xLabel?: string;
}

function Grid({ sx, sy, yTicks, xTicks, W, H, PAD, xLabel }: GridProps) {
  return (
    <g>
      <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke={C_AXIS} />
      <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke={C_AXIS} />
      {yTicks.map((v) => (
        <g key={`y-${v}`}>
          <line x1={PAD.l - 4} y1={sy(v)} x2={W - PAD.r} y2={sy(v)} stroke="rgba(255,255,255,0.04)" />
          <text x={PAD.l - 6} y={sy(v) + 3} textAnchor="end" fontSize={9} fill={C_ERR} fontFamily="JetBrains Mono, monospace">
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      {xTicks.map((v) => (
        <g key={`x-${v}`}>
          <line x1={sx(v)} y1={H - PAD.b} x2={sx(v)} y2={H - PAD.b + 4} stroke={C_AXIS} />
          <text x={sx(v)} y={H - PAD.b + 14} textAnchor="middle" fontSize={9} fill="#7a8094" fontFamily="JetBrains Mono, monospace">
            {v}
          </text>
        </g>
      ))}
      {xLabel && (
        <text x={W - PAD.r - 6} y={H - PAD.b - 4} textAnchor="end" fontSize={10} fill="#5b6478" fontFamily="JetBrains Mono, monospace">
          {xLabel}
        </text>
      )}
    </g>
  );
}
