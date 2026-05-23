import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowDown,
  ArrowRight,
  Boxes,
  Check,
  Compass,
  Database,
  Globe2,
  Layers,
  Map as MapIcon,
  Network,
  Search,
  Sigma,
  Sparkles,
  Tags,
  Telescope,
  Workflow,
} from "lucide-react";
import { TexBlock as BlockMath } from "./Tex";
import {
  CATEGORY_COLORS,
  CATEGORY_LABELS,
  corpus,
  findItem,
  type EmbeddingCategory,
  type EmbeddingItem,
} from "../data/embeddingCorpus";
import { embeddingModels } from "../data/embeddingModels";
import {
  EMBED_DIM,
  analogy,
  cosine,
  getEmbedding,
  nearestNeighbors,
} from "../lib/embeddingDemo";

export default function EmbeddingsViz() {
  return (
    <div className="flex flex-col gap-12">
      <Hero />
      <SemanticMapSection />
      <SimilaritySection />
      <NearestNeighborsSection />
      <AnalogySection />
      <HowEmbeddingsAreMade />
      <ModelComparisonSection />
      <WhyEmbeddingsMatter />
    </div>
  );
}

/* =============================== HERO =============================== */

function Hero() {
  return (
    <section className="flex flex-col gap-5">
      <Eyebrow icon={<Compass className="w-3 h-3" />}>The big idea</Eyebrow>
      <h2 className="text-2xl lg:text-3xl font-extrabold tracking-tight">
        Language becomes{" "}
        <span className="bg-gradient-to-r from-cyan-400 to-fuchsia-400 bg-clip-text text-transparent">
          geometry
        </span>
      </h2>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        An embedding model takes a piece of text and turns it into a fixed-size
        vector — a point in a high-dimensional space. Train it well and
        meaning becomes <em>direction</em>: similar texts land near each
        other, and you can ask geometric questions like "what's nearest to{" "}
        <em>cat</em>?" or "<em>king</em> is to <em>queen</em> as <em>man</em>{" "}
        is to ???"
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
        <RoleCard
          color="#22d3ee"
          icon={<Tags className="w-4 h-4" />}
          title="Text in"
          body="Words, sentences, paragraphs, even whole documents. Tokenize and feed an encoder."
        />
        <RoleCard
          color="#a78bfa"
          icon={<Network className="w-4 h-4" />}
          title="Vector out"
          body="A dense vector of e.g. 768 or 1536 floats. Each dimension is a learned semantic axis."
        />
        <RoleCard
          color="#f472b6"
          icon={<Telescope className="w-4 h-4" />}
          title="Distance = meaning"
          body="Cosine similarity between vectors approximates semantic similarity between the texts."
        />
      </div>
    </section>
  );
}

/* =============================== SEMANTIC MAP =============================== */

function SemanticMapSection() {
  const [selectedId, setSelectedId] = useState<number>(8); // "cat"
  const [hoverId, setHoverId] = useState<number | null>(null);
  const [activeCats, setActiveCats] = useState<Set<EmbeddingCategory>>(
    () =>
      new Set<EmbeddingCategory>([
        "animals",
        "vehicles",
        "food",
        "emotions",
        "tech",
        "sports",
        "royalty",
      ])
  );

  const selected = corpus[selectedId];
  const selectedEmbed = getEmbedding(selectedId);
  const topNeighbors = useMemo(
    () => nearestNeighbors(selectedId, 5),
    [selectedId]
  );

  const toggleCat = (c: EmbeddingCategory) => {
    setActiveCats((prev) => {
      const next = new Set(prev);
      if (next.has(c)) next.delete(c);
      else next.add(c);
      return next;
    });
  };

  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<MapIcon className="w-3 h-3" />}>The semantic map</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        50 words, mapped by meaning
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Each dot is a word embedded by a model and projected to 2D. Words that
        mean similar things cluster together — purely from how they appear in
        text, without anyone hand-labeling categories. Click a dot to inspect
        its vector and nearest neighbors.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-4">
        {/* Map */}
        <div className="rounded-2xl bg-ink-950/70 border border-ink-800/80 p-4 overflow-hidden">
          {/* Category legend / filter */}
          <div className="flex flex-wrap gap-1.5 mb-3">
            {(Object.keys(CATEGORY_LABELS) as EmbeddingCategory[]).map(
              (cat) => (
                <button
                  key={cat}
                  onClick={() => toggleCat(cat)}
                  className={[
                    "flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium border transition-all",
                    activeCats.has(cat)
                      ? "border-transparent"
                      : "opacity-40 border-ink-700",
                  ].join(" ")}
                  style={{
                    background: activeCats.has(cat)
                      ? `${CATEGORY_COLORS[cat]}22`
                      : "transparent",
                    borderColor: activeCats.has(cat)
                      ? `${CATEGORY_COLORS[cat]}66`
                      : undefined,
                    color: activeCats.has(cat)
                      ? CATEGORY_COLORS[cat]
                      : "#8a90a4",
                  }}
                >
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ background: CATEGORY_COLORS[cat] }}
                  />
                  {CATEGORY_LABELS[cat]}
                </button>
              )
            )}
          </div>

          <svg
            viewBox="0 0 720 540"
            className="w-full block"
            style={{ maxHeight: 540 }}
          >
            {/* Background grid */}
            <defs>
              <pattern
                id="grid"
                width="40"
                height="40"
                patternUnits="userSpaceOnUse"
              >
                <path
                  d="M 40 0 L 0 0 0 40"
                  fill="none"
                  stroke="rgba(255,255,255,0.04)"
                  strokeWidth="1"
                />
              </pattern>
              <radialGradient id="sel-glow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#ffffff" stopOpacity="0.4" />
                <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
              </radialGradient>
            </defs>
            <rect width="720" height="540" fill="url(#grid)" />

            {/* Cluster halos */}
            {(Object.keys(CATEGORY_LABELS) as EmbeddingCategory[])
              .filter((c) => activeCats.has(c))
              .map((cat) => {
                const items = corpus.filter((i) => i.category === cat);
                if (!items.length) return null;
                const cx =
                  items.reduce((s, i) => s + i.x, 0) / items.length;
                const cy =
                  items.reduce((s, i) => s + i.y, 0) / items.length;
                return (
                  <g key={cat}>
                    <circle
                      cx={cx}
                      cy={cy}
                      r={80}
                      fill={`${CATEGORY_COLORS[cat]}11`}
                    />
                    <text
                      x={cx}
                      y={cy - 70}
                      textAnchor="middle"
                      fontSize={11}
                      fontWeight={700}
                      letterSpacing={1.5}
                      fill={CATEGORY_COLORS[cat]}
                      opacity={0.6}
                    >
                      {CATEGORY_LABELS[cat].toUpperCase()}
                    </text>
                  </g>
                );
              })}

            {/* Lines from selected to top neighbors */}
            {selected &&
              topNeighbors.map((nb) => (
                <line
                  key={`l-${nb.item.id}`}
                  x1={selected.x}
                  y1={selected.y}
                  x2={nb.item.x}
                  y2={nb.item.y}
                  stroke={CATEGORY_COLORS[selected.category]}
                  strokeOpacity={0.18 + nb.similarity * 0.55}
                  strokeWidth={1}
                  strokeDasharray="3 3"
                />
              ))}

            {/* Dots */}
            {corpus.map((item) => {
              const visible = activeCats.has(item.category);
              const isSel = item.id === selectedId;
              const isHov = item.id === hoverId;
              const color = CATEGORY_COLORS[item.category];
              return (
                <g
                  key={item.id}
                  onClick={() => setSelectedId(item.id)}
                  onMouseEnter={() => setHoverId(item.id)}
                  onMouseLeave={() => setHoverId(null)}
                  style={{
                    cursor: "pointer",
                    opacity: visible ? 1 : 0.15,
                    transition: "opacity 200ms",
                  }}
                >
                  {isSel && (
                    <circle
                      cx={item.x}
                      cy={item.y}
                      r={18}
                      fill="url(#sel-glow)"
                    />
                  )}
                  <circle
                    cx={item.x}
                    cy={item.y}
                    r={isSel ? 7 : isHov ? 6 : 4.5}
                    fill={color}
                    stroke={isSel ? "#fff" : "rgba(255,255,255,0.25)"}
                    strokeWidth={isSel ? 2 : 1}
                  />
                  <text
                    x={item.x}
                    y={item.y - (isSel ? 12 : 9)}
                    textAnchor="middle"
                    fontSize={isSel || isHov ? 12 : 10}
                    fontWeight={isSel ? 700 : isHov ? 600 : 400}
                    fill={
                      isSel
                        ? "#ffffff"
                        : isHov
                          ? "#e6e8ee"
                          : "rgba(230,232,238,0.7)"
                    }
                    fontFamily="JetBrains Mono, monospace"
                  >
                    {item.text}
                  </text>
                </g>
              );
            })}
          </svg>
          <div className="text-[11px] text-ink-400 mt-2">
            2D projection (UMAP-style) of {corpus.length} embeddings. Real
            embedding models produce 384–3072 dimensions; we're flattening to
            two so your eyes can see structure.
          </div>
        </div>

        {/* Side: selected vector */}
        <div className="rounded-2xl bg-ink-950/70 border border-ink-800/80 p-4 flex flex-col gap-3">
          <div className="text-[11px] uppercase tracking-wider text-ink-400 font-semibold">
            Selected vector
          </div>
          <div className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-full"
              style={{ background: CATEGORY_COLORS[selected.category] }}
            />
            <span className="text-2xl font-bold font-mono">
              {selected.text}
            </span>
            <span
              className="ml-auto text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-md border"
              style={{
                color: CATEGORY_COLORS[selected.category],
                borderColor: `${CATEGORY_COLORS[selected.category]}66`,
                background: `${CATEGORY_COLORS[selected.category]}11`,
              }}
            >
              {CATEGORY_LABELS[selected.category]}
            </span>
          </div>

          <div>
            <div className="text-[10px] text-ink-400 font-mono mb-1">
              embedding ∈ ℝ^{EMBED_DIM}
            </div>
            <VectorHeatmap values={selectedEmbed} />
          </div>

          <div className="pt-2 border-t border-ink-800/60">
            <div className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold mb-2">
              Top 5 nearest neighbors
            </div>
            <div className="flex flex-col gap-1.5">
              {topNeighbors.map((nb) => (
                <button
                  key={nb.item.id}
                  onClick={() => setSelectedId(nb.item.id)}
                  className="grid grid-cols-[1fr_3.5rem] gap-2 items-center px-2 py-1.5 rounded-md hover:bg-ink-800/50 text-left"
                >
                  <span className="flex items-center gap-2 text-sm font-mono">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{
                        background: CATEGORY_COLORS[nb.item.category],
                      }}
                    />
                    {nb.item.text}
                  </span>
                  <span className="text-xs font-mono text-cyan-300 text-right">
                    {nb.similarity.toFixed(3)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* =============================== COSINE SIM =============================== */

function SimilaritySection() {
  const [a, setA] = useState<number>(8); // cat
  const [b, setB] = useState<number>(9); // dog

  const itemA = corpus[a];
  const itemB = corpus[b];
  const ea = getEmbedding(a);
  const eb = getEmbedding(b);
  const sim = cosine(ea, eb);
  const angleRad = Math.acos(Math.max(-1, Math.min(1, sim)));
  const angleDeg = (angleRad * 180) / Math.PI;

  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Sigma className="w-3 h-3" />}>Cosine similarity</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        How "close" are two pieces of text?
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Most embedding workflows use <strong>cosine similarity</strong>: the
        cosine of the angle between two vectors. It ignores magnitude and
        rewards two vectors pointing in the same direction. Pick two items
        and watch the math.
      </p>

      <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-4">
        <BlockMath
          math={String.raw`\cos(\theta) \;=\; \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert} \;\in\; [-1,\,1]`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ItemPicker
          label="A"
          accent="#22d3ee"
          selectedId={a}
          onSelect={setA}
        />
        <ItemPicker
          label="B"
          accent="#f472b6"
          selectedId={b}
          onSelect={setB}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-4">
        <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-5 flex flex-col gap-4">
          <div className="grid grid-cols-[4rem_1fr] gap-3 items-center">
            <span className="text-cyan-300 font-bold text-sm">A · {itemA.text}</span>
            <VectorHeatmap values={ea} accent="#22d3ee" />
          </div>
          <div className="grid grid-cols-[4rem_1fr] gap-3 items-center">
            <span className="text-pink-300 font-bold text-sm">B · {itemB.text}</span>
            <VectorHeatmap values={eb} accent="#f472b6" />
          </div>
          <div className="grid grid-cols-[4rem_1fr] gap-3 items-center pt-2 border-t border-ink-800/60">
            <span className="text-amber-300 font-bold text-sm">A · B</span>
            <ProductBars a={ea} b={eb} />
          </div>
        </div>

        <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-5 flex flex-col items-center justify-center gap-4">
          <div className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold">
            cosine similarity
          </div>
          <div
            className="text-5xl font-extrabold font-mono"
            style={{ color: simColor(sim) }}
          >
            {sim.toFixed(3)}
          </div>
          <AngleViz angle={angleRad} sim={sim} />
          <div className="text-xs font-mono text-ink-300">
            angle ≈ {angleDeg.toFixed(1)}°
          </div>
          <div className="text-[11px] text-ink-400 text-center max-w-[240px] leading-relaxed">
            {simVerdict(sim)}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ============================ NEAREST NEIGHBORS ============================ */

function NearestNeighborsSection() {
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<number>(40); // computer

  const matched = useMemo(() => {
    if (!query.trim()) return null;
    return findItem(query);
  }, [query]);

  const idToUse = matched?.id ?? selectedId;
  const neighbors = useMemo(() => nearestNeighbors(idToUse, 8), [idToUse]);
  const item = corpus[idToUse];

  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Search className="w-3 h-3" />}>Nearest neighbors</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        Semantic search, in one operation
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Given a query embedding, find the top-k items with the highest cosine
        similarity. That's it — that's semantic search, vector databases, and
        retrieval-augmented generation in their purest form.
      </p>

      <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-4">
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-3 items-center mb-4">
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold w-14">
              Query
            </span>
            <input
              value={query || item.text}
              onChange={(e) => {
                setQuery(e.target.value);
                const m = findItem(e.target.value);
                if (m) setSelectedId(m.id);
              }}
              placeholder="type a word from the corpus"
              className="flex-1 px-3 py-2 rounded-lg bg-ink-950/70 border border-ink-700 focus:border-cyan-400/60 focus:outline-none font-mono text-sm"
            />
          </div>
          <div className="text-[11px] text-ink-400">
            try: <em>algorithm</em>, <em>melancholic</em>, <em>plane</em>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[auto_1fr] gap-6 items-center">
          <div className="flex flex-col items-center gap-2 min-w-[140px]">
            <div className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold">
              query
            </div>
            <div
              className="px-3 py-2 rounded-xl font-mono font-bold text-base border"
              style={{
                background: `${CATEGORY_COLORS[item.category]}1c`,
                borderColor: `${CATEGORY_COLORS[item.category]}77`,
                color: "#fff",
              }}
            >
              {item.text}
            </div>
            <div className="text-[10px] text-ink-400">
              {CATEGORY_LABELS[item.category]}
            </div>
            <ArrowRight className="w-4 h-4 text-ink-500 mt-2" />
            <div className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold mt-1">
              top-k by cosine
            </div>
          </div>

          <div className="flex flex-col gap-2">
            {neighbors.map((nb, idx) => {
              const widthPct = (nb.similarity / 1) * 100;
              return (
                <motion.div
                  key={nb.item.id}
                  className="grid grid-cols-[1.5rem_8rem_1fr_4rem] gap-3 items-center"
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.04 }}
                >
                  <span className="text-[11px] font-mono text-ink-400 text-right">
                    #{idx + 1}
                  </span>
                  <span className="flex items-center gap-2 text-sm font-mono truncate">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{
                        background: CATEGORY_COLORS[nb.item.category],
                      }}
                    />
                    {nb.item.text}
                  </span>
                  <div className="h-2.5 rounded-full bg-ink-800 overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{
                        background: `linear-gradient(90deg, ${CATEGORY_COLORS[nb.item.category]}, ${CATEGORY_COLORS[nb.item.category]}66)`,
                      }}
                      initial={false}
                      animate={{ width: `${Math.max(0, widthPct)}%` }}
                      transition={{ type: "spring", stiffness: 220, damping: 24 }}
                    />
                  </div>
                  <span
                    className="text-xs font-mono text-right"
                    style={{ color: simColor(nb.similarity) }}
                  >
                    {nb.similarity.toFixed(3)}
                  </span>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-cyan-400/30 bg-cyan-400/[0.05] p-4 text-sm text-ink-200 leading-relaxed">
        <span className="text-cyan-300 font-semibold">In production →</span>{" "}
        you'd embed millions of documents once, store them in a vector DB
        (Pinecone, Weaviate, pgvector, Chroma, Qdrant…), and run approximate
        nearest-neighbor search (HNSW, IVF) to keep this fast at scale.
      </div>
    </section>
  );
}

/* ============================ ANALOGY ============================ */

function AnalogySection() {
  // king − man + woman = ?
  const [aId, setAId] = useState(48); // king
  const [bId, setBId] = useState(50); // man
  const [cId, setCId] = useState(51); // woman

  const result = useMemo(() => analogy(aId, bId, cId, 4), [aId, bId, cId]);
  const a = corpus[aId];
  const b = corpus[bId];
  const c = corpus[cId];

  const ALL_ROYALTY = corpus.filter((c) => c.category === "royalty");

  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Sparkles className="w-3 h-3" />}>Vector arithmetic</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        Meaning becomes <em>direction</em>
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        One of the most famous results in NLP: <em>king</em> − <em>man</em> +{" "}
        <em>woman</em> ≈ <em>queen</em>. The vector pointing from "man" to
        "woman" turns out to be roughly the "gender" axis — subtract it from
        "king" and you land on "queen". Try other pairs.
      </p>

      <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-5">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-6">
          {/* Controls */}
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-2 flex-wrap">
              <AnalogyPicker
                label="A"
                color="#c084fc"
                value={aId}
                onChange={setAId}
                options={ALL_ROYALTY}
              />
              <span className="text-xl text-ink-400">−</span>
              <AnalogyPicker
                label="B"
                color="#f472b6"
                value={bId}
                onChange={setBId}
                options={ALL_ROYALTY}
              />
              <span className="text-xl text-ink-400">+</span>
              <AnalogyPicker
                label="C"
                color="#22d3ee"
                value={cId}
                onChange={setCId}
                options={ALL_ROYALTY}
              />
              <span className="text-xl text-ink-400">=</span>
              <span className="px-3 py-2 rounded-lg border border-emerald-400/50 bg-emerald-400/10 font-mono text-emerald-200">
                ?
              </span>
            </div>

            <div className="rounded-lg bg-ink-950/70 border border-ink-800/80 p-3 text-sm leading-relaxed">
              <span className="font-mono">
                <span className="text-purple-300">{a.text}</span> −{" "}
                <span className="text-pink-300">{b.text}</span> +{" "}
                <span className="text-cyan-300">{c.text}</span> ={" "}
                <span className="text-emerald-300 font-bold">
                  {result[0]?.item.text}
                </span>
              </span>
            </div>

            {/* Vector math heatmaps */}
            <div className="flex flex-col gap-2 text-[10px] font-mono">
              <HeatRowLabeled label={a.text} color="#c084fc" values={getEmbedding(aId)} />
              <div className="text-center text-ink-400 text-base">−</div>
              <HeatRowLabeled label={b.text} color="#f472b6" values={getEmbedding(bId)} />
              <div className="text-center text-ink-400 text-base">+</div>
              <HeatRowLabeled label={c.text} color="#22d3ee" values={getEmbedding(cId)} />
              <div className="text-center text-ink-400 text-base">=</div>
              <HeatRowLabeled
                label="result"
                color="#34d399"
                values={getEmbedding(aId).map(
                  (x, i) => x - getEmbedding(bId)[i] + getEmbedding(cId)[i]
                )}
              />
            </div>
          </div>

          {/* Geometric viz: 2D parallelogram */}
          <div className="flex flex-col gap-3">
            <div className="text-[11px] uppercase tracking-wider text-ink-400 font-semibold">
              Visualized as a parallelogram
            </div>
            <AnalogyDiagram a={a} b={b} c={c} resultText={result[0]?.item.text ?? "?"} />
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-400 font-semibold mb-2">
                Top candidates
              </div>
              <div className="flex flex-col gap-1.5">
                {result.map((r, idx) => (
                  <div
                    key={r.item.id}
                    className="grid grid-cols-[1.5rem_1fr_4rem] gap-2 items-center"
                  >
                    <span className="text-[11px] font-mono text-ink-400 text-right">
                      #{idx + 1}
                    </span>
                    <span className="font-mono text-sm flex items-center gap-2">
                      {idx === 0 && (
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                      )}
                      {r.item.text}
                    </span>
                    <span
                      className="text-xs font-mono text-right"
                      style={{ color: simColor(r.similarity) }}
                    >
                      {r.similarity.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="text-[11px] text-ink-400 max-w-3xl">
        Real embeddings aren't <em>perfectly</em> linear like this — modern
        models are richer and the famous analogies sometimes break. But the
        intuition holds: meaningful axes really do emerge, and you can do
        first-class algebra on them.
      </div>
    </section>
  );
}

/* ============================ HOW EMBEDDINGS ARE MADE ============================ */

function HowEmbeddingsAreMade() {
  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Workflow className="w-3 h-3" />}>Under the hood</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        How a sentence becomes a single vector
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Most modern embedders are <strong>BERT-style encoders</strong>{" "}
        finetuned with a contrastive objective: pairs of "similar" texts
        should land close, "different" texts far apart. The architecture is
        almost identical to a regular transformer encoder; the magic is in
        the <em>pooling</em> step that compresses a sequence of token vectors
        into one.
      </p>

      <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 p-5">
        <PipelineDiagram />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <PoolingCard
          name="[CLS] token"
          desc="Use the encoder's [CLS] token vector as the sentence representation. BERT's classic recipe — but [CLS] is only good once you fine-tune for it."
          example="bge-large, original SBERT"
          color="#22d3ee"
        />
        <PoolingCard
          name="Mean pooling"
          desc="Average every token's final vector (masked by attention). Robust, simple, and almost always strong — the default for SBERT and E5."
          example="all-MiniLM, E5, nomic-embed"
          color="#a78bfa"
        />
        <PoolingCard
          name="Last-token (decoder)"
          desc="For decoder-only models (think GPT, Mistral, Qwen), take the last token's hidden state — the only one that has 'seen' the full sequence."
          example="GTE-Qwen, e5-mistral-7b"
          color="#f472b6"
        />
      </div>

      <div className="rounded-xl bg-ink-950/70 border border-ink-800/80 p-4">
        <div className="text-[11px] uppercase tracking-wider text-ink-300 font-semibold mb-3">
          Training: contrastive learning (in one picture)
        </div>
        <ContrastiveDiagram />
      </div>
    </section>
  );
}

/* ============================ MODEL COMPARISON ============================ */

function ModelComparisonSection() {
  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Layers className="w-3 h-3" />}>Modern models</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        The current landscape
      </h3>
      <p className="text-ink-300 max-w-3xl leading-relaxed">
        Embedding models span six orders of magnitude in size — from the
        22M-parameter MiniLM that runs on a CPU to the closed-source giants
        behind text-embedding-3-large. All speak the same interface: text in,
        vector out.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {embeddingModels.map((m) => (
          <ModelCard key={m.name} model={m} />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-xl border border-ink-800/80 bg-ink-950/40 p-4 text-sm leading-relaxed">
          <div className="font-semibold text-white mb-1.5 flex items-center gap-2">
            <Boxes className="w-4 h-4 text-cyan-300" />
            What's <em>Matryoshka</em>?
          </div>
          <p className="text-ink-300">
            A training trick where the model is taught so that the{" "}
            <em>first</em> 256 dims (or 64, or 128) of the 1536-dim vector are
            already a usable embedding. You can truncate at runtime to save
            storage and search cost — a 6× compression with almost no quality
            loss. text-embedding-3 and nomic-embed both ship this.
          </p>
        </div>
        <div className="rounded-xl border border-ink-800/80 bg-ink-950/40 p-4 text-sm leading-relaxed">
          <div className="font-semibold text-white mb-1.5 flex items-center gap-2">
            <Globe2 className="w-4 h-4 text-emerald-300" />
            Which one should I use?
          </div>
          <ul className="text-ink-300 list-disc pl-5 space-y-1">
            <li><strong>RAG on docs:</strong> bge-large or text-embedding-3-large.</li>
            <li><strong>Low budget / edge:</strong> all-MiniLM (CPU-friendly).</li>
            <li><strong>Open-everything stack:</strong> nomic-embed.</li>
            <li><strong>Asymmetric search (query vs doc):</strong> E5 (it's trained for that).</li>
          </ul>
        </div>
      </div>
    </section>
  );
}

/* ============================ WHY MATTER ============================ */

function WhyEmbeddingsMatter() {
  return (
    <section className="flex flex-col gap-4">
      <Eyebrow icon={<Database className="w-3 h-3" />}>Where they go</Eyebrow>
      <h3 className="text-xl lg:text-2xl font-extrabold tracking-tight">
        What embeddings unlock
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <UseCaseCard
          color="#22d3ee"
          title="Retrieval-Augmented Generation"
          body="Embed your documents → store in a vector DB → embed the user's query → fetch the top-k → stuff into the LLM's context. That's RAG, and it's how almost every production LLM app handles knowledge today."
        />
        <UseCaseCard
          color="#a78bfa"
          title="Semantic Search"
          body="Move beyond keyword match. A query for 'how do I cancel my subscription' will find FAQ entries about 'ending my plan' or 'removing my account' even when there's no word overlap."
        />
        <UseCaseCard
          color="#f472b6"
          title="Classification / Clustering"
          body="An embedding is a powerful feature vector. Throw a linear classifier on top for topic detection, intent classification, or zero-shot labeling by comparing to label embeddings."
        />
        <UseCaseCard
          color="#34d399"
          title="Deduplication & Diversity"
          body="Spot near-duplicate documents, deduplicate training data, sample diverse content. The cosine distance gives you a continuous knob between 'identical' and 'unrelated'."
        />
      </div>

      <div className="rounded-2xl border border-ink-800/80 bg-gradient-to-br from-cyan-500/[0.05] to-fuchsia-500/[0.05] p-5">
        <div className="text-[11px] uppercase tracking-wider text-ink-300 font-semibold mb-1.5">
          The big takeaway
        </div>
        <p className="text-ink-100 leading-relaxed">
          Embeddings are how machines reduce <em>arbitrary text</em> to{" "}
          <em>arithmetic</em>. Once you can add, subtract, average, and
          measure angles between meanings, you've turned natural-language
          problems into linear algebra — and linear algebra is something we
          know how to do very, very fast.
        </p>
      </div>
    </section>
  );
}

/* =================================================================== */
/* ===================== SHARED HELPER COMPONENTS ==================== */
/* =================================================================== */

function Eyebrow({
  children,
  icon,
}: {
  children: React.ReactNode;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-cyan-300 font-semibold">
      {icon ?? <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />}
      {children}
    </div>
  );
}

function RoleCard({
  color,
  icon,
  title,
  body,
}: {
  color: string;
  icon: React.ReactNode;
  title: string;
  body: string;
}) {
  return (
    <div
      className="rounded-2xl border p-4 flex flex-col gap-2"
      style={{
        background: `linear-gradient(180deg, ${color}1a, transparent)`,
        borderColor: `${color}55`,
      }}
    >
      <div className="flex items-center gap-2">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: `${color}33`, color }}
        >
          {icon}
        </div>
        <div className="font-bold">{title}</div>
      </div>
      <p className="text-sm text-ink-300 leading-relaxed">{body}</p>
    </div>
  );
}

function VectorHeatmap({
  values,
  accent,
  height = 22,
}: {
  values: number[];
  accent?: string;
  height?: number;
}) {
  const max = Math.max(...values.map((v) => Math.abs(v)), 0.001);
  return (
    <div
      className="flex rounded-md overflow-hidden border border-ink-800/80"
      style={{ height }}
    >
      {values.map((v, i) => {
        const t = Math.abs(v) / max;
        let bg: string;
        if (accent) {
          const alpha = Math.round(50 + 200 * t);
          bg =
            v < 0
              ? `${accent}${Math.round(40 + 80 * t).toString(16).padStart(2, "0")}`
              : `${accent}${alpha.toString(16).padStart(2, "0")}`;
        } else {
          // pink (positive) ←→ cyan (negative)
          if (v >= 0) {
            bg = `rgba(244, 114, 182, ${0.15 + t * 0.85})`;
          } else {
            bg = `rgba(34, 211, 238, ${0.15 + t * 0.85})`;
          }
        }
        return (
          <div
            key={i}
            style={{ background: bg, width: `${100 / values.length}%` }}
            title={`d${i}: ${v.toFixed(3)}`}
          />
        );
      })}
    </div>
  );
}

function HeatRowLabeled({
  label,
  values,
  color,
}: {
  label: string;
  values: number[];
  color: string;
}) {
  return (
    <div className="grid grid-cols-[5rem_1fr] gap-2 items-center">
      <span
        className="text-[11px] font-mono truncate"
        style={{ color }}
      >
        {label}
      </span>
      <VectorHeatmap values={values} accent={color} height={14} />
    </div>
  );
}

function ProductBars({ a, b }: { a: number[]; b: number[] }) {
  const products = a.map((x, i) => x * b[i]);
  const max = Math.max(...products.map((v) => Math.abs(v)), 0.001);
  return (
    <div
      className="flex gap-[2px] rounded-md overflow-hidden"
      style={{ height: 28 }}
    >
      {products.map((p, i) => {
        const t = Math.abs(p) / max;
        const color = p >= 0 ? "#34d399" : "#f87171";
        const height = `${4 + t * 96}%`;
        return (
          <div
            key={i}
            className="flex-1 flex flex-col justify-end relative"
            title={`a[${i}]·b[${i}] = ${p.toFixed(3)}`}
          >
            <div
              style={{
                height,
                background: color,
                opacity: 0.4 + t * 0.6,
              }}
            />
          </div>
        );
      })}
    </div>
  );
}

function AngleViz({ angle, sim }: { angle: number; sim: number }) {
  const size = 140;
  const cx = size / 2;
  const cy = size / 2;
  const r = 52;
  const ax = cx + r;
  const ay = cy;
  const bx = cx + r * Math.cos(angle);
  const by = cy - r * Math.sin(angle);
  return (
    <svg width={size} height={size}>
      <circle
        cx={cx}
        cy={cy}
        r={r}
        fill="none"
        stroke="rgba(255,255,255,0.08)"
        strokeDasharray="2 4"
      />
      <line x1={cx} y1={cy} x2={ax} y2={ay} stroke="#22d3ee" strokeWidth={2.5} />
      <circle cx={ax} cy={ay} r={4} fill="#22d3ee" />
      <line x1={cx} y1={cy} x2={bx} y2={by} stroke="#f472b6" strokeWidth={2.5} />
      <circle cx={bx} cy={by} r={4} fill="#f472b6" />
      <path
        d={`M ${cx + 18} ${cy} A 18 18 0 ${angle > Math.PI ? 1 : 0} 0 ${cx + 18 * Math.cos(angle)} ${cy - 18 * Math.sin(angle)}`}
        fill="none"
        stroke={simColor(sim)}
        strokeWidth={1.5}
      />
      <circle cx={cx} cy={cy} r={3} fill="#ffffff" />
    </svg>
  );
}

function ItemPicker({
  label,
  accent,
  selectedId,
  onSelect,
}: {
  label: string;
  accent: string;
  selectedId: number;
  onSelect: (id: number) => void;
}) {
  const selected = corpus[selectedId];
  return (
    <div className="rounded-xl bg-ink-950/60 border border-ink-800/80 p-3">
      <div className="flex items-center gap-2 mb-2.5">
        <div
          className="w-6 h-6 rounded-md flex items-center justify-center font-mono font-bold text-xs"
          style={{ background: `${accent}33`, color: accent, border: `1px solid ${accent}66` }}
        >
          {label}
        </div>
        <span className="text-sm font-mono">{selected.text}</span>
        <span
          className="ml-auto text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-md border"
          style={{
            color: CATEGORY_COLORS[selected.category],
            borderColor: `${CATEGORY_COLORS[selected.category]}66`,
            background: `${CATEGORY_COLORS[selected.category]}11`,
          }}
        >
          {CATEGORY_LABELS[selected.category]}
        </span>
      </div>
      <div className="flex flex-wrap gap-1 max-h-[120px] overflow-y-auto pr-1">
        {corpus.map((item) => {
          const isSel = item.id === selectedId;
          return (
            <button
              key={item.id}
              onClick={() => onSelect(item.id)}
              className={[
                "px-2 py-0.5 rounded text-[11px] font-mono border transition-all",
                isSel
                  ? "text-white"
                  : "text-ink-400 border-ink-800 hover:text-ink-200",
              ].join(" ")}
              style={{
                background: isSel ? `${accent}33` : "transparent",
                borderColor: isSel ? accent : undefined,
              }}
            >
              {item.text}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function AnalogyPicker({
  label,
  color,
  value,
  onChange,
  options,
}: {
  label: string;
  color: string;
  value: number;
  onChange: (id: number) => void;
  options: EmbeddingItem[];
}) {
  const sel = corpus[value];
  return (
    <div className="flex items-center gap-1.5">
      <span
        className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border"
        style={{ color, borderColor: `${color}66`, background: `${color}1a` }}
      >
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="px-2 py-1.5 rounded-lg bg-ink-950/70 border border-ink-700 font-mono text-sm focus:outline-none focus:border-cyan-400/60"
        style={{ color }}
      >
        {options.map((o) => (
          <option key={o.id} value={o.id} className="bg-ink-950">
            {o.text}
          </option>
        ))}
      </select>
      {/* sel only used to keep TS happy in case of debugging */}
      <span className="hidden">{sel.text}</span>
    </div>
  );
}

function AnalogyDiagram({
  a,
  b,
  c,
  resultText,
}: {
  a: EmbeddingItem;
  b: EmbeddingItem;
  c: EmbeddingItem;
  resultText: string;
}) {
  // Compute the predicted 2D result by parallelogram: a − b + c
  const rx = a.x - b.x + c.x;
  const ry = a.y - b.y + c.y;
  const pad = 30;
  const xs = [a.x, b.x, c.x, rx];
  const ys = [a.y, b.y, c.y, ry];
  const minX = Math.min(...xs) - pad;
  const minY = Math.min(...ys) - pad;
  const maxX = Math.max(...xs) + pad;
  const maxY = Math.max(...ys) + pad;
  const w = maxX - minX;
  const h = maxY - minY;

  return (
    <svg
      viewBox={`${minX} ${minY} ${w} ${h}`}
      className="w-full rounded-lg bg-ink-950/40 border border-ink-800/80"
      style={{ aspectRatio: `${w} / ${h}` }}
    >
      <defs>
        <marker
          id="arrow-end"
          markerWidth="8"
          markerHeight="8"
          refX="6"
          refY="4"
          orient="auto"
        >
          <path d="M0,0 L0,8 L8,4 z" fill="#34d399" />
        </marker>
      </defs>

      {/* Parallelogram sides */}
      <line
        x1={b.x}
        y1={b.y}
        x2={a.x}
        y2={a.y}
        stroke="#c084fc"
        strokeWidth={1.5}
        strokeDasharray="4 3"
      />
      <line
        x1={c.x}
        y1={c.y}
        x2={rx}
        y2={ry}
        stroke="#34d399"
        strokeWidth={2}
        markerEnd="url(#arrow-end)"
      />
      <line
        x1={b.x}
        y1={b.y}
        x2={c.x}
        y2={c.y}
        stroke="rgba(255,255,255,0.1)"
        strokeWidth={1}
      />
      <line
        x1={a.x}
        y1={a.y}
        x2={rx}
        y2={ry}
        stroke="rgba(255,255,255,0.1)"
        strokeWidth={1}
      />

      {/* Dots */}
      {[
        { item: a, color: "#c084fc", label: a.text },
        { item: b, color: "#f472b6", label: b.text },
        { item: c, color: "#22d3ee", label: c.text },
      ].map((d, i) => (
        <g key={i}>
          <circle
            cx={d.item.x}
            cy={d.item.y}
            r={6}
            fill={d.color}
            stroke="#fff"
            strokeWidth={1}
          />
          <text
            x={d.item.x}
            y={d.item.y - 12}
            textAnchor="middle"
            fontSize={11}
            fontWeight={700}
            fill={d.color}
            fontFamily="JetBrains Mono, monospace"
          >
            {d.label}
          </text>
        </g>
      ))}

      {/* Result dot */}
      <circle
        cx={rx}
        cy={ry}
        r={7}
        fill="#34d399"
        stroke="#fff"
        strokeWidth={2}
      />
      <text
        x={rx}
        y={ry - 14}
        textAnchor="middle"
        fontSize={12}
        fontWeight={800}
        fill="#34d399"
        fontFamily="JetBrains Mono, monospace"
      >
        = {resultText}
      </text>
    </svg>
  );
}

function PipelineDiagram() {
  return (
    <svg viewBox="0 0 720 220" className="w-full block">
      <defs>
        <marker
          id="pipe-arrow"
          markerWidth="10"
          markerHeight="10"
          refX="7"
          refY="3"
          orient="auto"
        >
          <path d="M0,0 L0,6 L7,3 z" fill="#7c5cff" />
        </marker>
      </defs>

      {/* 1. Text */}
      <PipeBox x={10} y={80} w={120} h={60} color="#22d3ee" label="text" sub='"a black cat"' />
      <PipeArrow x1={130} y1={110} x2={170} y2={110} />

      {/* 2. Tokenizer */}
      <PipeBox x={170} y={80} w={120} h={60} color="#a78bfa" label="tokenizer" sub="WordPiece / BPE" />
      <PipeArrow x1={290} y1={110} x2={330} y2={110} />

      {/* 3. Encoder (multi-layer) */}
      <PipeBox
        x={330}
        y={50}
        w={170}
        h={120}
        color="#f472b6"
        label="transformer encoder"
        sub="N layers · self-attention + FFN"
        stack
      />
      <PipeArrow x1={500} y1={110} x2={540} y2={110} />

      {/* 4. Pooling */}
      <PipeBox x={540} y={80} w={90} h={60} color="#fbbf24" label="pooling" sub="mean / [CLS]" />
      <PipeArrow x1={630} y1={110} x2={665} y2={110} />

      {/* 5. Final vector */}
      <g>
        <rect
          x={665}
          y={70}
          width={50}
          height={80}
          rx={10}
          fill="#34d39922"
          stroke="#34d39988"
        />
        {Array.from({ length: 8 }).map((_, i) => (
          <rect
            key={i}
            x={672}
            y={78 + i * 8}
            width={36}
            height={5}
            fill="#34d399"
            opacity={0.3 + 0.7 * Math.random()}
          />
        ))}
        <text
          x={690}
          y={170}
          textAnchor="middle"
          fontSize={11}
          fontWeight={700}
          fill="#34d399"
          fontFamily="JetBrains Mono, monospace"
        >
          ℝ^d
        </text>
      </g>
    </svg>
  );
}

function PipeBox({
  x,
  y,
  w,
  h,
  color,
  label,
  sub,
  stack,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
  label: string;
  sub?: string;
  stack?: boolean;
}) {
  return (
    <g>
      {stack && (
        <>
          <rect x={x + 6} y={y + 6} width={w} height={h} rx={12} fill={`${color}11`} stroke={`${color}33`} />
          <rect x={x + 3} y={y + 3} width={w} height={h} rx={12} fill={`${color}1a`} stroke={`${color}55`} />
        </>
      )}
      <rect x={x} y={y} width={w} height={h} rx={12} fill={`${color}22`} stroke={`${color}88`} />
      <text
        x={x + w / 2}
        y={y + h / 2 - (sub ? 4 : 0)}
        textAnchor="middle"
        fontSize={13}
        fontWeight={700}
        fill={color}
      >
        {label}
      </text>
      {sub && (
        <text
          x={x + w / 2}
          y={y + h / 2 + 14}
          textAnchor="middle"
          fontSize={10}
          fill="#c2c7d4"
          fontFamily="JetBrains Mono, monospace"
        >
          {sub}
        </text>
      )}
    </g>
  );
}

function PipeArrow({
  x1,
  y1,
  x2,
  y2,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}) {
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke="#7c5cff"
      strokeWidth={1.5}
      markerEnd="url(#pipe-arrow)"
    />
  );
}

function PoolingCard({
  name,
  desc,
  example,
  color,
}: {
  name: string;
  desc: string;
  example: string;
  color: string;
}) {
  return (
    <div
      className="rounded-xl border p-4 flex flex-col gap-2"
      style={{
        background: `${color}0a`,
        borderColor: `${color}55`,
      }}
    >
      <div className="font-bold" style={{ color }}>
        {name}
      </div>
      <p className="text-sm text-ink-300 leading-relaxed">{desc}</p>
      <div className="text-[10px] text-ink-400 font-mono pt-1">
        e.g. {example}
      </div>
    </div>
  );
}

function ContrastiveDiagram() {
  // Anchor → positive (close), negative (far)
  return (
    <svg viewBox="0 0 720 160" className="w-full">
      <defs>
        <marker
          id="pull"
          markerWidth="8"
          markerHeight="8"
          refX="6"
          refY="4"
          orient="auto"
        >
          <path d="M0,0 L0,8 L8,4 z" fill="#34d399" />
        </marker>
        <marker
          id="push"
          markerWidth="8"
          markerHeight="8"
          refX="6"
          refY="4"
          orient="auto"
        >
          <path d="M0,0 L0,8 L8,4 z" fill="#f87171" />
        </marker>
      </defs>

      {/* Anchor */}
      <circle cx={360} cy={80} r={14} fill="#7c5cff" stroke="#fff" strokeWidth={2} />
      <text x={360} y={114} textAnchor="middle" fontSize={11} fill="#fff" fontFamily="JetBrains Mono, monospace" fontWeight={700}>anchor</text>
      <text x={360} y={128} textAnchor="middle" fontSize={10} fill="#a3a8b8">"a happy dog"</text>

      {/* Positive */}
      <circle cx={520} cy={62} r={11} fill="#34d399" stroke="#fff" strokeWidth={1.5} />
      <text x={520} y={42} textAnchor="middle" fontSize={11} fill="#34d399" fontWeight={700}>positive</text>
      <text x={520} y={28} textAnchor="middle" fontSize={10} fill="#a3a8b8">"a joyful puppy"</text>
      <line x1={376} y1={75} x2={508} y2={64} stroke="#34d399" strokeWidth={2} markerEnd="url(#pull)" />
      <text x={440} y={62} textAnchor="middle" fontSize={11} fill="#34d399" fontWeight={700}>pull closer</text>

      {/* Negative */}
      <circle cx={520} cy={120} r={11} fill="#f87171" stroke="#fff" strokeWidth={1.5} />
      <text x={520} y={148} textAnchor="middle" fontSize={11} fill="#f87171" fontWeight={700}>negative</text>
      <text x={520} y={158} textAnchor="middle" fontSize={10} fill="#a3a8b8" opacity={0.7}>"quarterly earnings"</text>
      <line x1={376} y1={88} x2={508} y2={118} stroke="#f87171" strokeWidth={2} markerEnd="url(#push)" strokeDasharray="4 3" />
      <text x={440} y={120} textAnchor="middle" fontSize={11} fill="#f87171" fontWeight={700}>push apart</text>

      {/* Left side text */}
      <text x={160} y={70} textAnchor="middle" fontSize={11} fill="#e6e8ee" fontWeight={700}>InfoNCE / contrastive loss</text>
      <text x={160} y={92} textAnchor="middle" fontSize={10} fill="#a3a8b8">maximize cos(anchor, positive)</text>
      <text x={160} y={108} textAnchor="middle" fontSize={10} fill="#a3a8b8">minimize cos(anchor, negatives)</text>
    </svg>
  );
}

function ModelCard({ model: m }: { model: typeof embeddingModels[number] }) {
  return (
    <div
      className="rounded-2xl border p-4 flex flex-col gap-3 relative overflow-hidden"
      style={{
        background: `linear-gradient(180deg, ${m.accent}1a, transparent)`,
        borderColor: `${m.accent}55`,
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-base font-bold leading-tight">{m.name}</div>
          <div className="text-[11px] text-ink-400 mt-0.5">
            {m.family} · {m.releasedYear}
          </div>
        </div>
        <span
          className="text-[10px] font-mono px-1.5 py-0.5 rounded border"
          style={{
            color: m.openSource ? "#34d399" : "#fbbf24",
            borderColor: m.openSource ? "#34d39955" : "#fbbf2455",
            background: m.openSource ? "#34d39911" : "#fbbf2411",
          }}
        >
          {m.openSource ? "open" : "closed"}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <MetricMini label="dims" value={m.dims.toLocaleString()} />
        <MetricMini
          label="ctx"
          value={`${m.ctxTokens >= 1000 ? `${(m.ctxTokens / 1000).toFixed(m.ctxTokens === 8191 ? 1 : 0)}k` : m.ctxTokens}`}
        />
        <MetricMini label="pool" value={m.pooling} />
      </div>

      {m.dimsAlt && (
        <div className="text-[10px] text-ink-400">
          Matryoshka → truncate to{" "}
          <span className="text-cyan-300 font-mono">{m.dimsAlt}</span> dims
        </div>
      )}

      <div className="text-[11px] text-ink-400 italic leading-relaxed">
        {m.trainObjective}
      </div>

      <div className="text-[12.5px] text-ink-200 leading-relaxed">
        {m.strength}
      </div>
    </div>
  );
}

function MetricMini({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-ink-950/60 border border-ink-800/80 p-1.5 text-center">
      <div className="text-[9px] uppercase tracking-wider text-ink-400">
        {label}
      </div>
      <div className="text-xs font-mono font-bold text-white">{value}</div>
    </div>
  );
}

function UseCaseCard({
  color,
  title,
  body,
}: {
  color: string;
  title: string;
  body: string;
}) {
  return (
    <div
      className="rounded-2xl border p-5"
      style={{
        background: `${color}0a`,
        borderColor: `${color}55`,
      }}
    >
      <div className="font-bold text-white mb-2 text-base">{title}</div>
      <p className="text-sm text-ink-300 leading-relaxed">{body}</p>
    </div>
  );
}

/* =========================== little utils =========================== */

function simColor(s: number): string {
  // 1.0 → emerald, 0 → cyan, -1 → red
  if (s >= 0.7) return "#34d399";
  if (s >= 0.4) return "#22d3ee";
  if (s >= 0) return "#a78bfa";
  if (s >= -0.4) return "#fbbf24";
  return "#f87171";
}

function simVerdict(s: number): string {
  if (s >= 0.85) return "Practically synonyms in this space.";
  if (s >= 0.6) return "Closely related — same neighborhood of meaning.";
  if (s >= 0.3) return "Loosely related; might share context occasionally.";
  if (s >= 0) return "Mostly unrelated. The vectors are nearly orthogonal.";
  return "Actively dissimilar — they point in opposing directions.";
}

// Unused-import suppressor for lucide icons we want available to viewers.
const _icons = [ArrowDown];
void _icons;
