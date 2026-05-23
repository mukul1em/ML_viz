# ML Viz

An interactive lab to **visualize the functions, losses, and architectures that power modern ML and deep learning**.

Drag sliders, click curves, change temperatures — and watch how classical activations, losses, attention, and full LLM stacks behave. Every page is math-first: KaTeX equations next to live SVG plots, with deterministic mock data so the visuals match the formulas exactly.

## Available

### ML functions

- **Softmax** — turn logits into a probability distribution. Temperature, entropy, Jacobian, cross-entropy, categorical sampling, numerical-stability comparison, and an optional Python (NumPy) backend.
- **Sigmoid** `σ(x) = 1 / (1 + e⁻ˣ)` — interactive σ + σ′ plot, properties grid, BCE loss, logit inverse, family comparison (σ · tanh · hard-σ).
- **ReLU & friends** — `max(0, x)` plus Leaky · ELU · GELU (exact via `erf`) · SiLU / Swish · Mish · Softplus. Toggle variants on/off; cursor scrubs both `f(x)` and `f ′(x)`. Dying-ReLU + variant cheatsheet.
- **Tanh** `tanh(x)` — derivative `1 − tanh²`, inverse `artanh`, the `tanh(x) = 2σ(2x) − 1` identity, family comparison (tanh · hard-tanh · LeCun's `1.7159·tanh(⅔x)`).
- **Cross-Entropy** `H(p, q)` — interactive categorical CE on a 4-class softmax (probabilities & `q − y` gradient bars), binary CE on a sigmoid, the `−log q_y` confidence curve, KL decomposition `H(p, q) = H(p) + D_KL(p ‖ q)` with editable `p` and `q`.

### Attention

- **Q, K, V Attention** — step-by-step trace of a single query's attention over a sentence, full N×N attention heatmap with optional causal mask and `1/√dₖ` scaling, "why three projections?" diagrams.

### LLM models

- **GPT** — decoder-only graph (token → position → embedding → ×N(LayerNorm → Causal Self-Attention → FFN → Residual) → LM head → softmax). Click any node for math, mock data, and interactive demos.
- **BERT** — encoder-only counterpart: WordPiece tokenizer, three embeddings (token + position + segment), post-norm, bidirectional self-attention, MLM and NSP heads.
- **Qwen** — modern decoder stack with RMSNorm · RoPE · Grouped-Query Attention · SwiGLU FFN · tied LM head. Each signature feature has its own interactive sub-visual.
- **Embedding Models** — semantic map of a curated corpus, cosine similarity, k-nearest neighbors, vector-arithmetic analogies ("king − man + woman = queen"), pooling strategies, contrastive learning, comparison of OpenAI / BGE / E5 / SBERT / Nomic.

## Coming soon

- Gradient Descent variants (SGD, Momentum, Adam) on a loss landscape
- Batch / Layer / RMS Normalization side-by-side
- LLaMA

## Stack

- Vite + React 19 + TypeScript
- Tailwind CSS 3
- KaTeX for math typesetting (via a thin `Tex` wrapper)
- Recharts where convenient; custom SVG everywhere math precision matters
- Framer Motion for transitions
- React Router for navigation
- **Backend (optional):** FastAPI + NumPy for the Softmax advanced lab

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:5173`.

### Optional: Python math API

The Softmax page has an "advanced math lab" that can be powered by a real
Python (NumPy) backend. Visualizations gracefully fall back to client-side
math if the server isn't running; with it on you can also `curl` the
endpoints directly and explore Swagger UI.

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open <http://localhost:8000/docs> for the interactive API explorer. See
[`backend/README.md`](backend/README.md) for endpoint details.

## Build

```bash
npm run build
npm run preview
```

## Project structure

```
src/
  components/
    Layout.tsx               App shell + grouped sidebar (functions + LLMs)
    Tex.tsx                  KaTeX wrapper (use String.raw for LaTeX!)
    SoftmaxViz.tsx           Softmax interactive lab
    SoftmaxAdvancedLab.tsx   Jacobian · temperature sweep · sampling · stability
    SigmoidViz.tsx
    ReLUViz.tsx
    TanhViz.tsx
    CrossEntropyViz.tsx
    QKVViz.tsx
    EmbeddingsViz.tsx
    gpt/                     GPT graph · sidebar · per-component detail
    bert/                    BERT graph · sidebar · per-component detail
    qwen/                    Qwen graph · sidebar · per-component detail
  pages/                     One thin page per route
  data/
    functions.ts             Registry of ML functions (drives sidebar + home)
    models.ts                Registry of LLM models
    gptComponents.ts         Per-component metadata for GPT graph
    bertComponents.ts        ... for BERT
    qwenComponents.ts        ... for Qwen
    embeddingCorpus.ts       Hand-curated 2D semantic map
    embeddingModels.ts       Specs for modern embedding models
  lib/
    softmax.ts               Softmax + Jacobian + sampling + CE
    sigmoid.ts               σ + σ′ + logit + tanh identity + BCE
    relu.ts                  All ReLU-family activations + derivatives
    tanh.ts                  tanh + artanh + hardTanh + scaledTanh
    crossEntropy.ts          H(p), H(p,q), KL(p‖q), stable BCE
    qkvDemo.ts               Mock Q/K/V projections, traces, full matrices
    embeddingDemo.ts         16-D mock embeddings, cosine, analogies
    qwenDemo.ts              RoPE, GQA layout, SwiGLU, RMSNorm, param counts
    api.ts                   FastAPI client + useApiStatus hook
  App.tsx                    Router
  main.tsx                   Entry

backend/
  main.py                    FastAPI app + endpoints
  softmax_math.py            NumPy implementations
  requirements.txt
  README.md
```

### Adding a new visualization

1. Add an entry in `src/data/functions.ts` (set `available: true` and a `path`).
2. Create `src/lib/<name>.ts` for the math.
3. Create `src/components/<Name>Viz.tsx` for the visualization and `src/pages/<Name>Page.tsx` for the route wrapper.
4. Register the route in `src/App.tsx`.
5. **For LaTeX strings**, always use `String.raw` template literals (e.g. `` String.raw`\frac{a}{b}` ``) so JavaScript doesn't eat the backslashes.

### Adding a new LLM model

Same flow as above, but use `src/data/models.ts` and consider adding a graph
canvas + sidebar + detail panel under `src/components/<model>/`.
