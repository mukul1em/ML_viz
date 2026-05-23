# ML Viz

An interactive lab to **visualize the functions that power modern ML and deep learning**.

Drag sliders, change temperatures, randomize inputs — and watch how classical activations, losses, and optimizers behave. Built for students, engineers, and anyone who learns by playing.

## Available now

- **Softmax** — turn logits into a probability distribution. Tweak each logit, change the temperature, and see the math react in real time (entropy, top-class confidence, step-by-step table, numerically stable form).

## Coming soon

- Sigmoid · Tanh · ReLU & friends (Leaky ReLU, GELU, SiLU)
- Cross-Entropy loss
- Gradient Descent variants (SGD, Momentum, Adam) on a loss landscape
- Batch Normalization

## Stack

- Vite + React 19 + TypeScript
- Tailwind CSS 3
- Recharts for charts
- KaTeX for math typesetting
- Framer Motion for animations
- React Router

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:5173`.

### Optional: Python math API

The Softmax page has an "advanced math lab" that can be powered by a real
Python (numpy) backend. Visualizations gracefully fall back to client-side
math if it's not running, but having it on lets you also `curl` the
endpoints directly and explore in Swagger UI.

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
  components/        Reusable UI (Layout, Slider, SoftmaxViz)
  pages/             Routed pages (Home, SoftmaxPage, NotFound)
  data/functions.ts  Registry of all functions (drives nav + catalog)
  lib/softmax.ts     Pure-math softmax implementation
  App.tsx            Router
  main.tsx           Entry
```

### Adding a new visualization

1. Add an entry in `src/data/functions.ts` (set `available: true` and a `path`).
2. Create `src/pages/<Name>Page.tsx` and a `src/components/<Name>Viz.tsx`.
3. Register the route in `src/App.tsx`.
