"""ML Viz Python API.

A small FastAPI service that exposes the real numerical functions backing
the visualizations. Open http://localhost:8000/docs after starting to play
with every endpoint interactively.

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import softmax_math as sm

app = FastAPI(
    title="ML Viz API",
    version="0.1.0",
    description=(
        "Real numerical reference implementations powering the ML Viz frontend.\n\n"
        "Everything is plain numpy — use the endpoints directly to test inputs,"
        " or call them from the visualization for live data."
    ),
)

# Permissive CORS so the dev frontend on :5173 can hit us freely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class LogitsRequest(BaseModel):
    logits: List[float] = Field(..., min_length=2, max_length=200)
    temperature: float = Field(1.0, gt=0, le=1e3)
    stable: bool = Field(True, description="Use max-subtraction (recommended)")


class SweepRequest(BaseModel):
    logits: List[float] = Field(..., min_length=2, max_length=200)
    t_min: float = Field(0.05, gt=0)
    t_max: float = Field(5.0, gt=0)
    num_points: int = Field(80, ge=2, le=400)


class CrossEntropyRequest(BaseModel):
    logits: List[float] = Field(..., min_length=2, max_length=200)
    target: int = Field(..., ge=0)
    temperature: float = Field(1.0, gt=0)


class SampleRequest(BaseModel):
    logits: List[float] = Field(..., min_length=2, max_length=200)
    temperature: float = Field(1.0, gt=0)
    n_samples: int = Field(1000, ge=1, le=200_000)
    seed: Optional[int] = None


class StabilityRequest(BaseModel):
    logits: List[float] = Field(..., min_length=2, max_length=200)
    temperature: float = Field(1.0, gt=0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", tags=["health"])
def health():
    return {"ok": True, "service": "ml-viz-api", "version": app.version}


@app.post("/api/softmax/compute", tags=["softmax"])
def softmax_compute(req: LogitsRequest):
    """Compute softmax with full step-by-step trace."""
    T = max(req.temperature, 1e-12)
    z = np.array(req.logits, dtype=np.float64)
    z_scaled = z / T
    z_shifted = z_scaled - np.max(z_scaled)
    exp_vals = np.exp(z_shifted)
    sum_exp = float(np.sum(exp_vals))
    p = (exp_vals / sum_exp) if sum_exp > 0 else np.full_like(exp_vals, np.nan)
    p_clipped = np.clip(p, 1e-12, 1.0)
    return {
        "probabilities": p.tolist(),
        "entropy_nats": float(-np.sum(p_clipped * np.log(p_clipped))),
        "max_entropy_nats": sm.max_entropy(len(req.logits)),
        "max_prob": float(p.max()),
        "argmax": int(p.argmax()),
        "sum_exp": sum_exp,
        "scaled_logits": z_scaled.tolist(),
        "shifted_logits": z_shifted.tolist(),
        "exp_vals": exp_vals.tolist(),
    }


@app.post("/api/softmax/jacobian", tags=["softmax"])
def softmax_jacobian(req: LogitsRequest):
    """Jacobian of softmax: J_ij = p_i (δ_ij - p_j) / T."""
    T = max(req.temperature, 1e-12)
    p = sm.softmax(req.logits, temperature=req.temperature)
    J = sm.jacobian(p) / T
    return {
        "probabilities": p.tolist(),
        "jacobian": J.tolist(),
        "temperature": req.temperature,
    }


@app.post("/api/softmax/sweep", tags=["softmax"])
def softmax_sweep(req: SweepRequest):
    """Compute softmax at many temperatures — for plotting curves."""
    if req.t_max <= req.t_min:
        raise HTTPException(400, "t_max must be greater than t_min")
    return {"rows": sm.sweep_temperature(req.logits, req.t_min, req.t_max, req.num_points)}


@app.post("/api/softmax/cross-entropy", tags=["softmax"])
def softmax_cross_entropy(req: CrossEntropyRequest):
    """Categorical cross-entropy loss + gradient for a given target class."""
    if req.target >= len(req.logits):
        raise HTTPException(400, f"target {req.target} >= n_classes {len(req.logits)}")
    p = sm.softmax(req.logits, temperature=req.temperature)
    y = np.zeros_like(p)
    y[req.target] = 1.0
    loss = sm.cross_entropy(p, y)
    grad = sm.ce_gradient(p, y)
    return {
        "probabilities": p.tolist(),
        "target": req.target,
        "loss": loss,
        "gradient": grad.tolist(),
    }


@app.post("/api/softmax/sample", tags=["softmax"])
def softmax_sample(req: SampleRequest):
    """Draw categorical samples from softmax(logits, T)."""
    return sm.sample(req.logits, req.temperature, req.n_samples, req.seed)


@app.post("/api/softmax/stability", tags=["softmax"])
def softmax_stability(req: StabilityRequest):
    """Side-by-side: naive softmax vs max-shifted softmax."""
    return sm.stability_demo(req.logits, req.temperature)


# Make `python main.py` work too, in addition to the usual `uvicorn main:app`.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
