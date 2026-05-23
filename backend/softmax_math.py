"""Core softmax math, kept dependency-free apart from numpy.

Every function here is the *real* numerical reference — exactly what we want
the visualization to pull from when explaining math to users. Numbers
returned are always float / list-of-float so they JSON-serialize cleanly.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def _as_array(x: Iterable[float]) -> np.ndarray:
    return np.asarray(list(x), dtype=np.float64)


def softmax(
    logits: Iterable[float],
    temperature: float = 1.0,
    stable: bool = True,
) -> np.ndarray:
    """Standard softmax with optional max-subtraction (numerical stability).

    σ(z)_i = exp(z_i / T) / Σ_j exp(z_j / T)
    """
    T = max(float(temperature), 1e-12)
    z = _as_array(logits) / T
    if stable:
        z = z - np.max(z)
    e = np.exp(z)
    s = float(np.sum(e))
    return e / (s if s > 0 else 1.0)


def entropy(p: Iterable[float]) -> float:
    """Shannon entropy in nats."""
    p = _as_array(p)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def max_entropy(n: int) -> float:
    """Maximum possible entropy for a categorical over n classes."""
    return float(np.log(max(n, 1)))


def jacobian(p: Iterable[float]) -> np.ndarray:
    """Jacobian of softmax wrt its logits, evaluated at probabilities p.

    J_ij = ∂σ(z)_i / ∂z_j  =  p_i * (δ_ij - p_j)

    Note: this is the Jacobian at T=1. For temperature T it gets divided by T.
    """
    p = _as_array(p)
    return np.diag(p) - np.outer(p, p)


def cross_entropy(p: Iterable[float], y_onehot: Iterable[float]) -> float:
    """Categorical cross entropy: -Σ y_i log p_i."""
    p = np.clip(_as_array(p), 1e-12, 1.0)
    y = _as_array(y_onehot)
    return float(-np.sum(y * np.log(p)))


def ce_gradient(p: Iterable[float], y_onehot: Iterable[float]) -> np.ndarray:
    """Famous result: dL/dz = p - y."""
    return _as_array(p) - _as_array(y_onehot)


def sweep_temperature(
    logits: Iterable[float],
    t_min: float,
    t_max: float,
    n: int,
) -> List[dict]:
    """Compute softmax for `n` temperatures evenly spaced in [t_min, t_max]."""
    Ts = np.linspace(max(float(t_min), 1e-3), float(t_max), int(n))
    out: List[dict] = []
    for T in Ts:
        p = softmax(logits, temperature=float(T))
        out.append(
            {
                "T": float(T),
                "probs": p.tolist(),
                "entropy": entropy(p),
                "max_prob": float(p.max()),
            }
        )
    return out


def sample(
    logits: Iterable[float],
    temperature: float,
    n_samples: int,
    seed: int | None = None,
):
    """Draw n_samples categorical samples from softmax(logits, T)."""
    rng = np.random.default_rng(seed)
    p = softmax(logits, temperature=temperature)
    idx = rng.choice(len(p), size=int(n_samples), p=p)
    counts = np.bincount(idx, minlength=len(p))
    return {
        "probabilities": p.tolist(),
        "counts": counts.tolist(),
        "empirical": (counts / max(int(n_samples), 1)).astype(float).tolist(),
    }


def stability_demo(logits: Iterable[float], temperature: float) -> dict:
    """Compare naive softmax vs max-shifted softmax under extreme logits."""
    T = max(float(temperature), 1e-12)
    z = _as_array(logits) / T

    # Naive: compute exp(z) directly — may overflow to +inf
    with np.errstate(over="ignore", invalid="ignore"):
        e_naive = np.exp(z)
        s_naive = float(np.sum(e_naive))
        p_naive = e_naive / s_naive if s_naive > 0 else np.full_like(e_naive, np.nan)
    naive_ok = bool(np.all(np.isfinite(e_naive)) and np.isfinite(s_naive))

    # Stable: subtract max first
    z_shift = z - np.max(z)
    e_stable = np.exp(z_shift)
    s_stable = float(np.sum(e_stable))
    p_stable = (e_stable / s_stable).tolist()

    def safe(x):
        return float(x) if np.isfinite(x) else None

    return {
        "naive": {
            "ok": naive_ok,
            "exp_vals": [safe(v) for v in e_naive.tolist()],
            "sum_exp": safe(s_naive),
            "probs": [safe(v) for v in p_naive.tolist()],
            "max_exp": safe(float(np.max(e_naive))) if naive_ok else None,
        },
        "stable": {
            "ok": True,
            "exp_vals": e_stable.tolist(),
            "sum_exp": s_stable,
            "probs": p_stable,
            "max_exp": float(np.max(e_stable)),
            "shifted_logits": z_shift.tolist(),
        },
    }
