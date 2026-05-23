// Thin client for the FastAPI backend. Everything is best-effort — the UI
// keeps working when the backend is offline by falling back to client-side
// math (see `softmax.ts`).

import { useEffect, useRef, useState } from "react";

export const API_BASE: string =
  (import.meta.env.VITE_API_URL as string | undefined) ??
  "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function postJSON<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiError(res.status, text || res.statusText);
  }
  return (await res.json()) as T;
}

/* -------------------------------- Schemas -------------------------------- */

export interface ComputeResp {
  probabilities: number[];
  entropy_nats: number;
  max_entropy_nats: number;
  max_prob: number;
  argmax: number;
  sum_exp: number;
  scaled_logits: number[];
  shifted_logits: number[];
  exp_vals: number[];
}

export interface JacobianResp {
  probabilities: number[];
  jacobian: number[][];
  temperature: number;
}

export interface SweepRow {
  T: number;
  probs: number[];
  entropy: number;
  max_prob: number;
}

export interface CrossEntropyResp {
  probabilities: number[];
  target: number;
  loss: number;
  gradient: number[];
}

export interface SampleResp {
  probabilities: number[];
  counts: number[];
  empirical: number[];
}

export interface StabilityResp {
  naive: {
    ok: boolean;
    exp_vals: (number | null)[];
    sum_exp: number | null;
    probs: (number | null)[];
    max_exp: number | null;
  };
  stable: {
    ok: boolean;
    exp_vals: number[];
    sum_exp: number;
    probs: number[];
    max_exp: number;
    shifted_logits: number[];
  };
}

/* -------------------------------- Client --------------------------------- */

export const api = {
  health: async (signal?: AbortSignal): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/api/health`, { signal });
      return res.ok;
    } catch {
      return false;
    }
  },

  compute: (logits: number[], temperature = 1, signal?: AbortSignal) =>
    postJSON<ComputeResp>(
      "/api/softmax/compute",
      { logits, temperature, stable: true },
      signal
    ),

  jacobian: (logits: number[], temperature = 1, signal?: AbortSignal) =>
    postJSON<JacobianResp>(
      "/api/softmax/jacobian",
      { logits, temperature, stable: true },
      signal
    ),

  sweep: (
    logits: number[],
    t_min = 0.05,
    t_max = 5,
    num_points = 80,
    signal?: AbortSignal
  ) =>
    postJSON<{ rows: SweepRow[] }>(
      "/api/softmax/sweep",
      { logits, t_min, t_max, num_points },
      signal
    ),

  crossEntropy: (
    logits: number[],
    target: number,
    temperature = 1,
    signal?: AbortSignal
  ) =>
    postJSON<CrossEntropyResp>(
      "/api/softmax/cross-entropy",
      { logits, target, temperature },
      signal
    ),

  sample: (
    logits: number[],
    temperature = 1,
    n_samples = 1000,
    seed?: number,
    signal?: AbortSignal
  ) =>
    postJSON<SampleResp>(
      "/api/softmax/sample",
      { logits, temperature, n_samples, seed },
      signal
    ),

  stability: (logits: number[], temperature = 1, signal?: AbortSignal) =>
    postJSON<StabilityResp>(
      "/api/softmax/stability",
      { logits, temperature },
      signal
    ),
};

/* --------------------------- Connection status --------------------------- */

export type ApiStatus = "checking" | "online" | "offline";

/** Polls `/api/health` periodically. Returns the current status. */
export function useApiStatus(intervalMs = 15_000): ApiStatus {
  const [status, setStatus] = useState<ApiStatus>("checking");
  const aborted = useRef(false);

  useEffect(() => {
    aborted.current = false;
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      const ok = await api.health();
      if (aborted.current) return;
      setStatus(ok ? "online" : "offline");
      timer = setTimeout(tick, intervalMs);
    };
    tick();

    return () => {
      aborted.current = true;
      clearTimeout(timer);
    };
  }, [intervalMs]);

  return status;
}
