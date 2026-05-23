# ML Viz API

A small FastAPI service exposing the *real* numerical math behind the
visualizations. Use it to:

- Test softmax / Jacobian / cross-entropy on arbitrary inputs from the browser
- Drive the live charts on the Softmax page (the frontend hits these
  endpoints automatically when the server is running)
- Explore the Swagger UI at <http://localhost:8000/docs>

## Quick start

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

You should see:

```
Uvicorn running on http://0.0.0.0:8000
```

Open <http://localhost:8000/docs> for an interactive playground.

## Endpoints

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/api/health` | Liveness check |
| POST | `/api/softmax/compute` | Full softmax trace: probs, entropy, exp values |
| POST | `/api/softmax/jacobian` | `∂σ(z)/∂z` evaluated at the given logits |
| POST | `/api/softmax/sweep` | Probabilities & entropy across a temperature range |
| POST | `/api/softmax/cross-entropy` | CE loss + gradient `(p − y)` for a target class |
| POST | `/api/softmax/sample` | Draw categorical samples from the softmax distribution |
| POST | `/api/softmax/stability` | Naive vs max-shifted softmax on the same inputs |

## Example: try it from the terminal

```bash
curl -s http://localhost:8000/api/softmax/compute \
  -H 'content-type: application/json' \
  -d '{"logits": [2.0, 1.0, 0.1, -0.5], "temperature": 1.0}' | jq
```

```bash
curl -s http://localhost:8000/api/softmax/jacobian \
  -H 'content-type: application/json' \
  -d '{"logits": [3.0, 1.0, 0.0], "temperature": 1.0}' | jq
```

```bash
# Sweep temperature from 0.1 to 5 in 50 steps
curl -s http://localhost:8000/api/softmax/sweep \
  -H 'content-type: application/json' \
  -d '{"logits": [2, 1, 0.1, -0.5], "t_min": 0.1, "t_max": 5, "num_points": 50}'
```

## Frontend wiring

The frontend reads `VITE_API_URL` from its env (defaults to
`http://localhost:8000`). To point at a different host:

```bash
# from project root
echo "VITE_API_URL=http://localhost:9000" > .env.local
```
