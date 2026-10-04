# LLM Security Gateway

A defensive FastAPI proxy that screens prompts before an LLM call and redacts sensitive model output before returning it.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — start the FastAPI gateway on the workflow port
- `uv run --project . pytest` — run the Python tests
- `PYTHONPATH=artifacts/api-server uv run --project . python artifacts/api-server/scripts/benchmark.py --iterations 100` — run the local screening benchmark
- `pnpm --filter @workspace/api-server run typecheck` — check the retained TypeScript scaffold
- Default mode uses a deterministic local demo upstream. Remote mode needs `UPSTREAM_MODE`, `UPSTREAM_BASE_URL`, and `UPSTREAM_API_KEY` configured outside source control.

## Stack

- pnpm workspace, Python 3.13, FastAPI, Uvicorn, HTTPX
- Input classifier: scikit-learn TF-IDF + logistic regression
- Optional semantic detector: sentence-transformers MiniLM (disabled by default)
- Docker image definition: `artifacts/api-server/Dockerfile`

## Where things live

- `artifacts/api-server/firewall/` — gateway, detectors, input/output guards, and upstream adapters
- `artifacts/api-server/tests/` — pytest coverage
- `artifacts/api-server/scripts/benchmark.py` — local warm-pipeline benchmark
- `artifacts/api-server/README.md` — API usage, configuration, deployment, and evaluation methodology
- `pyproject.toml` / `uv.lock` — Python dependencies and lockfile

## Architecture decisions

- The API uses the full `/api` prefix because the artifact proxy does not strip service paths.
- The upstream URL is operator-configured, HTTPS-only in remote mode, and never accepted from a request.
- Streaming is rejected until output can be inspected before tokens are released.
- The default demo upstream avoids external calls; remote model credentials must be runtime secrets.
- MiniLM is opt-in because its download, memory use, and CPU latency vary by deployment.

## Product

Provides an OpenAI-compatible chat-completions endpoint, standalone input/output scanning endpoints, low-cardinality in-memory metrics, a trainable baseline classifier, and a synthetic benchmark.

## User preferences

No additional standing preferences recorded.

## Gotchas

- Input screening covers `user` and `tool` messages; system/developer roles are treated as operator-controlled.
- The bundled classifier corpus is illustrative only and must not be presented as a production evaluation.
- Enabling MiniLM requires installing `sentence-transformers` and may exceed a 50 ms screening budget.
- Do not log raw prompts, completions, or upstream credentials.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
