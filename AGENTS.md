# helper

Stateless Python gRPC server that processes chat requests through LLM adapters and serves embeddings for vector search. No database.

## Key facts

- **141 pytest tests, CI-gated** — CI (`docker.yml`) runs `pytest` before Docker build on every PR; the `test` job gates `validate` and `push`. Overall coverage **77%**; domain core at 97%, LLM adapters at 100%, metrics at 95%, embedding provider at 92%. Also runs `vector-parity.yml` (byte-parity gate against backend `cmd/hash_fixture`). No lint, no typecheck. On push to `main`, a `Deploy to Hermes` job runs on the self-hosted runner to deploy the new image automatically.
- **OpenCode adapters use `langchain_openai.ChatOpenAI`** — `opencode0` (`big-pickle`), `opencode1` (`deepseek-v4-flash-free`), and `opencode2` (`mimo-v2.5-free`) all go through the OpenAI-compatible API at `opencode.ai/zen/v1`. Model names are **hardcoded in `main.py`**, not env-driven (env `LLM_MODEL` only feeds the adapter's fallback when the model arg is omitted — current code always passes it explicitly).
- **Mistral adapter uses `langchain_openai.ChatOpenAI`** — Mistral's API is OpenAI-compatible; requires `MISTRAL_API_KEY`. The adapter is **only registered when `MISTRAL_API_KEY` is set** (`main.py` checks the env var and logs `"MISTRAL_API_KEY not set; skipping Mistral adapter"` otherwise).
- **Ollama adapter uses raw `requests`** — no langchain; full prompt (system+history+user) is concatenated into a single string and POSTed to `/api/generate` with `stream=False, think=False`. Production default model `qwen2.5:1.5b` (set in `main.py` from `OLLAMA_MODEL`; `infra/docker-compose.yml` mirrors this). Note: `OllamaLLMAdapter.__init__` also has its own default of `qwen3.5:0.8b`, but it is unreachable because `main.py` always passes `OLLAMA_MODEL` explicitly.
- **JSON format instructions are appended to the user message**, not the system prompt — some providers ignore system formatting. Only fires when `skip_role_detection=false`. Backend always sends `skip_role_detection=true` today, so this path is dormant.
- **HTTP sidecar on `:8084`** via stdlib `http.server` — serves `GET /health` AND `GET /metrics` (Prometheus text). Health is post-startup dependency-aware (LLM adapter reachability). Ollama adapter health is treated as "optional/local" and never downgrades overall status.
- **Fallback chain** (R5, cheap-first): OpenCode 0 → OpenCode 1 → OpenCode 2 → Mistral → Ollama (when no explicit provider is set). `HelperAgent.FALLBACK_CHAIN = ["opencode0", "opencode1", "opencode2", "mistral", "ollama"]`; env-configurable via `FALLBACK_CHAIN`. Mistral is silently skipped at runtime if not loaded (no `MISTRAL_API_KEY`).
- **Embedding service**: helper also exposes `Embed` and `EmbedBatch` gRPC methods backed by `OllamaEmbeddingProvider` (`internal/adapters/embedding_provider.py`, default model `granite-embedding:278m`, 768 dims). Used by backend re-embed path; does NOT participate in chat fallback chain. `DimensionMismatchError` (vector dim ≠ 768) surfaces as gRPC `FAILED_PRECONDITION` so backend never persists bad data.
- **Backfill script** (`scripts/backfill_embeddings.py`) — batch-upserts `worker_embeddings` rows for all workers. Generates field texts via the same normalization logic as the backend's `core.BuildFieldTexts()` (EN/ES profession aliases, JSON array joining). The byte-parity gate (`scripts/test_byte_parity_gate.sh`) ensures Go and Python produce identical field texts and SHA-256 hashes for the same worker profile input. Run after schema changes or first-time vector search enable.

## Architecture

Pure hexagonal: `HelperAgent` (in `internal/core/helper_agent.py`) depends only on `LLMPort` Protocol (`internal/ports/llm.py`). All adapters implement it. Adapters are loaded once at startup in `main.py`; the backend picks per-request via the `llm_provider` gRPC field. `EmbeddingProvider` is a similar informal base class in `internal/adapters/embedding_provider.py` (not ABC, matches the codebase style).

## Proto regeneration

If `proto/helper.proto` changes, regenerate with:
```bash
python -m grpc_tools.protoc -Iproto --python_out=proto --pyi_out=proto --grpc_python_out=proto proto/helper.proto
```
Requires `grpcio-tools` (not in `requirements.txt`). The generated `*_pb2.py` / `*_pb2_grpc.py` files are checked in.

## LLM response format

When `skip_role_detection=false`, the adapter is instructed (via the user message) to return JSON: `{"answer": "...", "role": "worker|client|"}`.
When JSON parsing fails, `HelperAgent._parse_response` falls back to raw text (role = `""`). Markdown-fenced JSON (` ```json ... ``` `) is unwrapped before parsing.

When `skip_role_detection=true` (today's backend behavior for worker/client intake and both passes of search), no JSON instruction is appended; the LLM returns free-form text and the backend's `[FIELDS]` / `[SEARCH]` block extractor parses structured data out.

## Dev

```bash
pip install -r requirements.txt
python main.py          # starts gRPC on :50051 + health on :8084
```

No `PYTHONPATH` needed when running from the project root. The Dockerfile sets `PYTHONPATH=/app:/app/proto` because the protoc-generated `helper_pb2_grpc.py` does `import helper_pb2` (not relative); `/app/proto` mirrors the `sys.path` hack in `helper/scripts/backfill_embeddings.py`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v tests/               # 141 tests
pytest --cov=internal tests/   # coverage report
```

Test files live in `helper/tests/`. Nine files covering: domain core (33), metrics (18), health cache (16), gRPC server (14), OpenCode adapter (7), Mistral adapter (7), Ollama adapter (10), embedding provider (23), main (13). Run from `helper/` directory.

Drift guard: `./scripts/check-test-count.sh` in CI fails with exit 1 (= per-file breakdown on stderr) if the count above drifts from `helper/tests/*.py`. Wire into `helper/.github/workflows/docker.yml` next to the existing pytest step so a PR-build breaks before any image push.

## Env vars

Required at startup (via `require_env` in `main.py` — exits 1 if missing): `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `GRPC_PORT`, `HEALTH_PORT`.

Optional:

| Variable | Default | Notes |
|----------|---------|-------|
| `GRPC_PORT` | `50051` | gRPC listen port |
| `HEALTH_PORT` | `8084` | Health check HTTP listen port |
| `LLM_BASE_URL` | `https://opencode.ai/zen/v1` | OpenCode API |
| `LLM_MODEL` | `deepseek-v4-flash-free` | OpenCode model (fallback when adapter constructed without explicit model) |
| `LLM_API_KEY` | — | Required for OpenCode |
| `MISTRAL_API_KEY` | — | Optional; if not set, Mistral adapter is skipped |
| `MISTRAL_MODEL` | `mistral-large-latest` | Mistral model |
| `MISTRAL_BASE_URL` | `https://api.mistral.ai/v1` | Mistral API |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (LLM AND embedding share this daemon) |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama chat model (fallback chat when cloud providers are down) |
| `EMBEDDING_MODEL` | `granite-embedding:278m` | Ollama embedding model (vector search) |
| `HELPER_AUTH_TOKEN` | — | gRPC auth token (Bearer); empty = no auth (R1) |
| `GRPC_MAX_CONCURRENT_RPCS` | `32` | Server `maximum_concurrent_rpcs` (R3) |
| `GRPC_MAX_WORKERS` | `16` | Thread pool `max_workers` for gRPC (R6) |
| `HEALTH_CACHE_TTL_S` | `20` | Health cache TTL in seconds (R2) |
| `MAX_QUESTION_LENGTH` | `32000` | Max question chars; longer → `INVALID_ARGUMENT` (R8) |
| `REQUEST_BUDGET_S` | `45.0` | Per-request LLM budget in seconds (R4) |
| `FALLBACK_CHAIN` | `opencode0,opencode1,opencode2,mistral,ollama` | Comma-separated fallback order (R5) |