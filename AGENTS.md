# helper

Stateless Python gRPC server that processes chat requests through LLM adapters and serves embeddings for vector search. No database.

## Key facts

- **No tests, no lint, no typecheck** — CI is `docker.yml` (Docker build/push) + `vector-parity.yml` (runs `scripts/test_byte_parity_gate.sh` against the backend's `cmd/hash_fixture`). The only smoke verification is that `python main.py` starts without error.
- **OpenCode adapters use `langchain_openai.ChatOpenAI`** — both `opencode1` (`deepseek-v4-flash-free`) and `opencode2` (`mimo-v2.5-free`) go through the OpenAI-compatible API at `opencode.ai/zen/v1`. Model names are **hardcoded in `main.py`**, not env-driven (env `LLM_MODEL` only feeds the adapter's fallback when the model arg is omitted — current code always passes it explicitly).
- **Mistral adapter uses `langchain_openai.ChatOpenAI`** — Mistral's API is OpenAI-compatible; requires `MISTRAL_API_KEY`. The adapter is **only registered when `MISTRAL_API_KEY` is set** (`main.py` checks the env var and logs `"MISTRAL_API_KEY not set; skipping Mistral adapter"` otherwise).
- **Ollama adapter uses raw `requests`** — no langchain; full prompt (system+history+user) is concatenated into a single string and POSTed to `/api/generate` with `stream=False, think=False`. Production default model `qwen2.5:1.5b` (set in `main.py` from `OLLAMA_MODEL`; `infra/docker-compose.yml` mirrors this). Note: `OllamaLLMAdapter.__init__` also has its own default of `qwen3.5:0.8b`, but it is unreachable because `main.py` always passes `OLLAMA_MODEL` explicitly.
- **JSON format instructions are appended to the user message**, not the system prompt — some providers ignore system formatting. Only fires when `skip_role_detection=false`. Backend always sends `skip_role_detection=true` today, so this path is dormant.
- **HTTP sidecar on `:8084`** via stdlib `http.server` — serves `GET /health` AND `GET /metrics` (Prometheus text). Health is post-startup dependency-aware (LLM adapter reachability). Ollama adapter health is treated as "optional/local" and never downgrades overall status.
- **Fallback chain**: Mistral → OpenCode 1 → OpenCode 2 → Ollama (when no explicit provider is set). `HelperAgent.FALLBACK_CHAIN = ["mistral", "opencode1", "opencode2", "ollama"]`; the `mistral` entry is silently skipped at runtime if the adapter wasn't loaded (no `MISTRAL_API_KEY`).
- **Embedding service**: helper also exposes `Embed` and `EmbedBatch` gRPC methods backed by `OllamaEmbeddingProvider` (`internal/adapters/embedding_provider.py`, default model `granite-embedding:278m`, 768 dims). Used by backend re-embed path; does NOT participate in chat fallback chain. `DimensionMismatchError` (vector dim ≠ 768) surfaces as gRPC `FAILED_PRECONDITION` so backend never persists bad data.

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
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama chat model (fallback chat when cloud providers are down; per VECTOR_SEARCH_PLAN §17) |
| `EMBEDDING_MODEL` | `granite-embedding:278m` | Ollama embedding model (vector search) |