# helper

Stateless Python gRPC server that processes chat requests through LLM adapters. No database.

## Key facts

- **No tests, no lint, no typecheck** — only CI is Docker build/push (`docker.yml`). The only verification is that `python main.py` starts without error.
- **OpenCode adapters use `langchain_openai.ChatOpenAI`** — both `opencode1` (deepseek-v4-flash-free) and `opencode2` (mimo-v2.5-free) go through the OpenAI-compatible API at `opencode.ai/zen/v1`.
- **Mistral adapter uses `langchain_openai.ChatOpenAI`** — Mistral's API is OpenAI-compatible; requires `MISTRAL_API_KEY`.
- **Ollama adapter uses raw `requests`** — no langchain; full prompt (system+history+user) is concatenated into a single string. Production default model `qwen2.5:1.5b` (set in `main.py` from `OLLAMA_MODEL`; `infra/docker-compose-dev.yaml` mirrors this). Note: `OllamaLLMAdapter.__init__` also has its own default of `qwen3.5:0.8b`, but it is unreachable because `main.py` always passes `OLLAMA_MODEL` explicitly.
- **JSON format instructions are appended to the user message**, not the system prompt — some providers ignore system formatting.
- **HTTP sidecar on `:8084`** via stdlib `http.server` — serves `GET /health` AND `GET /metrics` (Prometheus text). Health is post-startup dependency-aware (LLM adapter reachability).
- **Fallback chain**: Mistral → OpenCode 1 → OpenCode 2 → Ollama (when no explicit provider is set).
- **Embedding service**: helper also exposes `Embed` and `EmbedBatch` gRPC methods backed by `OllamaEmbeddingProvider` (`internal/adapters/embedding_provider.py`, default model `granite-embedding:278m`). Used by backend re-embed path; does NOT participate in chat fallback chain.

## Architecture

Pure hexagonal: `HelperAgent` (in `internal/core/helper_agent.py`) depends only on `LLMPort` Protocol (`internal/ports/llm.py`). All adapters implement it. Adapters are loaded once at startup in `main.py`; the backend picks per-request via the `llm_provider` gRPC field.

## Proto regeneration

If `proto/helper.proto` changes, regenerate with:
```bash
python -m grpc_tools.protoc -Iproto --python_out=proto --pyi_out=proto --grpc_python_out=proto proto/helper.proto
```
Requires `grpcio-tools` (not in `requirements.txt`). The generated `*_pb2.py` / `*_pb2_grpc.py` files are checked in.

## LLM response format

The adapter is instructed to return JSON: `{"answer": "...", "role": "worker|client|"}`.
If JSON parsing fails, `HelperAgent` falls back to raw text (role = `""`).

## Dev

```bash
pip install -r requirements.txt
python main.py          # starts gRPC on :50051 + health on :8084
```

No `PYTHONPATH` needed when running from the project root.

## Env vars

| Variable | Default | Notes |
|----------|---------|-------|
| `GRPC_PORT` | `50051` | gRPC listen port |
| `HEALTH_PORT` | `8084` | Health check HTTP listen port |
| `LLM_BASE_URL` | `https://opencode.ai/zen/v1` | OpenCode API |
| `LLM_MODEL` | `deepseek-v4-flash-free` | OpenCode model |
| `LLM_API_KEY` | — | Required for OpenCode |
| `MISTRAL_API_KEY` | — | Optional; if not set, Mistral adapter is skipped |
| `MISTRAL_MODEL` | `mistral-large-latest` | Mistral model |
| `MISTRAL_BASE_URL` | `https://api.mistral.ai/v1` | Mistral API |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (LLM AND embedding share this daemon) |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama chat model (fallback chat when cloud providers are down; per VECTOR_SEARCH_PLAN §17) |
| `EMBEDDING_MODEL` | `granite-embedding:278m` | Ollama embedding model (vector search) |