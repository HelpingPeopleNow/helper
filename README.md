# HelpingPeopleNow Helper

Stateless Python gRPC server that processes chat requests through LLM adapters and serves embeddings for vector search. Receives questions from the backend via gRPC, picks the right LLM adapter per request, and returns answers.

**Container:** `helpingpeoplenow-helper` | **gRPC Port:** `:50051` | **Health HTTP Port:** `:8084`

---

## Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.12 |
| **gRPC** | `grpcio` (server only) |
| **LLM Adapters** | `langchain-openai` (OpenCode ×2, Mistral) + raw `requests` (Ollama) |
| **Embeddings** | `httpx` against Ollama `/api/embeddings` |
| **Validation** | Pydantic v2 |
| **Metrics** | `prometheus_client` |
| **Container** | `python:3.12-slim` (source-direct, no multi-stage) |

---

## What It Does

1. **Chat completion** — receives an `AskRequest` with question, history, system prompt, and LLM provider selector; returns an answer + detected user role.
2. **Multi-adapter fallback** — loads `opencode0`, `opencode1`, `opencode2`, `ollama` unconditionally and `mistral` only when `MISTRAL_API_KEY` is set; the backend chooses which one per request via the `llm_provider` field; empty = automatic fallback chain (Mistral → OpenCode 0 → OpenCode 1 → OpenCode 2 → Ollama).
3. **Role detection** — the LLM's answer is scanned for worker/client intent; the system prompt instructs the model to return a role tag for classification. `skip_role_detection` flag suppresses this for profile intake chats. (Backend always sends `skip_role_detection=true` today.)
4. **Embeddings (vector search)** — exposes `Embed` and `EmbedBatch` gRPC methods backed by `OllamaEmbeddingProvider` (`granite-embedding:278m`, 768 dims). Used by the backend to re-embed worker profile field changes for vector search.
5. **Health + metrics sidecar** — HTTP `:8084` (stdlib `http.server`) serves liveness probes AND Prometheus metrics on the same port.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         main.py                               │
│  (entry point: loads adapters + embedding provider,            │
│   starts gRPC + health server)                                │
│                                                               │
│  adapters = {                                                 │
│      "opencode0": OpenCodeLLMAdapter(model="big-pickle"),     │
│      "opencode1": OpenCodeLLMAdapter(model="deepseek-…-free"),│
│      "opencode2": OpenCodeLLMAdapter(model="mimo-v2.5-free"), │
│      "ollama":    OllamaLLMAdapter(),                         │
│      # "mistral": added only if MISTRAL_API_KEY is set        │
│      "mistral":   MistralLLMAdapter(model="mistral-large-…")  │
│  }                                                            │
│  embedding_provider = OllamaEmbeddingProvider(model="granite- │
│      embedding:278m")  # shares OLLAMA_BASE_URL with chat     │
│  assistant = HelperAgent(adapters)                            │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────────────────────┐
│                  gRPC Server (:50051)                          │
│               internal/adapters/grpc_server.py                │
│                                                               │
│  HelperService.Ask(request) → response                        │
│    ├─ system_prompt: from backend cache                       │
│    ├─ llm_provider: from admin config or empty (= fallback)  │
│    ├─ skip_role_detection: true for profile intake chats      │
│    └─ calls helper_agent.answer(...)                          │
│                                                               │
│  HelperService.Embed(request) → embedding vector              │
│  HelperService.EmbedBatch(request) → list of embedding vectors│
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────────────────────┐
│              internal/core/helper_agent.py                    │
│                                                               │
│  HelperAgent.answer(question, system_prompt, history,         │
│                     llm_provider, skip_role_detection)        │
│    │                                                          │
│    ├─ builds provider chain:                                  │
│    │     if llm_provider: [llm_provider, *FALLBACK_CHAIN]    │
│    │     else:             FALLBACK_CHAIN                     │
│    ├─ for each provider: try adapter.complete(...)            │
│    │   └─ on failure: fall through to next provider           │
│    └─ parse response: JSON {"answer","role"} or raw text      │
│                                                               │
│  FALLBACK_CHAIN = ["mistral", "opencode0", "opencode1",       │
│                    "opencode2", "ollama"]                     │
└───────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Stateless** — no database connection, no file storage. Everything it needs comes via gRPC from the backend. A lightweight health HTTP endpoint is available for liveness checks.
- **Port-based** — `LLMPort` is a Python `Protocol` class (`internal/ports/llm.py`); all adapters implement it. `EmbeddingProvider` is a similar informal class in `internal/adapters/embedding_provider.py`. Adding a new provider requires only a new adapter class.
- **Runtime provider switching** — the backend chooses the adapter per request. The helper never restarts when the provider changes.
- **Fallback chain** — when no explicit provider is set (or the chosen one fails), adapters are tried in order: Mistral → OpenCode 0 → OpenCode 1 → OpenCode 2 → Ollama. The chain is built in `HelperAgent._answer_inner` — Mistral only participates if the Mistral adapter was loaded (i.e., `MISTRAL_API_KEY` is set).

---

## Request Flow

```
Backend (via gRPC)
       │
       ▼
AskRequest {
    question: "I need a plumber",
    history: [{role: "user", content: "hello"}],
    system_prompt: "You are a home services assistant...",
    llm_provider: "opencode1"  // or "opencode0", "opencode2", "mistral", "ollama", ""
    skip_role_detection: false  // true for worker/client intake chats
}
       │
       ▼
grpc_server.py
  ├─ log: provider=opencode1 (request='opencode1' default=opencode0)
  └─ helper_agent.answer(...)
       │
       ▼
helper_agent.py
  ├─ providers_chain = ["opencode1", "mistral", "opencode0", "opencode2", "ollama"]
  ├─ for provider in chain: try adapter.complete(system_prompt, user, history)
  └─ return _parse_response(raw)
       │
       ▼
OpenCodeLLMAdapter (or MistralLLMAdapter, or OllamaLLMAdapter)
  ├─ build messages: [system, ...history, user]
  ├─ call LLM API (OpenAI-compatible for OpenCode/Mistral, raw prompt for Ollama)
  └─ return assistant text
       │
       ▼
AskResponse {
    answer: "I can help you find plumbing services...",
    detected_role: "client"  // parsed from answer text (only when skip_role_detection=false)
}
```

---

## gRPC Contract

```protobuf
service HelperService {
  rpc Ask(AskRequest) returns (AskResponse);
  rpc Embed(EmbedRequest) returns (EmbedResponse);          // vector search
  rpc EmbedBatch(EmbedBatchRequest) returns (EmbedBatchResponse);  // backfill
}

message AskRequest {
  string question = 1;
  repeated Message history = 2;        // chat history context
  string system_prompt = 3;            // loaded by backend from DB
  string llm_provider = 4;             // "opencode0" | "opencode1" | "opencode2" | "mistral" | "ollama" | "" (= fallback chain)
  bool skip_role_detection = 5;        // if true, don't append JSON role-detection instructions
}

message AskResponse {
  string answer = 1;
  string detected_role = 2;            // "worker" | "client" | "" (not detected)
}

message EmbedRequest {
  string text = 1;
  string model = 2;                    // empty → use provider default (granite-embedding:278m)
}

message EmbedResponse {
  repeated float embedding = 1;        // 768 dims for granite-embedding:278m
  string model = 2;                    // actual model used
  int32 dimensions = 3;
}

message EmbedBatchRequest {
  repeated string texts = 1;
  string model = 2;
}

message EmbedBatchResponse {
  repeated EmbedResponse embeddings = 1;
}
```

Proto definition: `proto/helper.proto` (canonical source — shared with the backend repo via `proto/helper/helper.proto` mirror). Generated `proto/helper_pb2.py` / `proto/helper_pb2_grpc.py` are checked in.

### HTTP sidecar (health + metrics)

A lightweight HTTP server runs alongside the gRPC service on a separate port, using Python stdlib `http.server` (no extra dependencies). It serves both health checks AND Prometheus metrics.

| Endpoint | Method | Response |
|----------|--------|----------|
| `/health` | GET | JSON. `200 → status: ok` if gRPC server is up and at least one LLM adapter is reachable; `503 → status: degraded` if not. Includes `status`, `grpc`, `adapters`, `adapter_results` (per-adapter ok/down/skipped), `adapter_details` (per-adapter human-readable), `loaded_adapters`. |
| `/metrics` | GET | Prometheus text format (`prometheus_client`). Counters: `llm_requests_total`, `llm_errors_total`, `llm_tokens_total`, `grpc_requests_total`, `auth_errors_total`, `health_check_total`. Histograms: `llm_request_duration_seconds`, `grpc_request_duration_seconds`. Gauges: `active_requests`. |

Port is configurable via the `HEALTH_PORT` env var (default: `8084`).

The embedding model check is non-blocking — Ollama is treated as "optional/local" for the LLM adapter (not downgraded to degraded), and the embedding check surfaces its status in `adapter_results` without flipping overall status.

---

## LLM Adapters

### OpenCode (External) — 3 instances

All use `langchain_openai.ChatOpenAI` (OpenAI-compatible API). Models are pinned in `main.py` (not env-driven):

| Adapter | Model (hardcoded in main.py) | Base URL | Notes |
|---------|------------------------------|----------|-------|
| `opencode0` | `big-pickle` | `LLM_BASE_URL` (default `https://opencode.ai/zen/v1`) | Primary OpenCode model (fast) |
| `opencode1` | `deepseek-v4-flash-free` | `LLM_BASE_URL` | Secondary OpenCode model |
| `opencode2` | `mimo-v2.5-free` | `LLM_BASE_URL` | Tertiary OpenCode model |

Both adapters' constructor overrides accept `(model, base_url, api_key, temperature=0.3)` from env if not set in `main.py`. LangChain timeout: 20s (fail-fast).

### Mistral (External) — loaded only when `MISTRAL_API_KEY` is set

| Adapter | Model (default) | Base URL (default) |
|---------|-----------------|---------------------|
| `mistral` | `mistral-large-latest` (env `MISTRAL_MODEL`) | `https://api.mistral.ai/v1` (env `MISTRAL_BASE_URL`) |

Requires `MISTRAL_API_KEY`. When unset, the adapter is skipped — `main.py` logs `"MISTRAL_API_KEY not set; skipping Mistral adapter"` and `loaded_adapters` will not include it.

### Ollama (Local)

- Connects to a local Ollama instance at `OLLAMA_BASE_URL` (default `http://localhost:11434`).
- Uses raw `requests` (no langchain) — full prompt (system+history+user) is concatenated into a single string and POSTed to `/api/generate` with `stream=False, think=False`.
- Model: env `OLLAMA_MODEL` (default in `OllamaLLMAdapter.__init__` is `qwen3.5:0.8b`, but `main.py` always passes `OLLAMA_MODEL` explicitly so the runtime default is whatever compose sets — `qwen2.5:1.5b` in production per `infra/docker-compose.yml`).

### Embedding provider

Used by `Embed` and `EmbedBatch` gRPC methods (vector search backend path). Single Ollama-backed adapter, sharing `OLLAMA_BASE_URL` with the chat path:

| Provider | Model (default) | Base URL | Purpose |
|----------|-----------------|----------|---------|
| `OllamaEmbeddingProvider` | `granite-embedding:278m` (env `EMBEDDING_MODEL`) | `OLLAMA_BASE_URL` | Produces 768-dim vectors for `worker_embeddings` (VECTOR_SEARCH_PLAN §7.4). |

The proto definitions for `Embed` / `EmbedBatch` are in `proto/helper.proto`; both methods report dimensions and the actual model used. Dimension mismatch (≠768) raises `DimensionMismatchError` → gRPC `FAILED_PRECONDITION` so backend never persists bad data.

### Fallback Chain

```
Mistral → OpenCode 0 → OpenCode 1 → OpenCode 2 → Ollama
```

`HelperAgent.FALLBACK_CHAIN = ["mistral", "opencode0", "opencode1", "opencode2", "ollama"]` is the canonical chain. The `mistral` entry is silently skipped at runtime if the adapter wasn't loaded (no `MISTRAL_API_KEY`).

If an explicit `llm_provider` is set, it's tried first, then the remaining providers in fallback order.

### Adding a New Provider

Create a new file `internal/adapters/<provider>_llm.py` that implements `LLMPort`:

```python
from internal.ports.llm import LLMPort, Message

class MyProviderLLMAdapter(LLMPort):
    def complete(self, system_prompt: str, user: str, history: tuple[Message, ...] = ()) -> str:
        # Your implementation here
        return response_text
```

Then add it in `main.py`:

```python
adapters = {
    "opencode0": OpenCodeLLMAdapter(...),
    "opencode1": OpenCodeLLMAdapter(...),
    "opencode2": OpenCodeLLMAdapter(...),
    "ollama": OllamaLLMAdapter(...),
    # ... plus conditional Mistral
    "myprovider": MyProviderLLMAdapter(...),
}
```

If you want to change the chain order, edit `HelperAgent.FALLBACK_CHAIN` in `internal/core/helper_agent.py`.

---

## Logging

Structured logging to stdout with timestamps and module names:

| Component | Events |
|-----------|--------|
| `main.py` | Adapter loading, default provider, gRPC server start |
| `grpc_server.py` | Request size, history count, system prompt length, provider selection, skip_role_detection, Embed dims/model |
| `opencode_llm.py` | Model name, call duration, response size, errors |
| `mistral_llm.py` | Model name, call duration, response size, errors |
| `ollama_llm.py` | Model name, call duration, response size, errors |
| `embedding_provider.py` | Model/base_url/timeout at init, embed dim, model-not-pulled diagnostics |

Format:

```
2026-06-12 08:15:22 INFO internal.adapters.grpc_server gRPC Ask: q_len=42 history=3 sp_len=1200 provider=opencode1 skip_role=False
2026-06-12 08:15:22 INFO internal.adapters.opencode_llm LLM call: msgs=5 prompt_chars=1800
2026-06-12 08:15:25 INFO internal.adapters.opencode_llm LLM response: elapsed_ms=2950 response_chars=312
2026-06-12 08:15:25 INFO internal.adapters.grpc_server gRPC Ask done: answer_len=312 elapsed_ms=2951
```

---

## Environment Variables

Required at startup (via `require_env` in `main.py`): `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `GRPC_PORT`, `HEALTH_PORT`. Missing values exit 1.

Optional:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_PORT` | `50051` | gRPC listen port |
| `HEALTH_PORT` | `8084` | Health check HTTP listen port |
| `LLM_BASE_URL` | `https://opencode.ai/zen/v1` | OpenCode API base URL |
| `LLM_MODEL` | `deepseek-v4-flash-free` | OpenCode model name (used by `OpenCodeLLMAdapter` when constructed without an explicit model) |
| `LLM_API_KEY` | (required) | OpenCode API key |
| `MISTRAL_API_KEY` | — | Mistral API key. If unset, Mistral adapter is skipped. |
| `MISTRAL_MODEL` | `mistral-large-latest` | Mistral model name |
| `MISTRAL_BASE_URL` | `https://api.mistral.ai/v1` | Mistral API base URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (used by `ollama` provider AND `OllamaEmbeddingProvider` — same daemon) |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` (production) | Ollama chat model name (passed by `main.py` from env) |
| `EMBEDDING_MODEL` | `granite-embedding:278m` | Ollama embedding model name (vector search) |

---

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (needs backend running for gRPC calls)
python main.py

# Smoke test — server should bind 50051 (gRPC) + 8084 (HTTP sidecar)
python main.py

# Regenerate protobuf bindings (requires grpcio-tools, not in requirements.txt)
python -m grpc_tools.protoc -Iproto --python_out=proto --pyi_out=proto --grpc_python_out=proto proto/helper.proto

# Docker build
docker build -t ghcr.io/helpingpeoplenow/helper:latest .
```

CI (`.github/workflows/docker.yml`) builds and pushes the Docker image on push to `main`. A second workflow `vector-parity.yml` runs `scripts/test_byte_parity_gate.sh` against the backend's `cmd/hash_fixture` to gate byte-level parity between Go (`BuildFieldTexts`) and Python (`backfill_embeddings.py`).

**No unit tests, no lint, no typecheck** — only CI is Docker build/push + parity gate.

---

## Project Structure

```
helper/
├── main.py                       # Entry point: load adapters + embedding provider, start gRPC + health server
├── Dockerfile                    # Single-stage: python:3.12-slim (PYTHONPATH=/app:/app/proto)
├── requirements.txt              # Python dependencies (grpcio, langchain-openai, requests, httpx, prometheus_client, psycopg2-binary)
├── VERSION                       # 0.4
├── proto/
│   ├── helper.proto              # Protobuf contract (canonical source)
│   ├── helper_pb2.py             # Generated protobuf messages
│   └── helper_pb2_grpc.py        # Generated gRPC stubs
├── scripts/
│   ├── backfill_embeddings.py    # Phase 4 backfill (idempotent, byte-parity with backend/cmd/hash_fixture)
│   └── test_byte_parity_gate.sh  # Local harness for the byte-parity CI gate
└── internal/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   └── helper_agent.py       # Domain logic: selects adapter, calls LLM, fallback chain
    ├── ports/
    │   ├── __init__.py
    │   └── llm.py                # LLMPort protocol + Message dataclass
    └── adapters/
        ├── __init__.py
        ├── grpc_server.py        # gRPC server: HelperServicer (Ask/Embed/EmbedBatch) + health HTTP server
        ├── opencode_llm.py       # OpenCode adapter (langchain_openai → OpenAI-compatible API)
        ├── mistral_llm.py        # Mistral adapter (langchain_openai → Mistral API)
        ├── ollama_llm.py         # Ollama adapter (local LLM via raw requests)
        ├── embedding_provider.py # EmbeddingProvider base + OllamaEmbeddingProvider (granite-embedding:278m)
        └── metrics.py            # prometheus_client counters/histograms/gauges
```
