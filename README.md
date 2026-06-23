# HelpingPeopleNow Helper

Stateless Python gRPC server that processes chat requests using LLM adapters. Receives questions from the backend via gRPC, picks the right LLM adapter per request, and returns answers with optional user role detection.

**Container:** `helpingpeoplenow-helper` | **gRPC Port:** `:50051` | **Health HTTP Port:** `:8084`

---

## Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.12 |
| **gRPC** | `grpcio` (server only) |
| **LLM Adapters** | langchain-openai (OpenCode, Mistral) + raw requests (Ollama) |
| **Validation** | Pydantic v2 |
| **Container** | `python:3.12-slim` (source-direct, no multi-stage) |

---

## What It Does

1. **Chat completion** — receives an `AskRequest` with question, history, system prompt, and LLM provider selector; returns an answer + detected user role
2. **Multi-adapter fallback** — maintains OpenCode, Mistral, and Ollama adapters; the backend chooses which one per request via the `llm_provider` field; empty = automatic fallback chain (Mistral → OpenCode 1 → OpenCode 2 → Ollama)
3. **Role detection** — the LLM's answer is scanned for worker/client intent; the system prompt instructs the model to return a role tag for classification; `skip_role_detection` flag suppresses this for profile intake chats
4. **Embeddings (vector search)** — exposes `Embed` and `EmbedBatch` gRPC methods backed by an Ollama-hosted embedding model (`granite-embedding:278m`). Used by the backend to re-embed worker profile field changes for vector search.
5. **Health + metrics sidecar** — HTTP `:8084` serves liveness probes AND Prometheus metrics; same port, separate handlers

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         main.py                               │
│  (entry point: loads adapters, starts gRPC + health server)   │
│                                                               │
│  adapters = {                                                 │
│      "opencode1": OpenCodeLLMAdapter(deepseek-v4-flash-free), │
│      "opencode2": OpenCodeLLMAdapter(mimo-v2.5-free),        │
│      "mistral":   MistralLLMAdapter(mistral-large-latest),   │
│      "ollama":    OllamaLLMAdapter(),                         │
│  }                                                            │
│  fallback_chain = ["mistral", "opencode1", "opencode2", "ollama"]│
│  assistant = HelperAgent(adapters, fallback_chain)            │
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
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────────────────────┐
│              internal/core/helper_agent.py                    │
│                                                               │
│  HelperAgent.answer(question, system_prompt, history,         │
│                     llm_provider, skip_role_detection)        │
│    │                                                          │
│    ├─ builds provider chain: llm_provider or fallback_chain   │
│    ├─ for each provider: try adapter.complete(...)            │
│    │   └─ on failure: fall through to next provider           │
│    └─ parse response: JSON {"answer","role"} or raw text      │
└───────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Stateless** — no database connection, no file storage. Everything it needs comes via gRPC from the backend. A lightweight health HTTP endpoint is available for liveness checks.
- **Port-based** — `LLMPort` is a Python `Protocol` class; all adapters implement it. Adding a new provider requires only a new adapter class
- **Runtime provider switching** — the backend chooses the adapter per request. The helper never restarts when the provider changes
- **Fallback chain** — when no explicit provider is set (or the chosen one fails), adapters are tried in order: Mistral → OpenCode 1 → OpenCode 2 → Ollama

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
    llm_provider: "opencode1"  // or "mistral", "ollama", ""
    skip_role_detection: false  // true for worker/client intake chats
}
       │
       ▼
grpc_server.py
  ├─ log: provider=opencode1 (request='opencode1' default=mistral)
  └─ helper_agent.answer(...)
       │
       ▼
helper_agent.py
  ├─ provider = "opencode1" (from request, overrides default)
  ├─ llm = adapters["opencode1"]  → OpenCodeLLMAdapter
  └─ return llm.complete(system_prompt, user, history)
       │
       ▼
OpenCodeLLMAdapter (or MistralLLMAdapter, or OllamaLLMAdapter)
  ├─ build messages: [system, ...history, user]
  ├─ call LLM API (OpenAI-compatible)
  └─ return assistant text
       │
       ▼
AskResponse {
    answer: "I can help you find plumbing services...",
    detected_role: "client"  // parsed from answer text
}
```

---

## gRPC Contract

```protobuf
service HelperService {
  rpc Ask(AskRequest) returns (AskResponse);
}

message AskRequest {
  string question = 1;
  repeated Message history = 2;   // chat history context
  string system_prompt = 3;       // loaded by backend from DB
  string llm_provider = 4;        // "opencode1" | "opencode2" | "mistral" | "ollama" | "" (= fallback chain)
  bool skip_role_detection = 5;   // if true, don't append JSON role-detection instructions
}

message AskResponse {
  string answer = 1;
  string detected_role = 2;       // "worker" | "client" | "" (not detected)
}
```

Proto definition: `proto/helper.proto` (shared with the backend repo)

### HTTP sidecar (health + metrics)

A lightweight HTTP server runs alongside the gRPC service on a separate port, using Python stdlib `http.server` (no extra dependencies). It serves both health checks AND Prometheus metrics.

| Endpoint | Method | Response |
|----------|--------|----------|
| `/health` | GET | JSON. `200 → status: ok` if gRPC server is up and at least one LLM adapter is reachable; `503 → status: degraded` if not. Includes `grpc`, `adapters`, `adapter_results`, `adapter_details`, and `loaded_adapters`. |
| `/metrics` | GET | Prometheus text format (`prometheus_client`). Includes `llm_requests_total`, `llm_request_duration_seconds`, `llm_errors_total`, `llm_tokens_total`, `grpc_requests_total`, `grpc_request_duration_seconds`, `health_check_total`, `active_requests`. |

Port is configurable via the `HEALTH_PORT` env var (default: `8084`).

The embedding model check is non-blocking — `health_check_total{target="ollama_embed"}` reflects its state without downgrading the overall status.

---

## LLM Adapters

### OpenCode (External) — 2 instances

Both use `langchain_openai.ChatOpenAI` (OpenAI-compatible API via httpx under the hood):

| Adapter | Model | Purpose |
|---------|-------|---------|
| `opencode1` | `deepseek-v4-flash-free` | Primary OpenCode model |
| `opencode2` | `mimo-v2.5-free` | Secondary OpenCode model |

Base URL: `https://opencode.ai/zen/v1` (configurable via `LLM_BASE_URL`)

### Mistral (External)

Uses `langchain_openai.ChatOpenAI` (Mistral's API is OpenAI-compatible):

| Adapter | Model | Purpose |
|---------|-------|---------|
| `mistral` | `mistral-large-latest` | Mistral's flagship model |

Base URL: `https://api.mistral.ai/v1` (configurable via `MISTRAL_BASE_URL`)
Requires `MISTRAL_API_KEY` env var.

### Ollama (Local)

- Connects to a local Ollama instance at `OLLAMA_BASE_URL`
- Uses raw `requests` (no langchain) — full prompt (system+history+user) is concatenated into a single string
- Model: configurable via `OLLAMA_MODEL` (default: `qwen3.5:0.8b`)

### Embedding provider

Used by `Embed` and `EmbedBatch` gRPC methods (vector search backend path). Currently a single Ollama-backed adapter:

| Adapter | Model | Base URL | Purpose |
|---------|-------|----------|---------|
| `embedding` | `granite-embedding:278m` | `OLLAMA_BASE_URL` | VECTOR_SEARCH_PLAN §7.4 — produces vectors for `worker_embeddings`. |

Same `OLLAMA_BASE_URL` as the Ollama LLM adapter (same daemon). Model is configurable via `EMBEDDING_MODEL`.

The proto definitions for `Embed` / `EmbedBatch` are in `proto/helper.proto`; both methods also report dimensions and the actual model used.

If you'd like to swap providers, replace the `OllamaEmbeddingProvider` in `main.py` with another adapter that implements the `EmbeddingProvider` Protocol (`internal/adapters/embedding_provider.py`).

### Fallback Chain

When no explicit `llm_provider` is set (or the chosen provider fails), adapters are tried in order:

```
Mistral → OpenCode 1 → OpenCode 2 → Ollama
```

If an explicit provider is set, it's tried first, then the remaining providers in fallback order.

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
    "opencode1": OpenCodeLLMAdapter(...),
    "opencode2": OpenCodeLLMAdapter(...),
    "mistral": MistralLLMAdapter(...),
    "ollama": OllamaLLMAdapter(...),
    "myprovider": MyProviderLLMAdapter(...),
}
```

The backend sends `llm_provider: "myprovider"` → helper picks the right adapter.

---

## Logging

Structured logging to stdout with timestamps and module names:

| Component | Events |
|-----------|--------|
| `main.py` | Adapter loading, default provider, gRPC server start |
| `grpc_server.py` | Request size, history count, system prompt length, provider selection, skip_role_detection |
| `opencode_llm.py` | Model name, call duration, response size, errors |
| `mistral_llm.py` | Model name, call duration, response size, errors |
| `ollama_llm.py` | Model name, call duration, response size, errors |

Format:

```
2026-06-12 08:15:22 INFO internal.adapters.grpc_server gRPC Ask: q_len=42 history=3 sp_len=1200 provider=opencode1 skip_role=False
2026-06-12 08:15:22 INFO internal.adapters.opencode_llm LLM call: model=deepseek-v4-flash-free, messages=5, prompt_chars=1800
2026-06-12 08:15:25 INFO internal.adapters.opencode_llm LLM response: elapsed_ms=2950.12 response_chars=312
2026-06-12 08:15:25 INFO internal.adapters.grpc_server gRPC Ask done: answer_len=312 elapsed_ms=2951
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_PORT` | `50051` | gRPC listen port |
| `HEALTH_PORT` | `8084` | Health check HTTP listen port |
| `LLM_BASE_URL` | `https://opencode.ai/zen/v1` | OpenCode API base URL |
| `LLM_MODEL` | `deepseek-v4-flash-free` | OpenCode model name |
| `LLM_API_KEY` | (required) | OpenCode API key |
| `MISTRAL_API_KEY` | — | Mistral API key (optional; if not set, Mistral adapter is skipped) |
| `MISTRAL_MODEL` | `mistral-large-latest` | Mistral model name |
| `MISTRAL_BASE_URL` | `https://api.mistral.ai/v1` | Mistral API base URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (used by `ollama` provider AND the embedding adapter — same daemon) |
| `OLLAMA_MODEL` | `qwen3.5:0.8b` | Ollama chat model name |
| `EMBEDDING_MODEL` | `granite-embedding:278m` | Ollama embedding model name (vector search) |

---

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (needs backend running for gRPC calls)
python main.py

# Docker build
docker build -t ghcr.io/helpingpeoplenow/helper:latest .
```

---

## Project Structure

```
helper/
├── main.py                       # Entry point: load adapters, start gRPC + health server
├── Dockerfile                    # Single-stage: python:3.12-slim
├── requirements.txt              # Python dependencies (grpcio, langchain-openai, requests)
├── README.md
├── proto/
│   ├── helper.proto              # Protobuf contract (shared with backend)
│   ├── helper_pb2.py             # Generated protobuf messages
│   └── helper_pb2_grpc.py        # Generated gRPC stubs
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
        ├── grpc_server.py        # gRPC server: HelperServicer + health HTTP server
        ├── opencode_llm.py       # OpenCode adapter (langchain_openai → OpenAI-compatible API)
        ├── mistral_llm.py        # Mistral adapter (langchain_openai → Mistral API)
        └── ollama_llm.py         # Ollama adapter (local LLM via raw requests)
```
