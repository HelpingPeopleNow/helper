# HelpingPeopleNow Helper

Stateless Python gRPC server that processes chat requests using LLM adapters. Receives questions from the backend via gRPC, picks the right LLM adapter (OpenCode or Ollama) per request, and returns answers with optional user role detection.

**Container:** `helpingpeoplenow-helper` | **gRPC Port:** `:50051` | **Health HTTP Port:** `:8084`

---

## Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.12 |
| **gRPC** | `grpcio` (server only, no HTTP) |
| **LLM Adapters** | OpenAI-compatible (OpenCode via httpx) + local Ollama |
| **Validation** | Pydantic v2 |
| **Container** | `python:3.12-slim` (source-direct, no multi-stage) |

---

## What It Does

1. **Chat completion** — receives an `AskRequest` with question, history, system prompt, and LLM provider selector; returns an answer + detected user role
2. **Dual LLM adapters** — maintains both an OpenCode (external) and Ollama (local) adapter; the backend chooses which one per request via the `llm_provider` field; empty = falls back to the `USE_OLLAMA` env var
3. **Role detection** — the LLM's answer is scanned for worker/client intent; the system prompt instructs the model to return a role tag for classification

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         main.py                               │
│  (entry point: loads adapters, starts gRPC server)            │
│                                                               │
│  adapters = {                                                 │
│      "opencode": OpenCodeLLMAdapter(...),                     │
│      "ollama":  OllamaLLMAdapter(...),                        │
│  }                                                            │
│  default = USE_OLLAMA env var → "opencode" or "ollama"       │
│  assistant = HelperAgent(adapters, default)                   │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────────────────────┐
│                  gRPC Server (:50051)                          │
│               internal/adapters/grpc_server.py                │
│                                                               │
│  HelperService.Ask(request) → response                        │
│    ├─ system_prompt: from backend cache                       │
│    ├─ llm_provider: from admin config or empty (= env default)│
│    └─ calls helper_agent.answer(...)                          │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────────────────────┐
│              internal/core/helper_agent.py                    │
│                                                               │
│  HelperAgent.answer(question, system_prompt, history,         │
│                     llm_provider)                             │
│    │                                                          │
│    ├─ provider = llm_provider or self._default_provider      │
│    ├─ llm = self._adapters[provider]                          │
│    └─ return llm.complete(system_prompt, user, history)       │
└───────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Stateless** — no database connection, no file storage. Everything it needs comes via gRPC from the backend. A lightweight health HTTP endpoint is available for liveness checks.
- **Port-based** — `LLMPort` is a Python `Protocol` class; both adapters implement it. Adding a new provider (e.g., Anthropic, Groq) requires only a new adapter class
- **Runtime provider switching** — the backend chooses the adapter per request. The helper never restarts when the provider changes

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
    llm_provider: "opencode"  // or "ollama" or ""
}
       │
       ▼
grpc_server.py
  ├─ log: provider=opencode (request='opencode' default=opencode)
  └─ helper_agent.answer(...)
       │
       ▼
helper_agent.py
  ├─ provider = "opencode" (from request, overrides default)
  ├─ llm = adapters["opencode"]  → OpenCodeLLMAdapter
  └─ return llm.complete(system_prompt, user, history)
       │
       ▼
OpenCodeLLMAdapter (or OllamaLLMAdapter)
  ├─ build messages: [system, ...history, user]
  ├─ POST /chat/completions (OpenAI-compatible API)
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
  string llm_provider = 4;        // "opencode" | "ollama" | "" (= env default)
}

message AskResponse {
  string answer = 1;
  string detected_role = 2;       // "worker" | "client" | "" (not detected)
}
```

Proto definition: `proto/helper.proto` (shared with the backend repo)

### Health Check

A lightweight HTTP health check server runs alongside the gRPC service, using Python stdlib `http.server` (no extra dependencies).

| Endpoint | Method | Response |
|----------|--------|----------|
| `/health` | GET | `{"status":"ok"}` (200) |

Port is configurable via the `HEALTH_PORT` env var (default: `8084`).

---

## LLM Adapters

### OpenCode (External)

- Connects to `https://opencode.ai/zen/v1` (OpenAI-compatible API)
- Model configurable via `LLM_MODEL` env var
- Uses httpx for HTTP calls (no heavy SDK dependencies)
- Free tier available (`deepseek-v4-flash-free`)

### Ollama (Local)

- Connects to a local Ollama instance at `OLLAMA_BASE_URL`
- Uses local models (runs on CPU — slower but free and private)
- Compatible with any Ollama-served model

### Adding a New Provider

Create a new file `internal/adapters/<provider>_llm.py` that implements `LLMPort`:

```python
from internal.ports.llm import LLMPort, Message

class MyProviderLLMAdapter(LLMPort):
    def complete(self, system_prompt: str, user: str, history: Sequence[Message] = ()) -> str:
        # Your implementation here
        return response_text
```

Then add it in `main.py`:

```python
adapters = {
    "opencode": OpenCodeLLMAdapter(...),
    "ollama": OllamaLLMAdapter(...),
    "myprovider": MyProviderLLMAdapter(...),
}
assistant = HelperAgent(adapters, default=...)
```

The backend sends `llm_provider: "myprovider"` → helper picks the right adapter.

---

## Logging

Structured logging to stdout with timestamps and module names:

| Component | Events |
|-----------|--------|
| `main.py` | Adapter loading, default provider, gRPC server start |
| `grpc_server.py` | Request size, history count, system prompt length, provider selection |
| `opencode_llm.py` / `ollama_llm.py` | Model name, call duration, response size, errors |

Format:

```
2026-06-09 08:15:22 INFO internal.adapters.grpc_server gRPC Ask: q_len=42 history=3 sp_len=1200 provider=opencode
2026-06-09 08:15:22 INFO internal.adapters.opencode_llm LLM call: model=deepseek-v4-flash-free, messages=5, prompt_chars=1800
2026-06-09 08:15:25 INFO internal.adapters.opencode_llm LLM response: elapsed_ms=2950.12 response_chars=312
2026-06-09 08:15:25 INFO internal.adapters.grpc_server gRPC Ask done: answer_len=312 elapsed_ms=2951
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
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (used when provider=ollama) |
| `USE_OLLAMA` | `false` | Default provider: `"true"` or `"1"` → ollama, else opencode |

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
├── main.py                       # Entry point: load adapters, start gRPC server
├── Dockerfile                    # Single-stage: python:3.12-slim
├── requirements.txt              # Python dependencies (grpcio, httpx, pydantic)
├── README.md
├── proto/
│   ├── helper.proto              # Protobuf contract (shared with backend)
│   ├── helper_pb2.py             # Generated protobuf messages
│   └── helper_pb2_grpc.py        # Generated gRPC stubs
└── internal/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   └── helper_agent.py       # Domain logic: selects adapter, calls LLM
    ├── ports/
    │   ├── __init__.py
    │   └── llm.py                # LLMPort protocol + Message dataclass
    └── adapters/
        ├── __init__.py
        ├── grpc_server.py        # gRPC server: HelperServicer
        ├── opencode_llm.py       # OpenCode adapter (httpx → OpenAI-compatible API)
        └── ollama_llm.py         # Ollama adapter (local LLM via HTTP)
```
