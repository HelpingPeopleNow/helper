# Helper — General-Purpose AI Assistant

FastAPI + LangGraph assistant with hexagonal architecture. Provides chat completions via both HTTP and gRPC endpoints.

**HTTP Port:** `:8082` | **gRPC Port:** `:50051`

## Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.12 |
| **Framework** | FastAPI + uvicorn |
| **gRPC** | grpcio (server) |
| **LLM** | OpenCode Zen (OpenAI-compatible) via langchain-openai |
| **Graph** | LangGraph StateGraph |
| **DB** | PostgreSQL 16 via psycopg[binary] |
| **Validation** | Pydantic v2 |
| **Container** | python:3.12-slim |
| **CI/CD** | GitHub Actions → ghcr.io |

## Architecture (Hexagonal / Ports & Adapters)

```
HTTP POST /api/v1/ask ──► FastAPI routes ──► Dependencies (DI)
  └── gRPC :50051 ──► HelperServicer ──► HelperAgent (domain)
                                              │
                       ┌──────────────────────┼──────────────────────┐
                       ▼                      ▼                      ▼
              PromptRepository         LLMPort               GraphRunner
              (port/protocol)         (port/protocol)       (port/protocol)
                       │                      │                      │
                       ▼                      ▼                      ▼
              PostgresPromptRepo     OpenCodeLLMAdapter      LangGraphRunner
              (adapter)              (adapter)               (adapter)
                       │                      │
                       ▼                      ▼
                   PostgreSQL            OpenCode Zen API
```

**Layer rules:**
- **Core** (`core/`) — pure domain logic, zero dependencies (dataclasses only)
- **Ports** (`ports/`) — abstract protocols (Python `Protocol`)
- **Adapters** (`adapters/`) — concrete implementations (LLM, DB, gRPC, LangGraph)
- **API** (`api/`) — HTTP routes and dependency injection

## Request Flow

1. HTTP `POST /api/v1/ask` → FastAPI validates with Pydantic
2. DI container (`dependencies.py`) wires `OpenCodeLLMAdapter` + `PostgresPromptRepository` into `HelperAgent`
3. `HelperAgent.answer()` fetches system prompt from DB, then calls LLM
4. Response flows back through the same chain

Same flow applies to gRPC `Ask` RPC on `:50051`.

## API Endpoints

### HTTP

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{"status":"ok"}` |
| POST | `/api/v1/ask` | Ask a question — `{"question":"...", "history":[]}` → `{"answer":"..."}` |

```bash
curl -X POST http://localhost:8082/api/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What can you help me with?"}'
```

### gRPC

```protobuf
service HelperService {
  rpc Ask(AskRequest) returns (AskResponse);
}
```

Protobuf definition in `proto/helper.proto`. Go stubs are shared with the backend repo.

## System Prompts

System prompts are stored in the `system_prompts` PostgreSQL table (singleton row, `id=1`). The `helper_prompt` column controls the assistant's behavior. If no row exists, a default prompt is used.

To change the helper's personality, update the Helper Prompt via the frontend Admin page (`/admin`).

## Logging

All components log to stdout with timestamps, levels, and module names:

| Component | Events Logged |
|-----------|--------------|
| `main.py` | Startup, route registration, gRPC initialization |
| `routes.py` | Request size, history count, response timing, errors |
| `grpc_server.py` | Bind status, request size, response timing, errors |
| `opencode_llm.py` | Model config, message count, prompt size, call duration, errors |
| `postgres_repo.py` | Connection info, prompt load size, defaults fallback |

Format:
```
2026-06-08 15:41:49 INFO internal.api.routes ask request: q_len=5 history=0
2026-06-08 15:41:50 INFO internal.adapters.opencode_llm LLM response: elapsed_ms=1234.56 response_chars=456
2026-06-08 15:41:50 INFO internal.api.routes ask response: answer_len=456 elapsed_ms=1567
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8082` | HTTP listen port |
| `LLM_MODEL` | `deepseek-v4-flash-free` | OpenCode model |
| `LLM_BASE_URL` | `https://opencode.ai/zen/v1` | OpenCode API base |
| `LLM_API_KEY` | (required) | OpenCode API key |
| `DB_HOST` | `postgres` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_USER` | `postgres` | DB user |
| `DB_PASSWORD` | `postgres` | DB password |
| `DB_NAME` | `helpingpeoplenow` | DB name |
| `DB_SSLMODE` | `disable` | SSL mode |

## Development

```bash
pip install -r requirements.txt
python main.py                    # Run locally (port 8082)
docker build -t helper .          # Docker build
docker run -p 8082:8082 helper    # Run container
```

## Project Structure

```
helper/
├── main.py                       # Entry point, FastAPI app, gRPC startup
├── Dockerfile                    # python:3.12-slim
├── requirements.txt              # Python dependencies
├── README.md
├── proto/
│   ├── helper.proto              # Protobuf contract
│   ├── helper_pb2.py             # Generated protobuf messages
│   └── helper_pb2_grpc.py        # Generated gRPC stub/server
└── internal/
    ├── core/
    │   └── helper_agent.py       # Domain aggregate: HelperAgent
    ├── ports/
    │   ├── llm.py                # LLMPort protocol
    │   ├── prompt_repository.py  # PromptRepository protocol
    │   └── graph.py              # GraphRunner protocol
    ├── api/
    │   ├── routes.py             # HTTP endpoints
    │   └── dependencies.py       # Composition root (DI)
    └── adapters/
        ├── opencode_llm.py       # LLM adapter → OpenCode Zen API
        ├── postgres_repo.py      # DB adapter → PostgreSQL
        ├── grpc_server.py        # gRPC server adapter
        └── langgraph_runner.py   # LangGraph StateGraph wrapper
```
