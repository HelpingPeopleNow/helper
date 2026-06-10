# helper

Stateless Python gRPC server that processes chat requests through LLM adapters. No HTTP, no database.

## Key facts

- **No tests, no lint, no typecheck** — only CI is Docker build/push (`docker.yml`). The only verification is that `python main.py` starts without error.
- **OpenCode adapter uses `langchain_openai.ChatOpenAI`** (despite README claiming httpx). The `requirements.txt` has `langchain-openai` and `requests`.
- **Ollama adapter uses raw `requests`**, no langchain — full prompt (system+history+user) is concatenated into a single string.
- **JSON format instructions are appended to the user message**, not the system prompt — some providers ignore system formatting.
- **No HTTP health endpoint** despite `HEALTH_PORT=8084` in docker-compose. There's no `/health` or any HTTP listener.

## Architecture

Pure hexagonal: `HelperAgent` (in `internal/core/helper_agent.py`) depends only on `LLMPort` Protocol (`internal/ports/llm.py`). Both adapters implement it. Adapters are loaded once at startup in `main.py`; the backend picks per-request via the `llm_provider` gRPC field.

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
python main.py          # starts gRPC on :50051
```

No `PYTHONPATH` needed when running from the project root.

## Env vars

| Variable | Default | Notes |
|----------|---------|-------|
| `GRPC_PORT` | `50051` | gRPC listen port |
| `LLM_BASE_URL` | `https://opencode.ai/zen/v1` | OpenCode API |
| `LLM_MODEL` | `deepseek-v4-flash-free` | OpenCode model |
| `LLM_API_KEY` | — | Required for OpenCode |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen3:1.7b` | Ollama model |
| `USE_OLLAMA` | `false` | `"true"`/`"1"`/`"yes"` → default to ollama |
