# Helper — General-Purpose AI Assistant

A minimal, focused AI assistant built with **LangGraph** and **FastAPI** that answers user questions using a configurable system prompt. Powered by [opencode-zen](https://opencode.ai/zen) — a free LLM provider — with zero external dependencies beyond Python, LangChain, and a single API call graph.

---

## Project Overview

This service exposes a single REST endpoint that accepts a user's question, routes it through a **LangGraph StateGraph** with a single LLM node, and returns an answer using the system prompt configured in the database (`system_prompts.helper_prompt` column).

The graph is intentionally simple (one node) to keep the codebase minimal and easy to extend.

---

## Tech Stack

| Component           | Technology                                              |
|---------------------|---------------------------------------------------------|
| **Runtime**         | Python 3.12                                             |
| **Web Framework**   | FastAPI + uvicorn                                       |
| **Orchestration**   | LangGraph (StateGraph)                                  |
| **LLM Provider**    | langchain-openai (OpenAI-compatible) → opencode-zen     |
| **Model**           | deepseek-v4-flash-free (via opencode-zen)               |
| **Validation**      | Pydantic v2                                             |
| **Container**       | Docker (python:3.12-slim)                               |
| **CI/CD**           | GitHub Actions → ghcr.io                                |

---

## Project Structure

```
helper/
├── main.py                 # FastAPI application — routes, startup
├── internal/
│   ├── core/
│   │   └── helper_agent.py   # HelperAgent domain class
│   ├── ports/
│   │   ├── graph.py              # GraphRunner port
│   │   ├── llm.py                # LLMPort port
│   │   └── prompt_repository.py  # PromptRepository port
│   ├── adapters/
│   │   ├── langgraph_runner.py   # LangGraph implementation
│   │   ├── opencode_llm.py       # OpenCode LLM adapter
│   │   └── postgres_repo.py      # PostgreSQL prompt repository
│   └── api/
│       ├── dependencies.py       # DI composition root
│       └── routes.py             # HTTP route definitions
├── Dockerfile              # Container image definition (python:3.12-slim)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## API Endpoints

### `GET /health`
Returns a simple health-check response.

**Response `200`**
```json
{
  "status": "ok"
}
```

---

### `POST /api/v1/ask`
Send a question and receive an answer based on the configured system prompt.

**Request Body**
```json
{
  "question": "What can you help me with?"
}
```

**Response `200`**
```json
{
  "answer": "I can help you with a wide range of topics..."
}
```

---

## Environment Variables

| Variable         | Default                                     | Description                              |
|------------------|---------------------------------------------|------------------------------------------|
| `LLM_MODEL`      | `deepseek-v4-flash-free`                    | Model name passed to ChatOpenAI          |
| `LLM_BASE_URL`   | `https://opencode.ai/zen/v1`                | OpenAI-compatible API base URL           |
| `LLM_API_KEY`    | `""` (empty)                                | API key for the LLM provider             |
| `PORT`           | `8082`                                      | Port for the uvicorn server              |
| `DB_HOST`        | `postgres`                                  | PostgreSQL host                          |
| `DB_PORT`        | `5432`                                      | PostgreSQL port                          |
| `DB_USER`        | `postgres`                                  | PostgreSQL user                          |
| `DB_PASSWORD`    | `postgres`                                  | PostgreSQL password                      |
| `DB_NAME`        | `helpingpeoplenow`                          | Database name                            |

> **Note:** opencode-zen allows empty API keys for free-tier usage. To use a different provider (e.g., OpenAI, Anthropic via API), set `LLM_BASE_URL` and `LLM_API_KEY` accordingly.

---

## Setup & Running

### Locally (pip)

```bash
# 1. Clone the repository
git clone https://github.com/HelpingPeopleNow/helper.git
cd helper

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn main:app --host 0.0.0.0 --port 8082
```

The API will be available at `http://localhost:8082`.

### Quick test

```bash
# Health check
curl http://localhost:8082/health

# Ask a question
curl -X POST http://localhost:8082/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What can you help me with?"}'
```

---

## Docker Usage

### Build locally

```bash
docker build -t helper .
```

### Run

```bash
docker run -p 8082:8082 \
  -e LLM_MODEL=deepseek-v4-flash-free \
  -e LLM_BASE_URL=https://opencode.ai/zen/v1 \
  helper
```

Or with a different provider:

```bash
docker run -p 8082:8082 \
  -e LLM_MODEL=gpt-4o-mini \
  -e LLM_BASE_URL=https://api.openai.com/v1 \
  -e LLM_API_KEY=sk-... \
  helper
```

---

## CI/CD

On every push (or PR) to the `main` branch, **GitHub Actions** automatically:

1. Checks out the repository
2. Logs into **GitHub Container Registry** (`ghcr.io`)
3. Extracts Docker metadata (SHA tag, branch tag, `latest` on default branch)
4. Builds the image with **BuildKit layer caching** (GitHub Actions cache)
5. Pushes the image to `ghcr.io/HelpingPeopleNow/helper`

The workflow file is located at `.github/workflows/docker.yml`.

---

## System Prompt

The assistant's behavior is determined by the system prompt stored in the `system_prompts.helper_prompt` column in PostgreSQL. You can edit it from the frontend System Prompts section — changes take effect immediately.

If no row exists in the `system_prompts` table, a default prompt is used.

---

*Built with LangGraph, FastAPI, and opencode-zen.*
