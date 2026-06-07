# 🍕 Helper — Pizza-Only AI Assistant

A minimal, focused AI assistant built with **LangGraph** and **FastAPI** that **only answers questions about pizza**. Powered by [opencode-zen](https://opencode.ai/zen) — a free LLM provider — with zero external dependencies beyond Python, LangChain, and a single API call graph.

---

## Project Overview

This service exposes a single REST endpoint that accepts a user's question, routes it through a **LangGraph StateGraph** with a single LLM node, and returns an answer — but only if the question relates to pizza. The system prompt enforces a strict pizza-only policy:

- ✅ Pizza ingredients, history, recipes, cultural variations, preparation techniques
- ✅ Anything pizza-adjacent (e.g., dough chemistry, cheese types, oven temperatures)
- ❌ Rejects non-pizza questions with a polite refusal and explanation

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
├── pizza_graph.py          # LangGraph StateGraph with pizza-only LLM node
├── Dockerfile              # Container image definition (python:3.12-slim)
├── requirements.txt        # Python dependencies
├── .github/
│   └── workflows/
│       └── docker.yml      # CI/CD pipeline — builds & pushes to GHCR
└── README.md               # This file
```

### File Details

- **`main.py`** — Entry point. Defines FastAPI `app`, `GET /health`, `POST /api/v1/ask`, and the `if __name__ == "__main__"` launcher.
- **`pizza_graph.py`** — Defines `PizzaState` (TypedDict with `question` and `answer`), the `call_llm` node that invokes ChatOpenAI with a pizza-only system prompt, and compiles the graph for export.
- **`Dockerfile`** — Multi-stage not needed; simple `pip install` + `COPY` + `uvicorn run`.
- **`.github/workflows/docker.yml`** — On push/PR to `main`: checks out code, logs into GHCR, extracts metadata (sha, branch, latest tags), builds with BuildKit caching, and pushes the image.

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
Send a question and receive an answer — but only if it's about pizza.

**Request Body**
```json
{
  "question": "What is the best cheese for Neapolitan pizza?"
}
```

**Response `200`**
```json
{
  "answer": "For authentic Neapolitan pizza (VPN-certified), the best cheese is fresh mozzarella di bufala campana DOP — made from water buffalo milk. It has a delicate, milky flavor and melts perfectly in a wood-fired oven at 485°C (905°F). Alternatives include fior di latte (cow's milk mozzarella) for a milder version."
}
```

**Non-pizza question example:**

Request:
```json
{
  "question": "What is the capital of France?"
}
```

Response:
```json
{
  "answer": "I'm sorry, but I can only answer questions about pizza. Please ask me something pizza-related — ingredients, history, recipes, or preparation techniques!"
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

# Pizza question
curl -X POST http://localhost:8082/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I make a Detroit-style pizza crust?"}'

# Non-pizza question (should refuse)
curl -X POST http://localhost:8082/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I fix a car engine?"}'
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

## Behavior Details

The assistant is driven by a single LangGraph node with a **strict system prompt**:

```
You are a strict pizza-only assistant. You ONLY answer questions that are about pizza —
its ingredients, history, recipes, cultural variations, preparation techniques, or anything
pizza-adjacent. If the question is NOT about pizza, politely refuse to answer and explain
that you can only discuss pizza.
```

This means:

- **Pizza topics accepted**: Dough hydration, San Marzano tomatoes, Neapolitan vs. New York style, pineapple debate, cheese stretching properties, wood-fired vs. electric ovens, regional Italian styles, etc.
- **Non-pizza topics rejected**: Any question about non-pizza food, technology, science, history, geography, or any other domain — politely declined.
- **Pizza-adjacent is interpreted generously**: Topics like agriculture (tomato growing), metallurgy (oven materials), or supply chain (flour milling) may be answered if clearly connected to pizza.

---

## License

This project is available under the terms of the repository's license. See the `LICENSE` file if present, or contact the repository owner.

---

*Built with LangGraph, FastAPI, and opencode-zen.*
