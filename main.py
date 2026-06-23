"""
Application entry point — starts gRPC server.

Loads all LLM adapters at startup. The backend selects which one to use
per-request via the llm_provider gRPC field. Empty = default fallback chain:
Mistral → OpenCode 1 → OpenCode 2 → Ollama.

Also loads the embedding provider (VECTOR_SEARCH_PLAN §7.4) so the
server can serve Embed/EmbedBatch RPCs alongside the chat path.
"""
import logging
import os

from internal.adapters.embedding_provider import OllamaEmbeddingProvider
from internal.adapters.grpc_server import configure_health_handler, serve_grpc, serve_health
from internal.adapters.mistral_llm import MistralLLMAdapter
from internal.adapters.opencode_llm import OpenCodeLLMAdapter
from internal.adapters.ollama_llm import OllamaLLMAdapter
from internal.core.helper_agent import HelperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

def require_env(key: str) -> str:
    v = os.getenv(key, "").strip()
    if not v:
        logger.error("FATAL: missing required environment variable: %s", key)
        raise SystemExit(1)
    return v

def main():
    logger.info("=== Helper Service Starting ===")

    require_env("LLM_API_KEY")
    require_env("LLM_BASE_URL")
    require_env("LLM_MODEL")
    require_env("GRPC_PORT")
    require_env("HEALTH_PORT")

    adapters = {
        "opencode1": OpenCodeLLMAdapter(model="deepseek-v4-flash-free"),
        "opencode2": OpenCodeLLMAdapter(model="mimo-v2.5-free"),
        "ollama": OllamaLLMAdapter(),
    }
    adapter_details = {
        "opencode1": {
            "kind": "openai_compat",
            "base_url": os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/v1"),
            "api_key": os.getenv("LLM_API_KEY", ""),
            "model": "deepseek-v4-flash-free",
        },
        "opencode2": {
            "kind": "openai_compat",
            "base_url": os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/v1"),
            "api_key": os.getenv("LLM_API_KEY", ""),
            "model": "mimo-v2.5-free",
        },
        "ollama": {
            "kind": "ollama",
            "base_url": os.getenv("OLLAMA_BASE_URL", ""),
            "model": os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b"),
        },
    }
    if os.getenv("MISTRAL_API_KEY", "").strip():
        adapters["mistral"] = MistralLLMAdapter(model="mistral-large-latest")
        adapter_details["mistral"] = {
            "kind": "openai_compat",
            "base_url": os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
            "api_key": os.getenv("MISTRAL_API_KEY", ""),
            "model": "mistral-large-latest",
        }
    else:
        logger.warning("MISTRAL_API_KEY not set; skipping Mistral adapter")
    logger.info("Loaded LLM adapters: %s", list(adapters.keys()))

    # VECTOR_SEARCH_PLAN §7.4: embedding provider.
    # Reads EMBEDDING_MODEL env var (default granite-embedding:278m) and
    # OLLAMA_BASE_URL (same var the LLM adapters use — it's the same daemon).
    embedding_provider = OllamaEmbeddingProvider(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=os.getenv("EMBEDDING_MODEL", "granite-embedding:278m"),
    )
    adapter_details["embedding"] = {
        "kind": "embedding",
        "base_url": os.getenv("OLLAMA_BASE_URL", ""),
        "model": embedding_provider.model,
    }
    logger.info("Embedding provider ready: model=%s", embedding_provider.model)

    assistant = HelperAgent(adapters=adapters)
    logger.info("HelperAgent initialized (no DB dependency, multi-adapter)")
    logger.info("Default fallback chain: %s", HelperAgent.FALLBACK_CHAIN)

    grpc_port = int(os.getenv("GRPC_PORT", "50051"))
    logger.info("Starting gRPC server on port %d", grpc_port)
    grpc_server = serve_grpc(assistant, embedding_provider=embedding_provider, port=grpc_port)
    logger.info("=== Helper Service Ready ===")

    configure_health_handler(
        adapter_names=list(adapters.keys()) + ["embedding"],
        grpc_server=grpc_server,
        adapter_details=adapter_details,
    )

    health_port = int(os.getenv("HEALTH_PORT", "8084"))
    serve_health(port=health_port)
    logger.info("Health HTTP server on :%d", health_port)

    grpc_server.wait_for_termination()

if __name__ == "__main__":
    main()
