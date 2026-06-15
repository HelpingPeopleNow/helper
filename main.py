"""Application entry point — starts gRPC server.

Loads all LLM adapters at startup. The backend selects which one to use
per-request via the llm_provider gRPC field. Empty = default fallback chain:
Mistral → OpenCode 1 → OpenCode 2 → Ollama.
"""
import logging
import os

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

    # Validate required env vars before doing anything
    require_env("LLM_API_KEY")
    require_env("LLM_BASE_URL")
    require_env("LLM_MODEL")
    require_env("GRPC_PORT")
    require_env("HEALTH_PORT")

    # Load ALL adapters so the backend can switch per-request
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
    logger.info("Loaded adapters: %s", list(adapters.keys()))

    assistant = HelperAgent(adapters=adapters)
    logger.info("HelperAgent initialized (no DB dependency, multi-adapter)")

    # Default fallback chain: Mistral → OpenCode 1 → OpenCode 2 → Ollama
    logger.info("Default fallback chain: %s", HelperAgent.FALLBACK_CHAIN)

    grpc_port = int(os.getenv("GRPC_PORT", "50051"))
    logger.info("Starting gRPC server on port %d", grpc_port)
    grpc_server = serve_grpc(assistant, port=grpc_port)
    logger.info("=== Helper Service Ready ===")

    # Configure health handler with dependency info, then start it last
    configure_health_handler(
        adapter_names=list(adapters.keys()),
        grpc_server=grpc_server,
        adapter_details=adapter_details,
    )

    # Start lightweight health HTTP endpoint
    health_port = int(os.getenv("HEALTH_PORT", "8084"))
    serve_health(port=health_port)
    logger.info("Health HTTP server on :%d", health_port)

    grpc_server.wait_for_termination()

if __name__ == "__main__":
    main()
