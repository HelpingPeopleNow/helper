"""Application entry point — starts only the gRPC server.

Loads all LLM adapters at startup. The backend selects which one to use
per-request via the llm_provider gRPC field. Empty = default fallback chain:
Mistral → OpenCode 1 → OpenCode 2 → Ollama.
"""
import logging
import os

from internal.adapters.grpc_server import serve_grpc, serve_health
from internal.adapters.mistral_llm import MistralLLMAdapter
from internal.adapters.opencode_llm import OpenCodeLLMAdapter
from internal.adapters.ollama_llm import OllamaLLMAdapter
from internal.core.helper_agent import HelperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Helper Service Starting ===")

    # Load ALL adapters so the backend can switch per-request
    adapters = {
        "opencode1": OpenCodeLLMAdapter(model="deepseek-v4-flash-free"),
        "opencode2": OpenCodeLLMAdapter(model="mimo-v2.5-free"),
        "mistral": MistralLLMAdapter(model="mistral-large-latest"),
        "ollama": OllamaLLMAdapter(),
    }
    logger.info("Loaded adapters: %s", list(adapters.keys()))

    assistant = HelperAgent(adapters=adapters)
    logger.info("HelperAgent initialized (no DB dependency, multi-adapter)")

    # Default fallback chain: Mistral → OpenCode 1 → OpenCode 2 → Ollama
    logger.info("Default fallback chain: %s", HelperAgent.FALLBACK_CHAIN)

    grpc_port = int(os.getenv("GRPC_PORT", "50051"))
    logger.info("Starting gRPC server on port %d", grpc_port)
    server = serve_grpc(assistant, port=grpc_port)
    logger.info("=== Helper Service Ready ===")

    # Start lightweight health HTTP endpoint
    health_port = int(os.getenv("HEALTH_PORT", "8084"))
    serve_health(port=health_port)
    logger.info("Health HTTP server on :%d", health_port)

    server.wait_for_termination()


if __name__ == "__main__":
    main()
