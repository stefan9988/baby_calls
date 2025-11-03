import os
from .llm_interface import LLMInterface
from dotenv import load_dotenv

load_dotenv(override=True)


def get_llm_client(client_type: str, **kwargs) -> LLMInterface:
    """
    Factory function to create LLM client instances based on provider type.

    Supports multiple LLM providers with unified interface. Automatically
    loads API keys from environment variables if not provided.

    Args:
        client_type: Provider name - 'openai', 'huggingface', or 'ollama'
        **kwargs: Provider-specific parameters:
            - api_key (str, optional): API key (falls back to env vars)
            - model (str): Model identifier or name
            - device (str, optional): For HuggingFace - 'cuda' or 'cpu'
            - base_url (str, optional): For Ollama - API endpoint URL
            - timeout (int, optional): Request timeout in seconds

    Returns:
        LLMInterface: Configured client instance implementing LLMInterface

    Raises:
        ValueError: If client_type is not supported

    Examples:
        >>> client = get_llm_client('openai', model='gpt-4')
        >>> response = client.conv("Hello", "You are helpful")
        >>>
        >>> client = get_llm_client('ollama', model='llama2', base_url='http://localhost:11434')
    """
    client_types = ["openai", "huggingface", "ollama"]

    if client_type == "openai":
        from .openai_api import ChatGPTClient

        return ChatGPTClient(
            api_key=kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"),
            model=kwargs.get("model"),
        )

    elif client_type == "huggingface":
        from .huggingface_client import HuggingFaceLLM

        return HuggingFaceLLM(
            model_id=kwargs.get("model"),
            api_key=kwargs.get("api_key") or os.getenv("HUGGINGFACE_API_KEY"),
            device=kwargs.get("device"),
        )
    elif client_type == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(
            model=kwargs.get("model"),
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL"),
            timeout=kwargs.get("timeout", 120),
        )

    else:
        raise ValueError(
            f"Unsupported client_type: {client_type}. Supported types are {client_types}."
        )
