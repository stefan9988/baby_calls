from .llm_interface import LLMInterface


def get_llm_client(client_type: str, **kwargs) -> LLMInterface:
    """
    Factory function to create LLMInterface instances based on client_type.
    
    Args:
        client_type (str): 'openai' or 'huggingface'.
        kwargs: parameters required for the specific LLM client initialization.
                Common ones include:
                    - api_key
                    - model
                    - device
    """
    client_types = ["openai", "huggingface"]

    if client_type == "openai":
        from .openai_api import ChatGPTClient
        return ChatGPTClient(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model"),            
        )

    elif client_type == "huggingface":
        from .huggingface_client import HuggingFaceLLM
        return HuggingFaceLLM(
            model_id=kwargs.get("model"),
            api_key=kwargs.get("api_key"),
            device=kwargs.get("device"),
        )
    elif client_type == "ollama":
        from .ollama_client import OllamaClient
        return OllamaClient(
            model=kwargs.get("model"),
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
            timeout=kwargs.get("timeout", 120),
        )

    else:
        raise ValueError(
            f"Unsupported client_type: {client_type}. Supported types are {client_types}."
        )
