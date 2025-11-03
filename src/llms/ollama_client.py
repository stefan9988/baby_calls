import requests
from typing import Any, Dict, Optional
from .llm_interface import LLMInterface


class OllamaClient(LLMInterface):
    """
    Ollama client implementing LLMInterface.

    Notes:
    - Ollama runs locally; an API key is not required (kept for interface parity).
    - Set OLLAMA_BASE_URL to override the default base URL.
    - Maps:
        temperature -> options.temperature
        max_tokens  -> options.num_predict
      Pass extra generation options via **kwargs (e.g., top_p, seed, repeat_penalty, format="json").
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        Initialize Ollama client for local LLM inference.

        Args:
            model: Ollama model name (e.g., 'llama2', 'mistral')
            api_key: Not used for Ollama, kept for interface compatibility
            base_url: Ollama server URL (default: "http://localhost:11434")
            timeout: Request timeout in seconds (default: 120)
        """
        super().__init__(api_key or "", model)
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.timeout = timeout

    def conv(
        self,
        user_message: str,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using Ollama's local inference API.

        Sends a chat request to the Ollama server with system and user messages.
        Handles various response formats from different Ollama versions.

        Args:
            user_message: The user's input message
            system_message: System prompt to set behavior (default: "You are a helpful assistant.")
            temperature: Sampling temperature for generation (0.0-2.0)
            max_tokens: Maximum tokens to generate (-1 for unlimited)
            **kwargs: Additional Ollama options (e.g., top_p, seed, format)

        Returns:
            str: Model's response text

        Raises:
            RuntimeError: If HTTP request fails or Ollama returns an error
        """
        # Build Ollama options from known params + passthrough kwargs
        options: Dict[str, Any] = {
            "temperature": float(temperature),
            "num_predict": int(max_tokens) if max_tokens is not None else -1,
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "options": options,
            "stream": False,
        }

        url = f"{self.base_url}/api/chat"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            # Typical shape: {"message": {"role": "...","content": "..."}, "done": true, ...}
            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict):
                    return data["message"].get("content", "")

                # Fallbacks for other shapes
                if "response" in data and isinstance(data["response"], str):
                    return data["response"]

                if "messages" in data and isinstance(data["messages"], list):
                    return "".join(m.get("content", "") for m in data["messages"])

            # Last resort: stringify
            return str(data)

        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error {e.response.status_code}: {e.response.text}") from e
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e
