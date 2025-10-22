from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """
    Abstract base class for any Large Language Model client.
    Defines a consistent interface across multiple providers (OpenAI, Anthropic, Azure, etc.).
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def conv(
        self,
        user_message: str,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs,
    ) -> str:
        """
        Send a prompt to the LLM and return its response.
        Must be implemented by subclasses.
        """
        pass
