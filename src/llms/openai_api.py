from openai import OpenAI
from .llm_interface import LLMInterface


class ChatGPTClient(LLMInterface):
    """
    Implementation of LLMInterface for OpenAI's ChatGPT models.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize OpenAI ChatGPT client.

        Args:
            api_key: OpenAI API key
            model: Model identifier (default: "gpt-4o")
        """
        super().__init__(api_key, model)
        self.client = OpenAI(api_key=self.api_key)

    def conv(
        self,
        user_message: str,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs,
    ) -> str:
        """
        Send a message to ChatGPT and return the model's response.

        Args:
            user_message: The user's input message
            system_message: System prompt to set behavior (default: "You are a helpful assistant.")
            temperature: Sampling temperature 0.0-2.0 (currently not used)
            max_tokens: Maximum tokens in response
            **kwargs: Additional OpenAI API parameters (e.g., response_format)

        Returns:
            str: Model's response text, stripped of whitespace
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            # temperature=temperature,
            max_completion_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content.strip()
