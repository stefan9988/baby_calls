from openai import OpenAI

class ChatGPTClient:
    """
    A simple wrapper around OpenAI's Chat Completions API.

    Example:
        client = ChatGPTClient(api_key="YOUR_API_KEY")
        response = client.ask(
            user_message="Write a haiku about the sea.",
            system_message="You are a poetic assistant.",
            temperature=0.8,
        )
        print(response)
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API key not provided.")
        self.model = model
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
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content.strip()
