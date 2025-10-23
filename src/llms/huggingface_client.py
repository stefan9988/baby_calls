from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from .llm_interface import LLMInterface


class HuggingFaceLLM(LLMInterface):
    """
    Implementation of LLMInterface for Hugging Face transformer models.
    """

    def __init__(self, model_id: str, api_key: str = None, device: str = None):
        """
        Args:
            model_id: Hugging Face model name or local path
            api_key: Hugging Face token 
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        
        super().__init__(api_key=api_key, model=model_id)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Pass the token for private repo access
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=api_key)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, token=api_key).to(self.device)

    def conv(
        self,
        user_message: str,
        system_message: str = "",
        temperature: float = 0.7,
        max_tokens: int = 200,
        **kwargs,
    ) -> str:
        """
        Generate text using a Hugging Face model (EuroLLM, Llama, Mistral, etc.)
        """
        prompt = f"{system_message}\n{user_message}" if system_message else user_message

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            # **kwargs,
        )
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
