from dataset_operations import get_data
from openai_api import ChatGPTClient
from config import (
    KEYWORD_GENERATOR_LLM_MODEL,
    KEYWORD_GENERATOR_SYSTEM_PROMPT,
    KEYWORD_GENERATOR_TEMPERATURE,
    KEYWORD_GENERATOR_MAX_TOKENS,
)
import random
import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = "UNS dataset/json_english_v2"
KEYWORDS_PATH = "UNS dataset/json_english_aug/keywords.json"
FILE_PATTERN = "*e.json"
NUMBER_OF_SAMPLES = 20
RANDOM_SEED = 42

# random.seed(RANDOM_SEED)

client = ChatGPTClient(api_key=OPENAI_API_KEY, model=KEYWORD_GENERATOR_LLM_MODEL)

if __name__ == "__main__":
    example_data = get_data(DATA_DIR, FILE_PATTERN)
    sampled_data = random.sample(example_data, min(NUMBER_OF_SAMPLES, len(example_data)))
    keyword_examples = [sample['data']["summary"]["key_words"][0] for sample in sampled_data]
    
    reply = client.conv(
        user_message=f"Generate {NUMBER_OF_SAMPLES} keyword phrases based on the following examples:\n"
        + json.dumps(keyword_examples, indent=4),
        system_message=KEYWORD_GENERATOR_SYSTEM_PROMPT,
        temperature=KEYWORD_GENERATOR_TEMPERATURE,
        max_tokens=KEYWORD_GENERATOR_MAX_TOKENS,
        response_format=({"type": "json_object"}),
    )
    
    json_response = json.loads(reply)
    os.makedirs(os.path.dirname(KEYWORDS_PATH), exist_ok=True)
    with open(KEYWORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(json_response, f, indent=2, ensure_ascii=False)
    print(f"Saved {NUMBER_OF_SAMPLES} keyword phrases to {KEYWORDS_PATH}")
