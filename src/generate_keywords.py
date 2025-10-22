from dataset_operations import get_data, create_metadata_file
from llms.llm_factory import get_llm_client
import config

import random
import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

DATA_DIR = "UNS dataset/json_english_v2"
FILE_PATTERN = "*e.json"
NUMBER_OF_SAMPLES = 20
RANDOM_SEED = 42

# random.seed(RANDOM_SEED)

client = get_llm_client(client_type=config.CLIENT_TYPE, 
                        api_key=OPENAI_API_KEY, 
                        model=config.KEYWORD_GENERATOR_LLM_MODEL)

if __name__ == "__main__":
    example_data = get_data(DATA_DIR, FILE_PATTERN)
    sampled_data = random.sample(example_data, min(NUMBER_OF_SAMPLES, len(example_data)))
    keyword_examples = [sample['data']["summary"]["key_words"][0] for sample in sampled_data]
    
    reply = client.conv(
        user_message=f"Generate {NUMBER_OF_SAMPLES} keyword phrases based on the following examples:\n"
        + json.dumps(keyword_examples, indent=4),
        system_message=config.KEYWORD_GENERATOR_SYSTEM_PROMPT,
        temperature=config.KEYWORD_GENERATOR_TEMPERATURE,
        max_tokens=config.KEYWORD_GENERATOR_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    
    json_response = json.loads(reply)
    os.makedirs(os.path.dirname(config.KEYWORDS_PATH), exist_ok=True)
    with open(config.KEYWORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(json_response, f, indent=2, ensure_ascii=False)
    print(f"Saved {NUMBER_OF_SAMPLES} keyword phrases to {config.KEYWORDS_PATH}")
    create_metadata_file(config, filepath=config.METADATA_PATH)
