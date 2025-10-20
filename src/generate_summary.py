from dataset_operations import save_summaries
from openai_api import ChatGPTClient
from config import (
    SUMMARY_GENERATOR_LLM_MODEL,
    SUMMARY_GENERATOR_SYSTEM_PROMPT,
    SUMMARY_GENERATOR_TEMPERATURE,
    SUMMARY_GENERATOR_MAX_TOKENS,
)

import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

KEYWORDS_PATH = "UNS dataset/json_english_aug/keywords.json"
SUMMARIES_OUTPUT_DIR = "UNS dataset/json_english_aug"
BATCH_SIZE = 10
NUMBER_OF_SUMMARIES_PER_KEYWORD = 2

client = ChatGPTClient(api_key=OPENAI_API_KEY, model=SUMMARY_GENERATOR_LLM_MODEL)

if __name__ == "__main__":
    if os.path.exists(KEYWORDS_PATH):
        with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
            keywords = json.load(f)
        print(f"✅ Loaded {len(keywords['keywords'])} keywords")
    else:
        print("⚠️ File not found:", KEYWORDS_PATH)

    batch = [
        keywords["keywords"][i : i + BATCH_SIZE]
        for i in range(0, len(keywords["keywords"]), BATCH_SIZE)
    ]

    summaries = []
    for i, keywords in enumerate(batch):
        print(f"Generating summaries for batch {i + 1}/{len(batch)}")
        reply = client.conv(
            user_message=f"""Generate {NUMBER_OF_SUMMARIES_PER_KEYWORD} different summaries per keyword.
            Change context for each summary while keeping it realistic.
            Here are the keywords:\n"""
            + json.dumps(keywords, indent=4),
            system_message=SUMMARY_GENERATOR_SYSTEM_PROMPT,
            temperature=SUMMARY_GENERATOR_TEMPERATURE,
            max_tokens=SUMMARY_GENERATOR_MAX_TOKENS,
            response_format=({"type": "json_object"}),
        )

        json_response = json.loads(reply)
        batch_summaries = json_response.get("summaries", [])
        summaries.extend(batch_summaries)

    save_summaries(summaries=summaries, output_dir=SUMMARIES_OUTPUT_DIR, suffix="e.json")
