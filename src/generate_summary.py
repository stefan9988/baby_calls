from dataset_operations import create_metadata_file, save_summaries
from llms.llm_factory import get_llm_client
from utils import convert_response_to_json
import config

import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

BATCH_SIZE = 10
NUMBER_OF_SUMMARIES_PER_KEYWORD = 2

client = get_llm_client(
    client_type=config.CLIENT_TYPE,
    api_key=OPENAI_API_KEY,
    model=config.SUMMARY_GENERATOR_LLM_MODEL,
    timeout=600,
)

if __name__ == "__main__":
    if os.path.exists(config.KEYWORDS_PATH):
        with open(config.KEYWORDS_PATH, "r", encoding="utf-8") as f:
            keywords = json.load(f)
        print(f"✅ Loaded {len(keywords['keywords'])} keywords")
    else:
        print("⚠️ File not found:", config.KEYWORDS_PATH)
        exit(1)

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
            system_message=config.SUMMARY_GENERATOR_SYSTEM_PROMPT,
            temperature=config.SUMMARY_GENERATOR_TEMPERATURE,
            max_tokens=config.SUMMARY_GENERATOR_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        json_response = convert_response_to_json(reply)
        if not json_response:
            print("❌ Failed to generate summaries for this batch. Skipping.")
            continue

        batch_summaries = json_response.get("summaries", [])
        summaries.extend(batch_summaries)
        print(f"✅ Generated summaries for batch {i + 1}/{len(batch)}")

    save_summaries(summaries=summaries, output_dir=config.OUTPUT_DIR, suffix="e.json")
    create_metadata_file(config, filepath=config.METADATA_PATH)
