from dataset_operations import create_metadata_file, save_summaries
from utils import convert_response_to_json
from logger import setup_logger
import config

import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = setup_logger(__name__)

BATCH_SIZE = 4
NUMBER_OF_SUMMARIES_PER_KEYWORD = 2
MAX_WORKERS = 4

load_dotenv()

model = ChatOpenAI(
    model_name=config.SUMMARY_GENERATOR_LLM_MODEL,
    temperature=config.SUMMARY_GENERATOR_TEMPERATURE,
    max_tokens=config.SUMMARY_GENERATOR_MAX_TOKENS,
    api_key=os.getenv("OPENAI_API_KEY")
)


if __name__ == "__main__":
    if os.path.exists(config.KEYWORDS_PATH):
        with open(config.KEYWORDS_PATH, "r", encoding="utf-8") as f:
            keywords = json.load(f)
        all_keywords = keywords.get("keywords", [])
        logger.info(f"Loaded {len(all_keywords)} keywords")
    else:
        logger.error(f"File not found: {config.KEYWORDS_PATH}")
        exit(1)

    batches = [
        all_keywords[i : i + BATCH_SIZE]
        for i in range(0, len(all_keywords), BATCH_SIZE)
    ]
    total_batches = len(batches)
    if total_batches == 0:
        logger.warning("No keywords to process.")
        exit(0)

    logger.info(f"Created {total_batches} batches (batch size = {BATCH_SIZE})")

    # Limit workers to number of batches to avoid spinning idle threads
    workers = min(MAX_WORKERS, total_batches)
    logger.info(f"Running up to {workers} threads in parallel")

    conversations = []
    for keywords in batches:
        batch_msg = f"""Generate {NUMBER_OF_SUMMARIES_PER_KEYWORD} different summaries per keyword.
                Change context for each summary while keeping it realistic.
                Here are the keywords:\n""" + json.dumps(keywords, indent=4)
        
        conversation = [
            SystemMessage(content=config.SUMMARY_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=batch_msg),
        ]
        conversations.append(conversation)

    model_with_structure = model.with_structured_output(method="json_mode")
    response_summaries = model_with_structure.batch(conversations, config={"max_concurrency": workers})

    summaries = []
    for summary in response_summaries:
        json_response = convert_response_to_json(summary)
        if not json_response:
            logger.error("Failed to generate summaries for this batch. Skipping.")
            continue

        batch_summaries = json_response.get("summaries", [])
        summaries.extend(batch_summaries)        
    

    save_summaries(summaries=summaries, output_dir=config.OUTPUT_DIR, suffix="e.json")
    create_metadata_file(config, filepath=config.METADATA_PATH)
    logger.info(f"Successfully generated {len(summaries)} summaries")    