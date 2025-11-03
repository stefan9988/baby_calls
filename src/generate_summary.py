from dataset_operations import create_metadata_file, save_summaries
from llms.llm_factory import get_llm_client
from utils import convert_response_to_json
from logger import setup_logger
import config

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

logger = setup_logger(__name__)

BATCH_SIZE = 10
NUMBER_OF_SUMMARIES_PER_KEYWORD = 2
MAX_WORKERS = 5

def build_prompt(keywords_chunk):
    return (
        f"Generate {NUMBER_OF_SUMMARIES_PER_KEYWORD} different summaries per keyword.\n"
        f"Change context for each summary while keeping it realistic.\n"
        f"Here are the keywords:\n"
        + json.dumps(keywords_chunk, indent=4)
    )

def process_batch(batch_idx, keywords_chunk):
    """
    Run a single batch in its own thread.
    Returns (batch_idx, summaries_list) or (batch_idx, []) on failure.
    """
    try:
        client = get_llm_client(
            client_type=config.CLIENT_TYPE,
            model=config.SUMMARY_GENERATOR_LLM_MODEL,
            timeout=600,
        )

        reply = client.conv(
            user_message=build_prompt(keywords_chunk),
            system_message=config.SUMMARY_GENERATOR_SYSTEM_PROMPT,
            temperature=config.SUMMARY_GENERATOR_TEMPERATURE,
            max_tokens=config.SUMMARY_GENERATOR_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        json_response = convert_response_to_json(reply)
        if not json_response:
            logger.error(f"Failed to generate summaries for batch {batch_idx + 1}")
            return batch_idx, []

        batch_summaries = json_response.get("summaries", [])
        logger.info(f"Generated summaries for batch {batch_idx + 1}")
        return batch_idx, batch_summaries

    except Exception as e:
        logger.error(f"Exception in batch {batch_idx + 1}: {e}")
        return batch_idx, []

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
        logger.warning("No keywords to process")
        exit(0)

    logger.info(f"Created {total_batches} batches (batch size = {BATCH_SIZE})")

    # Limit workers to number of batches to avoid spinning idle threads
    workers = min(MAX_WORKERS, total_batches)
    logger.info(f"Running up to {workers} threads in parallel")

    summaries = []
    # To preserve deterministic ordering (optional), collect results in a dict and then flatten by index
    results_by_idx = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_batch, idx, chunk): idx
            for idx, chunk in enumerate(batches)
        }
        for future in as_completed(futures):
            idx = futures[future]
            batch_idx, batch_summaries = future.result()
            results_by_idx[batch_idx] = batch_summaries
            logger.info(f"Batch {batch_idx + 1}/{total_batches} completed")

    # Merge in original batch order
    for idx in range(total_batches):
        summaries.extend(results_by_idx.get(idx, []))

    save_summaries(summaries=summaries, output_dir=config.OUTPUT_DIR, suffix="e.json")
    create_metadata_file(config, filepath=config.METADATA_PATH)    
