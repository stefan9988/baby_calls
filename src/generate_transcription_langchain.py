from dataset_operations import get_data, create_metadata_file
from utils import convert_response_to_json
from logger import setup_logger
import config

import json
import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = setup_logger(__name__)

FILE_PATTERN = "*e.json"
# Tune this if you hit rate limits or want more/less parallelism
MAX_WORKERS = 10

load_dotenv()

# Initialize LangChain model
model = ChatOpenAI(
    model_name=config.TRANSCRIPTION_GENERATOR_LLM_MODEL,
    temperature=config.TRANSCRIPTION_GENERATOR_TEMPERATURE,
    max_tokens=config.TRANSCRIPTION_GENERATOR_MAX_TOKENS,
    api_key=os.getenv("OPENAI_API_KEY")
)

def safe_get_summary_text(item: Dict[str, Any]) -> str:
    """
    Extract summary text from a data item with fallback handling.

    Supports multiple summary formats:
    - Dictionary with 'text' field (returns the text value)
    - Direct string value
    - Missing or malformed data (returns empty string)

    Args:
        item: Dictionary containing 'data' -> 'summary' structure

    Returns:
        str: Extracted summary text, or empty string if unavailable
    """
    summary = item.get("data", {}).get("summary", "")
    if isinstance(summary, dict):
        return summary.get("text", "") or ""
    if isinstance(summary, str):
        return summary
    return ""

def build_prompt(summary_text: str) -> str:
    """
    Build a user prompt for generating transcriptions from summary text.

    Args:
        summary_text: Summary text describing the medical case

    Returns:
        str: Formatted prompt string for the LLM
    """
    return f"Generate a transcription for the following text:{summary_text}"

if __name__ == "__main__":
    data = get_data(data_dir=config.OUTPUT_DIR, file_pattern=FILE_PATTERN)

    if not data:
        logger.warning("No matching files found")
        create_metadata_file(config, filepath=config.METADATA_PATH)
        raise SystemExit(0)

    logger.info(f"Found {len(data)} files to evaluate (pattern: {FILE_PATTERN})")

    # Filter out items that already have transcriptions
    items_to_process = []
    for item in data:
        if "transcription" in item.get("data", {}):
            logger.info(f"Transcription already exists. Skipping file: {item.get('file_path')}")
        else:
            items_to_process.append(item)

    if not items_to_process:
        logger.info("All files already have transcriptions. Nothing to process.")
        create_metadata_file(config, filepath=config.METADATA_PATH)
        raise SystemExit(0)

    logger.info(f"Processing {len(items_to_process)} files")
    workers = min(MAX_WORKERS, max(1, len(items_to_process)))
    logger.info(f"Running with max concurrency of {workers}")

    # Build all conversations upfront
    conversations = []
    for item in items_to_process:
        summary_text = safe_get_summary_text(item)
        conversation = [
            SystemMessage(content=config.TRANSCRIPTION_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=build_prompt(summary_text)),
        ]
        conversations.append(conversation)

    # Process all conversations in batch
    model_with_structure = model.with_structured_output(method="json_mode")
    responses = model_with_structure.batch(conversations, config={"max_concurrency": workers})

    successes = 0
    failures = 0

    # Process responses and save to files
    for item, response in zip(items_to_process, responses):
        file_path = item.get("file_path", "<unknown>")
        data_dict = item.get("data", {})

        try:
            json_response = convert_response_to_json(response)
            if not json_response:
                logger.error(f"Failed to decode JSON from model response. Skipping file: {file_path}")
                failures += 1
                continue

            # Extract participants from response (order preserved by first appearance)
            participants = []
            for entry in json_response.get("transcription", []):
                sp = entry.get("speaker")
                if sp and sp not in participants:
                    participants.append(sp)

            final_doc = {
                "call_id": data_dict.get("call_id"),
                "participants": participants,
                "transcription": json_response.get("transcription", []),
                "summary": data_dict.get("summary", {}),
            }

            # Write back to the same file (each file is unique => no lock needed)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(final_doc, f, indent=2, ensure_ascii=False)

            logger.info(f"Transcription generated and saved for file: {file_path}")
            successes += 1

        except Exception as e:
            logger.error(f"Exception: {e} | File: {file_path}")
            failures += 1

    logger.info(f"Done. Success: {successes}, Failures: {failures}, Total: {len(items_to_process)}")
    create_metadata_file(config, filepath=config.METADATA_PATH)
