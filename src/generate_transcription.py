from dataset_operations import get_data, create_metadata_file
from llms.llm_factory import get_llm_client
from utils import convert_response_to_json
from logger import setup_logger
import config

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from typing import Dict, Any, Tuple, Optional

logger = setup_logger(__name__)

FILE_PATTERN = "*e.json"
# Tune this if you hit rate limits or want more/less parallelism
MAX_WORKERS = 10

def safe_get_summary_text(item: Dict[str, Any]) -> str:
    """
    Extract a plain summary string from the item if present.
    Accepts both dict-based summaries (with 'text') and string summaries.
    """
    summary = item.get("data", {}).get("summary", "")
    if isinstance(summary, dict):
        return summary.get("text", "") or ""
    if isinstance(summary, str):
        return summary
    return ""

def build_prompt(summary_text: str) -> str:
    return f"Generate a transcription for the following text:{summary_text}"

def process_one(item: Dict[str, Any]) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single file.
    Returns (file_path, success, error_message_or_none).
    """
    file_path = item.get("file_path", "<unknown>")
    data = item.get("data", {})

    # Skip if transcription already exists
    if "transcription" in data:
        logger.info(f"Transcription already exists. Skipping file: {file_path}")
        return file_path, True, None

    logger.info(f"Creating transcription for file: {file_path}")

    try:
        # Per-thread client to avoid shared-state/thread-safety issues
        client = get_llm_client(
            client_type=config.CLIENT_TYPE,
            model=config.TRANSCRIPTION_GENERATOR_LLM_MODEL,
            timeout=600,
        )

        summary_text = safe_get_summary_text(item)

        reply = client.conv(
            user_message=build_prompt(summary_text),
            system_message=config.TRANSCRIPTION_GENERATOR_SYSTEM_PROMPT,
            temperature=config.TRANSCRIPTION_GENERATOR_TEMPERATURE,
            max_tokens=config.TRANSCRIPTION_GENERATOR_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        json_response = convert_response_to_json(reply)
        if not json_response:
            msg = "Failed to decode JSON from model response"
            logger.error(f"{msg}. Skipping file: {file_path}")
            return file_path, False, msg

        # Extract participants from response (order preserved by first appearance)
        participants = []
        for entry in json_response.get("transcription", []):
            sp = entry.get("speaker")
            if sp and sp not in participants:
                participants.append(sp)

        final_doc = {
            "call_id": data.get("call_id"),
            "participants": participants,
            "transcription": json_response.get("transcription", []),
            "summary": data.get("summary", {}),
        }

        # Write back to the same file (each file is unique => no lock needed)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(final_doc, f, indent=2, ensure_ascii=False)

        logger.info(f"Transcription generated and saved for file: {file_path}")
        return file_path, True, None

    except Exception as e:
        msg = f"Exception: {e}"
        logger.error(f"{msg} | File: {file_path}")
        return file_path, False, msg

if __name__ == "__main__":
    data = get_data(data_dir=config.OUTPUT_DIR, file_pattern=FILE_PATTERN)

    if not data:
        logger.warning("No matching files found")
        create_metadata_file(config, filepath=config.METADATA_PATH)
        raise SystemExit(0)

    logger.info(f"Found {len(data)} files to evaluate (pattern: {FILE_PATTERN})")
    workers = min(MAX_WORKERS, max(1, len(data)))
    logger.info(f"Running up to {workers} threads in parallel")

    successes = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(process_one, item): item for item in data}
        for fut in as_completed(future_map):
            _, ok, _ = fut.result()
            if ok:
                successes += 1
            else:
                failures += 1

    logger.info(f"Done. Success: {successes}, Failures: {failures}, Total: {len(data)}")
    create_metadata_file(config, filepath=config.METADATA_PATH)
