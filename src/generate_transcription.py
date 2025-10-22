from dataset_operations import get_data, create_metadata_file
from llms.llm_factory import get_llm_client
import config

import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

FILE_PATTERN = "*e.json"

client = get_llm_client(client_type=config.CLIENT_TYPE, api_key=OPENAI_API_KEY, model=config.TRANSCRIPTION_GENERATOR_LLM_MODEL)


def get_participants(transcription_data: dict) -> list:
    """
    Return unique speakers in order of appearance.
    """
    speakers = []
    for entry in transcription_data.get("transcription", []):
        speaker = entry.get("speaker")
        if speaker and speaker not in speakers:
            speakers.append(speaker)
    return speakers


if __name__ == "__main__":
    data = get_data(data_dir=config.OUTPUT_DIR, file_pattern=FILE_PATTERN)

    for item in data:
        # Skip if transcription already exists
        if "transcription" in item["data"]:
            print(f"Transcription already exists. Skipping file: {item['file_path']}")
            continue

        print(f"Creating transcription for file: {item['file_path']}")

        # Pull summary text (fallback to empty string if missing)
        summary_text = ""
        if isinstance(item.get("data", {}).get("summary"), dict):
            summary_text = item["data"]["summary"].get("text", "") or ""

        reply = client.conv(
            user_message=f"""Generate a transcription for the following text:{summary_text}""",
            system_message=config.TRANSCRIPTION_GENERATOR_SYSTEM_PROMPT,
            temperature=config.TRANSCRIPTION_GENERATOR_TEMPERATURE,
            max_tokens=config.TRANSCRIPTION_GENERATOR_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        json_response = json.loads(reply)

        participants = get_participants(json_response)

        call_id = item["data"].get("call_id")
        
        final_doc = {
            "call_id": call_id,
            "participants": participants,
            "transcription": json_response.get("transcription", []),
            "summary": item["data"].get("summary", {}),
        }

        with open(item["file_path"], "w", encoding="utf-8") as f:
            json.dump(final_doc, f, indent=2, ensure_ascii=False)

    create_metadata_file(config, filepath=config.METADATA_PATH)        
