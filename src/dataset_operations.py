import os
import json
import time
from pathlib import Path
from logger import setup_logger

logger = setup_logger(__name__)


def get_data(data_dir, file_pattern):
    """
    Load JSON files matching a pattern from a directory.

    Args:
        data_dir (str): Directory path containing JSON files
        file_pattern (str): Glob pattern to match files (e.g., "*e.json")

    Returns:
        list[dict]: List of dictionaries, each containing:
            - 'file_path': Absolute path to the JSON file
            - 'data': Parsed JSON content

    Raises:
        FileNotFoundError: If data_dir does not exist

    Examples:
        >>> data = get_data("UNS dataset/json", "*e.json")
        >>> print(len(data))
        150
    """
    folder_path = Path(data_dir)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    json_files = sorted(folder_path.glob(file_pattern))
    all_data = []

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Return both the data and the file path
                all_data.append({"file_path": str(file_path), "data": data})
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding {file_path}: {e}")

    logger.info(f"Loaded {len(all_data)} files successfully")
    return all_data


def save_summaries(summaries, output_dir, suffix="e.json"):
    """
    Save summary dictionaries to numbered JSON files with auto-incrementing names.

    Scans existing files in output_dir to determine the next available number,
    ensuring no files are overwritten. Each summary is enriched with a unique
    call_id before saving.

    Args:
        summaries (list[dict]): List of summary dictionaries to save
        output_dir (str): Target directory for output files
        suffix (str, optional): File suffix/extension. Defaults to "e.json"

    Returns:
        None

    Side effects:
        - Creates output_dir if it doesn't exist
        - Writes numbered JSON files (e.g., 1e.json, 2e.json, ...)
        - Logs progress for each saved file

    Examples:
        >>> summaries = [{"summary": {"text": ["Patient info..."]}}]
        >>> save_summaries(summaries, "output", "e.json")
        # Creates output/1e.json, output/2e.json, etc.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get next available number based on existing files
    existing_files = [
        f
        for f in os.listdir(output_dir)
        if f.endswith(suffix) and f.split(suffix)[0].isdigit()
    ]
    existing_numbers = [int(f.split(suffix)[0]) for f in existing_files]
    start_index = max(existing_numbers, default=0) + 1

    for i, summary in enumerate(summaries, start=start_index):
        # Create unique call_id (with timestamp)
        call_id = f"{i}-record-{int((time.time() * 1000) + i)}_ms"
        # Ensure call_id is the first key in the dict
        summary = {
            "call_id": call_id,
            **{k: v for k, v in summary.items() if k != "call_id"},
        }

        # Create filename like 1e.json, 2e.json, ...
        file_name = f"{i}{suffix}"
        file_path = os.path.join(output_dir, file_name)

        # Save file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {file_name}")

    logger.info(f"Total {len(summaries)} summaries saved at {output_dir}")


def create_metadata_file(config_module, filepath):
    """
    Create a metadata JSON file containing all configuration variables.

    Extracts all non-private, non-callable attributes from the provided
    config module and saves them as a JSON file for reproducibility and
    audit purposes.

    Args:
        config_module: Python module object containing configuration variables
        filepath (str): Target path for the metadata JSON file

    Returns:
        None

    Side effects:
        - Creates parent directories if they don't exist
        - Writes JSON file with all config variables
        - Logs success message

    Examples:
        >>> import config
        >>> create_metadata_file(config, "output/metadata.json")
        # Creates output/metadata.json with all config variables
    """
    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Collect all non-dunder, non-callable attributes from config.py
    config_dict = {
        key: getattr(config_module, key)
        for key in dir(config_module)
        if not key.startswith("__") and not callable(getattr(config_module, key))
    }

    # Write metadata to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Metadata file created at {filepath}")
