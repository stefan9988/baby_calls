import os
import json
import time
from pathlib import Path

def get_data(data_dir, file_pattern):
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
                all_data.append({
                    "file_path": str(file_path),
                    "data": data
                })
        except json.JSONDecodeError as e:
            print(f"❌ Error decoding {file_path}: {e}")

    print(f"✅ Loaded {len(all_data)} files successfully.")
    return all_data


def save_summaries(summaries, output_dir, suffix="e.json"):
    os.makedirs(output_dir, exist_ok=True)

    # Get next available number based on existing files
    existing_files = [
        f for f in os.listdir(output_dir)
        if f.endswith(suffix) and f.split(suffix)[0].isdigit()
    ]
    existing_numbers = [int(f.split(suffix)[0]) for f in existing_files]
    start_index = max(existing_numbers, default=0) + 1

    for i, summary in enumerate(summaries, start=start_index):
        # Create unique call_id (with timestamp)
        call_id = f"{i}-record-{int((time.time() * 1000) + i)}_ms"
        # Ensure call_id is the first key in the dict
        summary = {"call_id": call_id, **{k: v for k, v in summary.items() if k != "call_id"}}

        # Create filename like 1e.json, 2e.json, ...
        file_name = f"{i}{suffix}"
        file_path = os.path.join(output_dir, file_name)

        # Save file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {file_name}")

    print(f"\nTotal {len(summaries)} summaries saved at {output_dir}.")

