# README

A pipeline for generating **keywords**, **summaries** and **conversations**  for augmenting the UNS dataset.

---

## Prerequisites

1. **UNS dataset is present**

   * Make sure all required JSON files are already in your local `UNS dataset` folder.

2. **API key in your environment**

   * Create a `.env` file in the project root and add your model provider key(s), for example:

     ```
     OPENAI_API_KEY=your_key_here
     ```

3. **Python environment**

   * Python 3.12 recommended.
   * Install dependencies:

     ```bash
     poetry install
     ```

---

## Usage

### 1) Generate keywords

Run the keywords script first. It will create keyword entries based on your provided examples and save them to the folder you specify.

```bash
python generate_keywords.py 
```

**What it does**

* Reads from examples.
* Generates keyword phrases.
* Saves them into the specified file (e.g., `UNS dataset/json_english_aug/keywords.json`).

---

### 2) Generate summaries

After your augmented keywords are ready (stored in UNS dataset/json_english_aug/keywords.json), run:

```bash
python generate_summary.py
```

**What it does**

* Loads keywords from AUGMENTED_KEYWORDS_PATH.

* Batches them (BATCH_SIZE, default: 10) and sends each batch to your LLM.

* Asks the model to create NUMBER_OF_SUMMARIES_PER_KEYWORD summaries per keyword (default: 2), each with a slightly different but realistic context.

* Collects all returned summaries and writes them into numbered JSON files in OUTPUT_DIR (default: UNS dataset/json_english_aug), using save_summaries().

* Before saving, the script scans the OUTPUT_DIR and checks all existing files.
If, for example, files up to 5e.json already exist, the next run will start naming from 6e.json.
This ensures that file numbering continues automatically and no existing files are overwritten.

---
### 3) Generate transcriptions (conversations)

Turns each file’s summary text into a structured transcription and appends it back into the same file.

```bash
python generate_transcription.py
```

**What it does**

* Loads items via get_data(data_dir=OUTPUT_DIR, file_pattern=FILE_PATTERN).

* Skips any file that already contains a "transcription" field (idempotent).

* Extracts the summary text from item["data"]["summary"]["text"] (falls back to "" if missing).

* Calls the LLM that creates transcripts based on the summary

* Builds the final document and writes it back to the same file path

