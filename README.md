# README

A pipeline for generating **keywords**, **summaries** and **conversations** for augmenting the UNS dataset.

Supports two methods for generating conversations:
- **Standard prompting**: Direct LLM calls with system prompts
- **sdialog**: Agent-based framework for more natural multi-turn dialogues

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

   * The sdialog library is included for agent-based dialogue generation (optional)

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

There are two methods to generate transcriptions from summaries:

#### Option A: Standard LLM Prompting (Recommended for batch processing)

Turns each file's summary text into a structured transcription using direct LLM calls with system prompts.

```bash
python generate_transcription.py
```

**What it does**

* Loads items via `get_data(data_dir=OUTPUT_DIR, file_pattern=FILE_PATTERN)`
* Skips any file that already contains a "transcription" field (idempotent)
* Extracts the summary text from `item["data"]["summary"]["text"]`
* Calls the LLM with a detailed system prompt to create transcripts
* Supports multithreading for parallel processing (configurable via `MAX_WORKERS`)
* Builds the final document and writes it back to the same file path

**Configuration:** Edit `config.py` to customize:
- `TRANSCRIPTION_GENERATOR_LLM_MODEL`: Model to use
- `TRANSCRIPTION_GENERATOR_TEMPERATURE`: Sampling temperature
- `TRANSCRIPTION_GENERATOR_SYSTEM_PROMPT`: Detailed instructions for the LLM

---

#### Option B: sdialog Agent Framework (Recommended for natural dialogues)

Uses the [sdialog](https://github.com/idiap/sdialog) library to generate conversations through agent-based simulation with personas.

```bash
python sdialog_generate_transcription.py
```

**What it does**

* Creates two agents with distinct personas:
  - **Nurse Agent**: Triage nurse with calm, professional personality
  - **Caller Agent**: Worried parent calling about their sick baby
* Uses `LengthOrchestrator` to control conversation length (min/max turns)
* Generates natural multi-turn dialogues where agents interact dynamically
* Each agent has specific rules and behaviors defined in `config_sdialog.py`
* Outputs more natural and varied conversations compared to single-shot prompting

**Configuration:** Edit `config_sdialog.py` to customize:
- `MAX_TURNS`: Maximum conversation turns
- `LENGTH_ORCHESTRATOR_MIN/MAX`: Conversation length bounds
- `nurse` and `caller` Persona objects with personality, rules, and language
- `context`: Dialogue context including topics and behavioral notes

**Key differences from Option A:**
- ✅ More natural turn-taking and dialogue flow
- ✅ Agents respond dynamically to each other
- ✅ Better handling of follow-up questions
- ❌ Slower (sequential processing, no multithreading)
- ❌ Requires sdialog library dependency

---

### 4) One-shot pipeline via Docker helper script

You can build the Docker image and run all three steps (keywords → summaries → transcriptions) in one go using run_docker.sh.

```bash
chmod +x run_docker.sh
./run_docker.sh
```

**What it does**

* Builds the Docker image named baby-calls

* Runs generate_keywords.py in the container

* Mounts your local dataset folder into the container:

   Host: $(pwd)/UNS dataset

   Container: /app/src/UNS dataset

   Loads environment variables from .env

   Uses host networking (--network host)

* Runs generate_summary.py (same mount, env, and network settings)

* Runs generate_transcription.py (same mount, env, and network settings)

* If any step fails, the script stops and shows a red ✗ message; otherwise you’ll see a final green “All tasks completed successfully!” ✓