# README

A pipeline for generating **keywords**, **summaries** and **conversations** for augmenting the UNS dataset.

Supports multiple implementation approaches:
- **Standard prompting**: Direct OpenAI API calls with system prompts
- **LangChain**: Framework-based implementation with better batch processing and structured outputs
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
   * LangChain library is included for improved batch processing and structured outputs

---

## Implementation Approaches

The project provides three different implementations for generating data:

### Standard Implementation
Direct OpenAI API calls. Simple and straightforward.
- Scripts: `generate_keywords.py`, `generate_summary.py`, `generate_transcription.py`
- Best for: Basic use cases, learning the pipeline

### LangChain Implementation (Recommended)
Uses the LangChain framework for better structure and reliability.
- Scripts: `generate_keywords_langchain.py`, `generate_summary_langchain.py`, `generate_transcription_langchain.py`
- Benefits:
  - Improved batch processing with `model.batch()`
  - Built-in structured output handling with `with_structured_output()`
  - Better error handling and retry logic
  - Cleaner code organization
- Best for: Production use, processing large datasets

### sdialog Implementation
Agent-based dialogue generation for natural conversations.
- Script: `sdialog_generate_transcription.py`
- Best for: Most natural and varied conversations

---

## Usage

### 1) Generate keywords

Run the keywords script first. It will create keyword entries based on your provided examples and save them to the folder you specify.

**Standard version:**
```bash
python src/generate_keywords.py
```

**LangChain version (recommended):**
```bash
python src/generate_keywords_langchain.py
```

**What it does**

* Reads from examples.
* Generates keyword phrases.
* Saves them into the specified file (e.g., `UNS dataset/json_english_aug/keywords.json`).

---

### 2) Generate summaries

After your augmented keywords are ready (stored in UNS dataset/json_english_aug/keywords.json), run:

**Standard version:**
```bash
python src/generate_summary.py
```

**LangChain version (recommended):**
```bash
python src/generate_summary_langchain.py
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

There are three methods to generate transcriptions from summaries:

#### Option A: LangChain Implementation (Recommended)

Uses LangChain's batch processing for efficient, structured transcription generation.

```bash
python src/generate_transcription_langchain.py
```

**What it does**
* Loads items via `get_data(data_dir=OUTPUT_DIR, file_pattern=FILE_PATTERN)`
* Skips any file that already contains a "transcription" field (idempotent)
* Uses LangChain's `model.batch()` for efficient parallel processing
* Automatic structured output parsing with `with_structured_output(method="json_mode")`
* Better error handling and retry logic compared to standard implementation

**Benefits over standard:**
- Faster batch processing with configurable concurrency
- Built-in JSON validation and structured output
- Automatic retry on failures
- Cleaner code with LangChain abstractions

---

#### Option B: Standard LLM Prompting

Turns each file's summary text into a structured transcription using direct LLM calls with system prompts.

```bash
python src/generate_transcription.py
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

#### Option C: sdialog Agent Framework (For most natural dialogues)

Uses the [sdialog](https://github.com/idiap/sdialog) library to generate conversations through agent-based simulation with personas.

```bash
python src/sdialog_generate_transcription.py
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

**Comparison of all three options:**

| Feature | LangChain (A) | Standard (B) | sdialog (C) |
|---------|---------------|--------------|-------------|
| Speed | ⚡⚡⚡ Fastest (batch) | ⚡⚡ Fast (multi-thread) | ⚡ Slower (sequential) |
| Natural dialogues | ✅ Good | ✅ Good | ✅✅ Most natural |
| Batch processing | ✅✅ Excellent | ✅ Good | ❌ No |
| Error handling | ✅✅ Built-in retries | ✅ Basic | ✅ Basic |
| Code complexity | 🔧 Low (LangChain abstractions) | 🔧 Medium | 🔧🔧 Higher (agents) |
| Dependencies | LangChain + OpenAI | OpenAI only | sdialog + OpenAI |
| **Recommended for** | Production, large datasets | Learning, simple use cases | Maximum dialogue quality |

---

### 4) One-shot pipeline via Docker helper script

You can build the Docker image and run all three steps (keywords → summaries → transcriptions) in one go using run_docker.sh.

```bash
chmod +x run_docker.sh
./run_docker.sh
```

**What it does**

The script provides three pipeline options:
1. **Standard pipeline**: Uses standard implementation scripts
2. **LangChain pipeline** (recommended): Uses LangChain implementation scripts
3. **SDialog pipeline**: Uses sdialog for transcription generation only

**Features:**
* Builds the Docker image named baby-calls
* Runs the selected pipeline scripts in sequence
* Mounts your local dataset folder into the container:
  - Host: `$(pwd)/UNS dataset`
  - Container: `/app/src/UNS dataset`
* Loads environment variables from `.env`
* Uses host networking (`--network host`)
* If any step fails, the script stops and shows a red ✗ message; otherwise you'll see a final green "All tasks completed successfully!" ✓

**Example:**
```
=== Baby Calls Docker Runner ===

Select which pipeline to run:
  1) Standard pipeline (keywords → summary → transcription)
  2) LangChain pipeline (keywords → summary → transcription) - Recommended
  3) SDialog transcription

Enter your choice (1, 2, or 3): 2
```