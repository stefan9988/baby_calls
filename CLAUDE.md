# Claude Code Documentation

## Project Overview
A pipeline for generating medical conversation data from the UNS dataset. Creates keywords, summaries, and realistic transcriptions of parent-nurse phone consultations.

## Project Structure

### Main Scripts
The project provides three implementation approaches:

1. **Standard Implementation** (Direct OpenAI API calls):
   - `generate_keywords.py` - Generates medical keywords from examples
   - `generate_summary.py` - Creates case summaries from keywords
   - `generate_transcription.py` - Generates conversation transcripts from summaries

2. **LangChain Implementation** (Uses LangChain framework):
   - `generate_keywords_langchain.py` - LangChain version of keywords generation
   - `generate_summary_langchain.py` - LangChain version of summary generation
   - `generate_transcription_langchain.py` - LangChain version with batch processing

3. **Agent-Based Implementation** (sdialog framework):
   - `sdialog_generate_transcription.py` - Multi-agent dialogue generation

### Supporting Files
- `config.py` - Configuration for all LLM models and prompts
- `config_sdialog.py` - Agent personas and sdialog-specific settings
- `dataset_operations.py` - File I/O operations for UNS dataset
- `utils.py` - JSON parsing and utility functions
- `logger.py` - Logging configuration

## Usage

### Standard vs LangChain
- **Standard**: Simple, direct API calls. Good for basic use cases.
- **LangChain**: Better batch processing, structured outputs, built-in retry logic. Recommended for production.

### Running Scripts
```bash
# Standard implementation
python src/generate_keywords.py
python src/generate_summary.py
python src/generate_transcription.py

# LangChain implementation (recommended)
python src/generate_keywords_langchain.py
python src/generate_summary_langchain.py
python src/generate_transcription_langchain.py

# Agent-based (for most natural dialogues)
python src/sdialog_generate_transcription.py
```

### Configuration
Edit `config.py` to modify:
- Output directory: `OUTPUT_DIR`
- LLM models: `*_LLM_MODEL` variables
- System prompts: `*_SYSTEM_PROMPT` variables
- Temperature and max tokens for each component

## Testing
Run the full test suite using pytest: `PYTHONPATH=src python -m pytest tests/ -v`

## Dependencies
- `openai` - OpenAI API client
- `langchain` - LangChain framework for LLM applications
- `sdialog` - Agent-based dialogue generation
- `python-dotenv` - Environment variable management
- `pytest` - Testing framework

