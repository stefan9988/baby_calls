import sdialog
from sdialog.agents import Agent
from sdialog.orchestrators import LengthOrchestrator
import os
from dotenv import load_dotenv
import config_sdialog
from dataset_operations import get_data

load_dotenv()

sdialog.config.llm(config_sdialog.LLM, api_key=os.getenv("OPENAI_API_KEY"))

FILE_PATTERN = "*e.json"
data = get_data(data_dir=config_sdialog.INPUT_DIR, file_pattern=FILE_PATTERN)

nurse = config_sdialog.nurse
caller = config_sdialog.caller
ctx = config_sdialog.ctx

nurse_agent = Agent(
    persona=nurse,
    first_utterance="Hello, this is the triage nurse speaking. How can I assist you today?",
)
len_orchestrator = LengthOrchestrator(min=12, max=20)
nurse_agent = nurse_agent | len_orchestrator

for item in data:
    filename = os.path.basename(item["file_path"])

    circumstances = "\n - ".join(item["data"]["summary"]["text"])
    caller.circumstances = circumstances
    caller_agent = Agent(persona=caller)

    dialog = nurse_agent.dialog_with(caller_agent, context=ctx, max_turns=25)
    dialog.print()
    dialog.to_file(config_sdialog.OUTPUT_DIR + "/" + filename)
    break
