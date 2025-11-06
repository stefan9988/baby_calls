from dataset_operations import get_data, create_metadata_file
from llms.llm_factory import get_llm_client
from utils import convert_response_to_json
from logger import setup_logger
import config

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import random
import json
import os

logger = setup_logger(__name__)


DATA_DIR = "UNS dataset/json_english_v2"
FILE_PATTERN = "*e.json"
NUMBER_OF_SAMPLES = 5
RANDOM_SEED = 42

# random.seed(RANDOM_SEED)

client = get_llm_client(
    client_type=config.CLIENT_TYPE,
    model=config.KEYWORD_GENERATOR_LLM_MODEL,
    timeout=600,
)
model = ChatOpenAI(
    model_name=config.KEYWORD_GENERATOR_LLM_MODEL,
    temperature=config.KEYWORD_GENERATOR_TEMPERATURE,
    max_tokens=config.KEYWORD_GENERATOR_MAX_TOKENS,
)


def save_keywords():
    """
    Save generated keywords to a JSON file.

    Uses the global json_response variable containing keywords generated
    by the LLM and saves them to the path specified in config.KEYWORDS_PATH.

    Returns:
        None

    Side effects:
        - Creates parent directories if they don't exist
        - Writes keywords to config.KEYWORDS_PATH
        - Logs success message

    Raises:
        NameError: If json_response is not defined in the global scope
    """
    os.makedirs(os.path.dirname(config.KEYWORDS_PATH), exist_ok=True)
    with open(config.KEYWORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(json_response, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {NUMBER_OF_SAMPLES} keyword phrases to {config.KEYWORDS_PATH}")


if __name__ == "__main__":
    example_data = get_data(DATA_DIR, FILE_PATTERN)
    sampled_data = random.sample(
        example_data, min(NUMBER_OF_SAMPLES, len(example_data))
    )
    keyword_examples = [
        sample["data"]["summary"]["key_words"][0] for sample in sampled_data
    ]

    conversation = [
        SystemMessage(content=config.KEYWORD_GENERATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=f"""Generate {NUMBER_OF_SAMPLES} keyword phrases based on the following examples:\n
            {json.dumps(keyword_examples, indent=4)}"""
        ),
    ]
    
    model_with_structure = model.with_structured_output(method="json_mode")
    reply = model.invoke(conversation)

    json_response = convert_response_to_json(reply.content)
    if not json_response:
        logger.error("Failed to generate keywords")
        exit(1)

    save_keywords()
    create_metadata_file(config, filepath=config.METADATA_PATH)
