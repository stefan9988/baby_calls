import json
import re
from logger import setup_logger

logger = setup_logger(__name__)


def convert_response_to_json(response):
    """
    Convert an LLM response into a JSON-compatible Python object.

    Handles multiple response formats including:
    - Standard JSON strings
    - Markdown-formatted JSON with code fences (```json ... ```)
    - Pre-parsed Python dicts or lists
    - JSON with optional language hints

    Args:
        response: LLM response as string, dict, or list

    Returns:
        dict or list: Parsed JSON object, or None if parsing fails

    Examples:
        >>> convert_response_to_json('{"key": "value"}')
        {'key': 'value'}
        >>> convert_response_to_json('```json\\n{"key": "value"}\\n```')
        {'key': 'value'}
        >>> convert_response_to_json({'key': 'value'})
        {'key': 'value'}
    """

    # Case 1: response is already a dict or list
    if isinstance(response, (dict, list)):
        return response

    # Case 2: clean Markdown-style code fences if present
    if isinstance(response, str):
        # Remove triple backticks and optional language hints (e.g. ```json)
        response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)

    # Case 3: try parsing JSON
    try:
        json_response = json.loads(response)
        return json_response
    except json.JSONDecodeError:
        logger.error("Error decoding JSON response")
        logger.debug(f"Response was: {response}")
        return None
