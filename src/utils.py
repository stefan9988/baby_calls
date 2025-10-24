import json
import re


def convert_response_to_json(response):
    """
    Converts an LLM response into a JSON-compatible Python object.

    Handles:
    - Standard JSON strings
    - Markdown-formatted JSON (```json ... ```)
    - JSON lists or dicts
    - Already-parsed Python dicts/lists
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
        print("❌ Error decoding JSON response")
        print("Response was:\n", response)
        return None
