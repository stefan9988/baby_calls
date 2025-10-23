import json


def convert_response_to_json(response):
    try:
        json_response = json.loads(response)
        return json_response
    except json.JSONDecodeError:
        print("❌ Error decoding JSON response")
        return None
