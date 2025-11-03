import pytest
import json
from unittest.mock import patch


class TestConvertResponseToJson:
    """Test suite for convert_response_to_json function."""

    def test_already_dict(self):
        """Test that dict input is returned as-is."""
        from src.utils import convert_response_to_json

        input_dict = {"key": "value", "number": 42}
        result = convert_response_to_json(input_dict)

        assert result == input_dict
        assert result is input_dict  # Should be same object

    def test_already_list(self):
        """Test that list input is returned as-is."""
        from src.utils import convert_response_to_json

        input_list = ["item1", "item2", {"nested": "dict"}]
        result = convert_response_to_json(input_list)

        assert result == input_list
        assert result is input_list  # Should be same object

    def test_plain_json_string_dict(self):
        """Test parsing plain JSON string containing a dict."""
        from src.utils import convert_response_to_json

        json_string = '{"name": "test", "value": 123}'
        result = convert_response_to_json(json_string)

        assert result == {"name": "test", "value": 123}
        assert isinstance(result, dict)

    def test_plain_json_string_list(self):
        """Test parsing plain JSON string containing a list."""
        from src.utils import convert_response_to_json

        json_string = '["item1", "item2", "item3"]'
        result = convert_response_to_json(json_string)

        assert result == ["item1", "item2", "item3"]
        assert isinstance(result, list)

    def test_markdown_wrapped_json_with_language_hint(self):
        """Test parsing JSON wrapped in markdown with 'json' language hint."""
        from src.utils import convert_response_to_json

        markdown_string = '```json\n{"key": "value", "count": 5}\n```'
        result = convert_response_to_json(markdown_string)

        assert result == {"key": "value", "count": 5}

    def test_markdown_wrapped_json_without_language_hint(self):
        """Test parsing JSON wrapped in markdown without language hint."""
        from src.utils import convert_response_to_json

        markdown_string = '```\n{"key": "value", "count": 5}\n```'
        result = convert_response_to_json(markdown_string)

        assert result == {"key": "value", "count": 5}

    def test_markdown_wrapped_json_uppercase_hint(self):
        """Test parsing JSON with uppercase JSON language hint."""
        from src.utils import convert_response_to_json

        markdown_string = '```JSON\n{"test": true}\n```'
        result = convert_response_to_json(markdown_string)

        assert result == {"test": True}

    def test_markdown_wrapped_json_mixed_case(self):
        """Test parsing JSON with mixed case JSON language hint."""
        from src.utils import convert_response_to_json

        markdown_string = '```Json\n{"test": true}\n```'
        result = convert_response_to_json(markdown_string)

        assert result == {"test": True}

    def test_markdown_wrapped_with_extra_whitespace(self):
        """Test parsing markdown-wrapped JSON with extra whitespace."""
        from src.utils import convert_response_to_json

        markdown_string = '```json  \n  {"key": "value"}  \n```  '
        result = convert_response_to_json(markdown_string)

        assert result == {"key": "value"}

    def test_nested_json_structure(self):
        """Test parsing complex nested JSON structure."""
        from src.utils import convert_response_to_json

        json_string = '''{
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"]
                },
                "array": [1, 2, 3]
            }
        }'''
        result = convert_response_to_json(json_string)

        expected = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"]
                },
                "array": [1, 2, 3]
            }
        }
        assert result == expected

    def test_json_with_unicode_characters(self):
        """Test parsing JSON with unicode characters."""
        from src.utils import convert_response_to_json

        json_string = '{"text": "Тест с кириллицей", "emoji": "🚀"}'
        result = convert_response_to_json(json_string)

        assert result == {"text": "Тест с кириллицей", "emoji": "🚀"}

    def test_json_with_escaped_characters(self):
        """Test parsing JSON with escaped characters."""
        from src.utils import convert_response_to_json

        json_string = '{"text": "Line1\\nLine2\\tTabbed", "quote": "\\"quoted\\""}'
        result = convert_response_to_json(json_string)

        assert result["text"] == "Line1\nLine2\tTabbed"
        assert result["quote"] == '"quoted"'

    def test_json_with_null_value(self):
        """Test parsing JSON with null values."""
        from src.utils import convert_response_to_json

        json_string = '{"key": null, "other": "value"}'
        result = convert_response_to_json(json_string)

        assert result == {"key": None, "other": "value"}

    def test_json_with_boolean_values(self):
        """Test parsing JSON with boolean values."""
        from src.utils import convert_response_to_json

        json_string = '{"isTrue": true, "isFalse": false}'
        result = convert_response_to_json(json_string)

        assert result == {"isTrue": True, "isFalse": False}

    def test_json_array_of_objects(self):
        """Test parsing JSON array containing objects."""
        from src.utils import convert_response_to_json

        json_string = '[{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]'
        result = convert_response_to_json(json_string)

        assert result == [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]

    def test_empty_dict_string(self):
        """Test parsing empty dict JSON string."""
        from src.utils import convert_response_to_json

        json_string = '{}'
        result = convert_response_to_json(json_string)

        assert result == {}

    def test_empty_array_string(self):
        """Test parsing empty array JSON string."""
        from src.utils import convert_response_to_json

        json_string = '[]'
        result = convert_response_to_json(json_string)

        assert result == []

    @patch('src.utils.logger')
    def test_invalid_json_string(self, mock_logger):
        """Test that invalid JSON returns None and logs error."""
        from src.utils import convert_response_to_json

        invalid_json = '{"key": invalid value}'
        result = convert_response_to_json(invalid_json)

        assert result is None
        # Verify error was logged
        mock_logger.error.assert_called_once_with("Error decoding JSON response")
        mock_logger.debug.assert_called_once()

    @patch('src.utils.logger')
    def test_empty_string(self, mock_logger):
        """Test that empty string returns None and logs error."""
        from src.utils import convert_response_to_json

        result = convert_response_to_json('')

        assert result is None
        mock_logger.error.assert_called_once()

    @patch('src.utils.logger')
    def test_plain_text_not_json(self, mock_logger):
        """Test that plain text (not JSON) returns None and logs error."""
        from src.utils import convert_response_to_json

        plain_text = 'This is just plain text, not JSON'
        result = convert_response_to_json(plain_text)

        assert result is None
        mock_logger.error.assert_called_once_with("Error decoding JSON response")

    @patch('src.utils.logger')
    def test_markdown_wrapped_invalid_json(self, mock_logger):
        """Test markdown-wrapped invalid JSON returns None."""
        from src.utils import convert_response_to_json

        markdown_string = '```json\n{invalid: json}\n```'
        result = convert_response_to_json(markdown_string)

        assert result is None
        mock_logger.error.assert_called_once()

    def test_json_with_numbers(self):
        """Test parsing JSON with various number types."""
        from src.utils import convert_response_to_json

        json_string = '{"int": 42, "float": 3.14, "negative": -10, "exp": 1.5e3}'
        result = convert_response_to_json(json_string)

        assert result == {"int": 42, "float": 3.14, "negative": -10, "exp": 1500.0}

    def test_markdown_with_multiple_backticks_only_outer_removed(self):
        """Test that only outer backticks are removed, not inner ones."""
        from src.utils import convert_response_to_json

        # This should work - inner backticks in string value are preserved
        markdown_string = '```json\n{"code": "```python\\nprint()\\n```"}\n```'
        result = convert_response_to_json(markdown_string)

        assert result == {"code": "```python\nprint()\n```"}

    def test_json_string_with_leading_trailing_whitespace(self):
        """Test that whitespace before/after JSON is handled."""
        from src.utils import convert_response_to_json

        json_string = '   \n  {"key": "value"}  \n  '
        result = convert_response_to_json(json_string)

        assert result == {"key": "value"}

    def test_markdown_with_backticks_at_start_only(self):
        """Test JSON with opening backticks but no closing."""
        from src.utils import convert_response_to_json

        # Regex should handle this gracefully
        markdown_string = '```json\n{"key": "value"}'
        result = convert_response_to_json(markdown_string)

        assert result == {"key": "value"}

    def test_markdown_with_backticks_at_end_only(self):
        """Test JSON with closing backticks but no opening."""
        from src.utils import convert_response_to_json

        # Regex should handle this gracefully
        markdown_string = '{"key": "value"}\n```'
        result = convert_response_to_json(markdown_string)

        assert result == {"key": "value"}

    def test_very_long_json(self):
        """Test parsing very long JSON string."""
        from src.utils import convert_response_to_json

        # Create a large JSON object
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        json_string = json.dumps(large_dict)
        result = convert_response_to_json(json_string)

        assert result == large_dict
        assert len(result) == 1000

    def test_json_with_special_characters_in_keys(self):
        """Test JSON with special characters in keys."""
        from src.utils import convert_response_to_json

        json_string = '{"key-with-dash": "value", "key.with.dots": "value2", "key_underscore": "value3"}'
        result = convert_response_to_json(json_string)

        assert result == {
            "key-with-dash": "value",
            "key.with.dots": "value2",
            "key_underscore": "value3"
        }

    def test_json_single_number(self):
        """Test JSON that is just a single number."""
        from src.utils import convert_response_to_json

        json_string = '42'
        result = convert_response_to_json(json_string)

        assert result == 42

    def test_json_single_string(self):
        """Test JSON that is just a single quoted string."""
        from src.utils import convert_response_to_json

        json_string = '"hello world"'
        result = convert_response_to_json(json_string)

        assert result == "hello world"

    def test_json_single_boolean(self):
        """Test JSON that is just a single boolean."""
        from src.utils import convert_response_to_json

        assert convert_response_to_json('true') is True
        assert convert_response_to_json('false') is False

    def test_json_null(self):
        """Test JSON that is just null."""
        from src.utils import convert_response_to_json

        json_string = 'null'
        result = convert_response_to_json(json_string)

        assert result is None
