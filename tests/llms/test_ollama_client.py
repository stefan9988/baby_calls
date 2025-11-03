import pytest
from unittest.mock import Mock, patch
import requests


class TestOllamaClient:
    """Test suite for OllamaClient."""

    def test_initialization_with_defaults(self):
        """Test OllamaClient initialization with default values."""
        from src.llms.ollama_client import OllamaClient

        client = OllamaClient(model='llama2')

        assert client.model == 'llama2'
        assert client.api_key == ''  # Empty string for compatibility
        assert client.base_url == 'http://localhost:11434'
        assert client.timeout == 120

    def test_initialization_with_custom_base_url(self):
        """Test OllamaClient initialization with custom base URL."""
        from src.llms.ollama_client import OllamaClient

        client = OllamaClient(model='mistral', base_url='http://ollama-server:8080')

        assert client.base_url == 'http://ollama-server:8080'
        assert client.model == 'mistral'

    def test_initialization_strips_trailing_slash(self):
        """Test that trailing slash is removed from base URL."""
        from src.llms.ollama_client import OllamaClient

        client = OllamaClient(model='llama2', base_url='http://localhost:11434/')

        assert client.base_url == 'http://localhost:11434'

    def test_initialization_with_custom_timeout(self):
        """Test OllamaClient initialization with custom timeout."""
        from src.llms.ollama_client import OllamaClient

        client = OllamaClient(model='llama2', timeout=60)

        assert client.timeout == 60

    def test_initialization_with_api_key(self):
        """Test that API key is accepted (for interface compatibility)."""
        from src.llms.ollama_client import OllamaClient

        client = OllamaClient(model='llama2', api_key='not-needed-but-accepted')

        assert client.api_key == 'not-needed-but-accepted'

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_basic_call(self, mock_post):
        """Test basic conversation call with standard response format."""
        from src.llms.ollama_client import OllamaClient

        # Mock standard Ollama response format
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?"
            },
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2')
        result = client.conv("Hello")

        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == 'http://localhost:11434/api/chat'

        payload = call_args[1]['json']
        assert payload['model'] == 'llama2'
        assert payload['messages'] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        assert payload['stream'] is False
        assert payload['options']['temperature'] == 0.7
        assert payload['options']['num_predict'] == 500

        assert result == "Hello! How can I help you?"

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_with_custom_system_message(self, mock_post):
        """Test conversation with custom system message."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "I'm a coding expert."
            },
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='codellama')
        result = client.conv(
            "Help me code",
            system_message="You are an expert programmer."
        )

        payload = mock_post.call_args[1]['json']
        assert payload['messages'][0]['content'] == "You are an expert programmer."
        assert result == "I'm a coding expert."

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_with_temperature(self, mock_post):
        """Test conversation with custom temperature."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2')
        client.conv("Test", temperature=0.2)

        payload = mock_post.call_args[1]['json']
        assert payload['options']['temperature'] == 0.2

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_with_max_tokens(self, mock_post):
        """Test conversation with custom max_tokens."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2')
        client.conv("Test", max_tokens=100)

        payload = mock_post.call_args[1]['json']
        assert payload['options']['num_predict'] == 100

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_with_none_max_tokens(self, mock_post):
        """Test that None max_tokens becomes -1 (unlimited)."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2')
        client.conv("Test", max_tokens=None)

        payload = mock_post.call_args[1]['json']
        assert payload['options']['num_predict'] == -1

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_response_format_fallback(self, mock_post):
        """Test fallback to 'response' field in response."""
        from src.llms.ollama_client import OllamaClient

        # Alternative response format
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "This is a response string",
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2')
        result = client.conv("Test")

        assert result == "This is a response string"

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_messages_array_format(self, mock_post):
        """Test fallback to messages array in response."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": [
                {"role": "assistant", "content": "Part 1 "},
                {"role": "assistant", "content": "Part 2"}
            ],
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2')
        result = client.conv("Test")

        # Should concatenate message contents
        assert result == "Part 1 Part 2"

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_unknown_format_stringify(self, mock_post):
        """Test that unknown response format is stringified."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "unknown_field": "unknown_value",
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2')
        result = client.conv("Test")

        # Should convert to string
        assert "unknown_field" in result
        assert "unknown_value" in result

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_empty_content(self, mock_post):
        """Test handling of empty content in response."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2')
        result = client.conv("Test")

        assert result == ""

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_missing_content_field(self, mock_post):
        """Test handling of missing content field."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant"
                # No content field
            },
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2')
        result = client.conv("Test")

        # Should return empty string
        assert result == ""

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_http_error(self, mock_post):
        """Test handling of HTTP errors."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2')

        with pytest.raises(RuntimeError) as exc_info:
            client.conv("Test")

        error_msg = str(exc_info.value)
        assert "Ollama HTTP error" in error_msg
        assert "500" in error_msg
        assert "Internal Server Error" in error_msg

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_connection_error(self, mock_post):
        """Test handling of connection errors."""
        from src.llms.ollama_client import OllamaClient

        mock_post.side_effect = requests.ConnectionError("Connection refused")

        client = OllamaClient(model='llama2')

        with pytest.raises(RuntimeError) as exc_info:
            client.conv("Test")

        error_msg = str(exc_info.value)
        assert "Ollama request failed" in error_msg
        assert "Connection refused" in error_msg

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_timeout_error(self, mock_post):
        """Test handling of timeout errors."""
        from src.llms.ollama_client import OllamaClient

        mock_post.side_effect = requests.Timeout("Request timed out")

        client = OllamaClient(model='llama2')

        with pytest.raises(RuntimeError) as exc_info:
            client.conv("Test")

        error_msg = str(exc_info.value)
        assert "Ollama request failed" in error_msg
        assert "timed out" in error_msg.lower()

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_custom_base_url_used(self, mock_post):
        """Test that custom base URL is used in requests."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2', base_url='http://custom-server:9999')
        client.conv("Test")

        # Verify custom URL was used
        call_args = mock_post.call_args
        assert call_args[0][0] == 'http://custom-server:9999/api/chat'

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_custom_timeout_used(self, mock_post):
        """Test that custom timeout is passed to requests."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2', timeout=30)
        client.conv("Test")

        # Verify timeout was passed
        call_args = mock_post.call_args
        assert call_args[1]['timeout'] == 30

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_with_multiline_response(self, mock_post):
        """Test conversation with multiline response."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Line 1\nLine 2\nLine 3"
            },
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2')
        result = client.conv("Test")

        # Multiline should be preserved
        assert result == "Line 1\nLine 2\nLine 3"

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_with_unicode_response(self, mock_post):
        """Test conversation with unicode characters."""
        from src.llms.ollama_client import OllamaClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "你好世界 🌍 Привет"
            },
            "done": True
        }
        mock_post.return_value = mock_response

        client = OllamaClient(model='llama2')
        result = client.conv("Test")

        # Unicode should be preserved
        assert result == "你好世界 🌍 Привет"

    @patch('src.llms.ollama_client.requests.post')
    def test_conv_multiple_calls_same_client(self, mock_post):
        """Test multiple conversation calls with same client."""
        from src.llms.ollama_client import OllamaClient

        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "message": {"role": "assistant", "content": "First"},
            "done": True
        }

        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "message": {"role": "assistant", "content": "Second"},
            "done": True
        }

        mock_post.side_effect = [mock_response1, mock_response2]

        client = OllamaClient(model='llama2')
        result1 = client.conv("First message")
        result2 = client.conv("Second message")

        assert result1 == "First"
        assert result2 == "Second"
        assert mock_post.call_count == 2
