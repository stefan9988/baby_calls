import pytest
from unittest.mock import Mock, patch, MagicMock


class TestChatGPTClient:
    """Test suite for ChatGPTClient."""

    @patch('src.llms.openai_api.OpenAI')
    def test_initialization_with_default_model(self, mock_openai):
        """Test ChatGPTClient initialization with default model."""
        from src.llms.openai_api import ChatGPTClient

        mock_openai_instance = Mock()
        mock_openai.return_value = mock_openai_instance

        client = ChatGPTClient(api_key='test-api-key')

        # Verify OpenAI client was created with correct API key
        mock_openai.assert_called_once_with(api_key='test-api-key')
        # Verify default model is set
        assert client.model == 'gpt-4o'
        assert client.api_key == 'test-api-key'
        assert client.client == mock_openai_instance

    @patch('src.llms.openai_api.OpenAI')
    def test_initialization_with_custom_model(self, mock_openai):
        """Test ChatGPTClient initialization with custom model."""
        from src.llms.openai_api import ChatGPTClient

        mock_openai_instance = Mock()
        mock_openai.return_value = mock_openai_instance

        client = ChatGPTClient(api_key='test-api-key', model='gpt-3.5-turbo')

        mock_openai.assert_called_once_with(api_key='test-api-key')
        assert client.model == 'gpt-3.5-turbo'
        assert client.api_key == 'test-api-key'

    @patch('src.llms.openai_api.OpenAI')
    def test_conv_basic_call(self, mock_openai):
        """Test basic conversation call with minimal parameters."""
        from src.llms.openai_api import ChatGPTClient

        # Setup mock response
        mock_message = Mock()
        mock_message.content = "Hello! How can I help you?"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_openai_instance

        client = ChatGPTClient(api_key='test-key', model='gpt-4')
        result = client.conv("Hello")

        # Verify API was called correctly
        mock_openai_instance.chat.completions.create.assert_called_once_with(
            model='gpt-4',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
            max_completion_tokens=500
        )
        assert result == "Hello! How can I help you?"

    @patch('src.llms.openai_api.OpenAI')
    def test_conv_with_custom_system_message(self, mock_openai):
        """Test conversation with custom system message."""
        from src.llms.openai_api import ChatGPTClient

        mock_message = Mock()
        mock_message.content = "Sure, I'll help with coding."
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_openai_instance

        client = ChatGPTClient(api_key='test-key')
        result = client.conv(
            "Write a function",
            system_message="You are an expert Python programmer."
        )

        # Verify custom system message was used
        call_args = mock_openai_instance.chat.completions.create.call_args
        assert call_args[1]['messages'][0]['content'] == "You are an expert Python programmer."
        assert call_args[1]['messages'][1]['content'] == "Write a function"
        assert result == "Sure, I'll help with coding."

    @patch('src.llms.openai_api.OpenAI')
    def test_conv_with_max_tokens(self, mock_openai):
        """Test conversation with custom max_tokens."""
        from src.llms.openai_api import ChatGPTClient

        mock_message = Mock()
        mock_message.content = "Short response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_openai_instance

        client = ChatGPTClient(api_key='test-key')
        result = client.conv("Tell me a story", max_tokens=100)

        # Verify max_tokens was passed
        call_args = mock_openai_instance.chat.completions.create.call_args
        assert call_args[1]['max_completion_tokens'] == 100
        assert result == "Short response"

    @patch('src.llms.openai_api.OpenAI')
    def test_conv_with_additional_kwargs(self, mock_openai):
        """Test conversation with additional OpenAI parameters."""
        from src.llms.openai_api import ChatGPTClient

        mock_message = Mock()
        mock_message.content = '{"result": "json response"}'
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_openai_instance

        client = ChatGPTClient(api_key='test-key')
        result = client.conv(
            "Generate JSON",
            response_format={"type": "json_object"}
        )

        # Verify response_format was passed through kwargs
        call_args = mock_openai_instance.chat.completions.create.call_args
        assert call_args[1]['response_format'] == {"type": "json_object"}
        assert result == '{"result": "json response"}'

    @patch('src.llms.openai_api.OpenAI')
    def test_conv_strips_whitespace(self, mock_openai):
        """Test that response content is stripped of whitespace."""
        from src.llms.openai_api import ChatGPTClient

        mock_message = Mock()
        mock_message.content = "  \n  Response with whitespace  \n  "
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_openai_instance

        client = ChatGPTClient(api_key='test-key')
        result = client.conv("Test")

        # Verify whitespace was stripped
        assert result == "Response with whitespace"

    @patch('src.llms.openai_api.OpenAI')
    def test_conv_with_empty_response(self, mock_openai):
        """Test handling of empty response content."""
        from src.llms.openai_api import ChatGPTClient

        mock_message = Mock()
        mock_message.content = "   "
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_openai_instance

        client = ChatGPTClient(api_key='test-key')
        result = client.conv("Test")

        # Should return empty string after stripping
        assert result == ""

    @patch('src.llms.openai_api.OpenAI')
    def test_conv_with_multiline_response(self, mock_openai):
        """Test conversation with multiline response."""
        from src.llms.openai_api import ChatGPTClient

        mock_message = Mock()
        mock_message.content = "Line 1\nLine 2\nLine 3"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_openai_instance

        client = ChatGPTClient(api_key='test-key')
        result = client.conv("Tell me three things")

        # Multiline content should be preserved
        assert result == "Line 1\nLine 2\nLine 3"

    @patch('src.llms.openai_api.OpenAI')
    def test_conv_with_unicode_content(self, mock_openai):
        """Test conversation with unicode characters."""
        from src.llms.openai_api import ChatGPTClient

        mock_message = Mock()
        mock_message.content = "Привет! 你好! 🚀"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_openai_instance

        client = ChatGPTClient(api_key='test-key')
        result = client.conv("Say hello in multiple languages")

        # Unicode should be preserved
        assert result == "Привет! 你好! 🚀"

    @patch('src.llms.openai_api.OpenAI')
    def test_conv_multiple_calls(self, mock_openai):
        """Test multiple conversation calls with same client."""
        from src.llms.openai_api import ChatGPTClient

        mock_message1 = Mock()
        mock_message1.content = "First response"
        mock_choice1 = Mock()
        mock_choice1.message = mock_message1
        mock_response1 = Mock()
        mock_response1.choices = [mock_choice1]

        mock_message2 = Mock()
        mock_message2.content = "Second response"
        mock_choice2 = Mock()
        mock_choice2.message = mock_message2
        mock_response2 = Mock()
        mock_response2.choices = [mock_choice2]

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.side_effect = [
            mock_response1,
            mock_response2
        ]
        mock_openai.return_value = mock_openai_instance

        client = ChatGPTClient(api_key='test-key')
        result1 = client.conv("First message")
        result2 = client.conv("Second message")

        # Verify both calls worked correctly
        assert result1 == "First response"
        assert result2 == "Second response"
        assert mock_openai_instance.chat.completions.create.call_count == 2

    @patch('src.llms.openai_api.OpenAI')
    def test_temperature_parameter_documented_but_not_used(self, mock_openai):
        """Test that temperature parameter is accepted but not currently used."""
        from src.llms.openai_api import ChatGPTClient

        mock_message = Mock()
        mock_message.content = "Response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_openai_instance

        client = ChatGPTClient(api_key='test-key')
        # Temperature parameter should be accepted (for backwards compatibility)
        result = client.conv("Test", temperature=0.9)

        # Verify temperature is not passed to API (commented out in implementation)
        call_args = mock_openai_instance.chat.completions.create.call_args
        assert 'temperature' not in call_args[1]
        assert result == "Response"

    @patch('src.llms.openai_api.OpenAI')
    def test_initialization_with_none_api_key(self, mock_openai):
        """Test initialization with None API key (should work, fail on actual call)."""
        from src.llms.openai_api import ChatGPTClient

        mock_openai_instance = Mock()
        mock_openai.return_value = mock_openai_instance

        # Should not raise during initialization
        client = ChatGPTClient(api_key=None)

        mock_openai.assert_called_once_with(api_key=None)
        assert client.api_key is None
