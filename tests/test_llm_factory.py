import pytest
from unittest.mock import patch, Mock


class TestGetLLMClient:
    """Test suite for get_llm_client factory function."""

    @patch('src.llms.openai_api.ChatGPTClient')
    @patch('src.llms.llm_factory.os.getenv')
    def test_openai_with_explicit_api_key(self, mock_getenv, mock_chatgpt_client):
        """Test OpenAI client creation with explicit API key."""
        from src.llms.llm_factory import get_llm_client

        mock_client_instance = Mock()
        mock_chatgpt_client.return_value = mock_client_instance

        client = get_llm_client('openai', api_key='test-api-key', model='gpt-4')

        # Verify ChatGPTClient was called with correct parameters
        mock_chatgpt_client.assert_called_once_with(
            api_key='test-api-key',
            model='gpt-4'
        )
        # Verify OPENAI_API_KEY environment variable was not checked (explicit key provided)
        assert not any(call[0][0] == 'OPENAI_API_KEY' for call in mock_getenv.call_args_list if call[0])
        # Verify correct client instance returned
        assert client == mock_client_instance

    @patch('src.llms.openai_api.ChatGPTClient')
    @patch('src.llms.llm_factory.os.getenv')
    def test_openai_with_env_var(self, mock_getenv, mock_chatgpt_client):
        """Test OpenAI client creation using environment variable."""
        from src.llms.llm_factory import get_llm_client

        mock_getenv.return_value = 'env-api-key'
        mock_client_instance = Mock()
        mock_chatgpt_client.return_value = mock_client_instance

        client = get_llm_client('openai', model='gpt-3.5-turbo')

        # Verify environment variable was checked
        mock_getenv.assert_called_once_with('OPENAI_API_KEY')
        # Verify ChatGPTClient was called with env var
        mock_chatgpt_client.assert_called_once_with(
            api_key='env-api-key',
            model='gpt-3.5-turbo'
        )
        assert client == mock_client_instance

    @patch('src.llms.openai_api.ChatGPTClient')
    @patch('src.llms.llm_factory.os.getenv')
    def test_openai_explicit_key_overrides_env(self, mock_getenv, mock_chatgpt_client):
        """Test that explicit API key takes precedence over environment variable."""
        from src.llms.llm_factory import get_llm_client

        mock_getenv.return_value = 'env-api-key'
        mock_client_instance = Mock()
        mock_chatgpt_client.return_value = mock_client_instance

        client = get_llm_client('openai', api_key='explicit-key', model='gpt-4')

        # Verify explicit key was used, not env var
        mock_chatgpt_client.assert_called_once_with(
            api_key='explicit-key',
            model='gpt-4'
        )
        # The fact that explicit-key was used proves env var was not used

    @patch('src.llms.openai_api.ChatGPTClient')
    @patch('src.llms.llm_factory.os.getenv')
    def test_openai_no_api_key_provided(self, mock_getenv, mock_chatgpt_client):
        """Test OpenAI client when no API key available."""
        from src.llms.llm_factory import get_llm_client

        mock_getenv.return_value = None
        mock_client_instance = Mock()
        mock_chatgpt_client.return_value = mock_client_instance

        client = get_llm_client('openai', model='gpt-4')

        # Should still create client with None (will fail on actual API call)
        mock_chatgpt_client.assert_called_once_with(
            api_key=None,
            model='gpt-4'
        )

    @patch('src.llms.huggingface_client.HuggingFaceLLM')
    @patch('src.llms.llm_factory.os.getenv')
    def test_huggingface_with_explicit_api_key(self, mock_getenv, mock_hf_client):
        """Test HuggingFace client creation with explicit API key."""
        from src.llms.llm_factory import get_llm_client

        mock_client_instance = Mock()
        mock_hf_client.return_value = mock_client_instance

        client = get_llm_client(
            'huggingface',
            api_key='hf-test-key',
            model='meta-llama/Llama-2-7b',
            device='cuda'
        )

        mock_hf_client.assert_called_once_with(
            model_id='meta-llama/Llama-2-7b',
            api_key='hf-test-key',
            device='cuda'
        )
        mock_getenv.assert_not_called()
        assert client == mock_client_instance

    @patch('src.llms.huggingface_client.HuggingFaceLLM')
    @patch('src.llms.llm_factory.os.getenv')
    def test_huggingface_with_env_var(self, mock_getenv, mock_hf_client):
        """Test HuggingFace client creation using environment variable."""
        from src.llms.llm_factory import get_llm_client

        mock_getenv.return_value = 'env-hf-key'
        mock_client_instance = Mock()
        mock_hf_client.return_value = mock_client_instance

        client = get_llm_client('huggingface', model='gpt2', device='cpu')

        mock_getenv.assert_called_once_with('HUGGINGFACE_API_KEY')
        mock_hf_client.assert_called_once_with(
            model_id='gpt2',
            api_key='env-hf-key',
            device='cpu'
        )

    @patch('src.llms.huggingface_client.HuggingFaceLLM')
    @patch('src.llms.llm_factory.os.getenv')
    def test_huggingface_without_device(self, mock_getenv, mock_hf_client):
        """Test HuggingFace client creation without device specified."""
        from src.llms.llm_factory import get_llm_client

        mock_getenv.return_value = 'hf-key'
        mock_client_instance = Mock()
        mock_hf_client.return_value = mock_client_instance

        client = get_llm_client('huggingface', model='gpt2')

        # Device should be None when not specified
        mock_hf_client.assert_called_once_with(
            model_id='gpt2',
            api_key='hf-key',
            device=None
        )

    @patch('src.llms.ollama_client.OllamaClient')
    @patch('src.llms.llm_factory.os.getenv')
    def test_ollama_with_base_url(self, mock_getenv, mock_ollama_client):
        """Test Ollama client creation with explicit base_url."""
        from src.llms.llm_factory import get_llm_client

        mock_client_instance = Mock()
        mock_ollama_client.return_value = mock_client_instance

        client = get_llm_client(
            'ollama',
            model='llama2',
            base_url='http://localhost:11434',
            timeout=60
        )

        mock_ollama_client.assert_called_once_with(
            model='llama2',
            api_key=None,
            base_url='http://localhost:11434',
            timeout=60
        )
        mock_getenv.assert_not_called()
        assert client == mock_client_instance

    @patch('src.llms.ollama_client.OllamaClient')
    @patch('src.llms.llm_factory.os.getenv')
    def test_ollama_with_env_base_url(self, mock_getenv, mock_ollama_client):
        """Test Ollama client creation using environment variable for base_url."""
        from src.llms.llm_factory import get_llm_client

        mock_getenv.return_value = 'http://ollama-server:11434'
        mock_client_instance = Mock()
        mock_ollama_client.return_value = mock_client_instance

        client = get_llm_client('ollama', model='mistral')

        mock_getenv.assert_called_once_with('OLLAMA_BASE_URL')
        mock_ollama_client.assert_called_once_with(
            model='mistral',
            api_key=None,
            base_url='http://ollama-server:11434',
            timeout=120  # Default timeout
        )

    @patch('src.llms.ollama_client.OllamaClient')
    @patch('src.llms.llm_factory.os.getenv')
    def test_ollama_default_timeout(self, mock_getenv, mock_ollama_client):
        """Test Ollama client uses default timeout when not specified."""
        from src.llms.llm_factory import get_llm_client

        mock_getenv.return_value = None
        mock_client_instance = Mock()
        mock_ollama_client.return_value = mock_client_instance

        client = get_llm_client('ollama', model='codellama')

        # Should use default timeout of 120
        mock_ollama_client.assert_called_once_with(
            model='codellama',
            api_key=None,
            base_url=None,
            timeout=120
        )

    @patch('src.llms.ollama_client.OllamaClient')
    @patch('src.llms.llm_factory.os.getenv')
    def test_ollama_with_api_key(self, mock_getenv, mock_ollama_client):
        """Test Ollama client with optional API key."""
        from src.llms.llm_factory import get_llm_client

        mock_getenv.return_value = None  # No env vars set
        mock_client_instance = Mock()
        mock_ollama_client.return_value = mock_client_instance

        client = get_llm_client('ollama', model='llama2', api_key='optional-key')

        mock_ollama_client.assert_called_once_with(
            model='llama2',
            api_key='optional-key',
            base_url=None,
            timeout=120
        )

    def test_invalid_client_type(self):
        """Test that invalid client_type raises ValueError."""
        from src.llms.llm_factory import get_llm_client

        with pytest.raises(ValueError) as exc_info:
            get_llm_client('invalid_provider', model='test')

        # Verify error message contains the invalid type and supported types
        error_msg = str(exc_info.value)
        assert 'invalid_provider' in error_msg
        assert 'openai' in error_msg
        assert 'huggingface' in error_msg
        assert 'ollama' in error_msg
        assert 'Unsupported client_type' in error_msg

    def test_empty_client_type(self):
        """Test that empty client_type raises ValueError."""
        from src.llms.llm_factory import get_llm_client

        with pytest.raises(ValueError) as exc_info:
            get_llm_client('', model='test')

        error_msg = str(exc_info.value)
        assert 'Unsupported client_type' in error_msg

    def test_none_client_type(self):
        """Test that None client_type raises ValueError."""
        from src.llms.llm_factory import get_llm_client

        with pytest.raises(ValueError) as exc_info:
            get_llm_client(None, model='test')

        error_msg = str(exc_info.value)
        assert 'Unsupported client_type' in error_msg

    def test_case_sensitive_client_type(self):
        """Test that client_type is case-sensitive."""
        from src.llms.llm_factory import get_llm_client

        # Should fail with capitalized version
        with pytest.raises(ValueError):
            get_llm_client('OpenAI', model='gpt-4')

        with pytest.raises(ValueError):
            get_llm_client('OLLAMA', model='llama2')

    @patch('src.llms.openai_api.ChatGPTClient')
    @patch('src.llms.llm_factory.os.getenv')
    def test_openai_with_only_model(self, mock_getenv, mock_chatgpt_client):
        """Test OpenAI client with minimal parameters."""
        from src.llms.llm_factory import get_llm_client

        mock_getenv.return_value = 'test-key'
        mock_client_instance = Mock()
        mock_chatgpt_client.return_value = mock_client_instance

        # Should work with just model parameter
        client = get_llm_client('openai', model='gpt-4')

        mock_chatgpt_client.assert_called_once_with(
            api_key='test-key',
            model='gpt-4'
        )
        assert client == mock_client_instance

    @patch('src.llms.huggingface_client.HuggingFaceLLM')
    @patch('src.llms.llm_factory.os.getenv')
    def test_huggingface_extra_kwargs_ignored(self, mock_getenv, mock_hf_client):
        """Test that extra kwargs don't break HuggingFace client creation."""
        from src.llms.llm_factory import get_llm_client

        mock_getenv.return_value = 'hf-key'
        mock_client_instance = Mock()
        mock_hf_client.return_value = mock_client_instance

        # Pass extra kwargs that HF doesn't use
        client = get_llm_client(
            'huggingface',
            model='gpt2',
            timeout=60,  # This is for Ollama, should be ignored
            extra_param='ignored'
        )

        # Should only pass recognized parameters
        mock_hf_client.assert_called_once_with(
            model_id='gpt2',
            api_key='hf-key',
            device=None
        )

