import pytest
from unittest.mock import Mock, patch, MagicMock
import torch


def create_mock_tokenizer_output(input_ids_tensor):
    """Helper to create a mock tokenizer output with .to() method."""
    mock_output = {
        'input_ids': input_ids_tensor
    }
    # Make it support .to() method by wrapping in a class
    class MockTokenizerOutput:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            return self.data[key]

        def keys(self):
            return self.data.keys()

        def values(self):
            return self.data.values()

        def items(self):
            return self.data.items()

        def to(self, device):
            return self

    return MockTokenizerOutput(mock_output)


class TestHuggingFaceLLM:
    """Test suite for HuggingFaceLLM."""

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_initialization_with_cuda_available(self, mock_cuda_available, mock_tokenizer, mock_model):
        """Test HuggingFaceLLM initialization when CUDA is available."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = True
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model_instance = Mock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='gpt2', api_key='test-token')

        # Verify device selection
        assert client.device == 'cuda'
        # Verify model and tokenizer were loaded with token
        mock_tokenizer.from_pretrained.assert_called_once_with('gpt2', token='test-token')
        mock_model.from_pretrained.assert_called_once_with('gpt2', token='test-token')
        # Verify model was moved to device
        mock_model_instance.to.assert_called_once_with('cuda')

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_initialization_with_cuda_unavailable(self, mock_cuda_available, mock_tokenizer, mock_model):
        """Test HuggingFaceLLM initialization when CUDA is not available."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = False
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model_instance = Mock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='gpt2')

        # Should default to CPU
        assert client.device == 'cpu'
        mock_model_instance.to.assert_called_once_with('cpu')

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_initialization_with_explicit_device(self, mock_cuda_available, mock_tokenizer, mock_model):
        """Test HuggingFaceLLM initialization with explicit device."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = True  # Would normally use cuda
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model_instance = Mock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Force CPU even though CUDA is available
        client = HuggingFaceLLM(model_id='gpt2', device='cpu')

        assert client.device == 'cpu'
        mock_model_instance.to.assert_called_once_with('cpu')

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_initialization_without_api_key(self, mock_cuda_available, mock_tokenizer, mock_model):
        """Test initialization without API key (for public models)."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = False
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model_instance = Mock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='gpt2')

        # Should pass None as token
        mock_tokenizer.from_pretrained.assert_called_once_with('gpt2', token=None)
        mock_model.from_pretrained.assert_called_once_with('gpt2', token=None)

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_conv_basic_call(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        """Test basic conversation call."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = False

        # Setup tokenizer mock
        mock_tokenizer_instance = Mock()
        input_tensor = torch.tensor([[1, 2, 3, 4]])
        mock_tokenizer_output = create_mock_tokenizer_output(input_tensor)
        mock_tokenizer_instance.return_value = mock_tokenizer_output
        mock_tokenizer_instance.decode.return_value = "Hello! How can I help you?"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        # Setup model mock
        mock_model_instance = Mock()
        mock_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])  # Extended sequence
        mock_model_instance.generate.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='gpt2')
        result = client.conv("Hello")

        # Verify tokenizer was called
        mock_tokenizer_instance.assert_called_once_with("Hello", return_tensors="pt")

        # Verify model.generate was called
        mock_model_instance.generate.assert_called_once()
        gen_args = mock_model_instance.generate.call_args[1]
        assert gen_args['max_new_tokens'] == 200  # Default
        assert gen_args['do_sample'] is True
        assert gen_args['temperature'] == 0.7  # Default

        # Verify decode was called (with new tokens only)
        assert mock_tokenizer_instance.decode.called

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_conv_with_system_message(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        """Test conversation with system message."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = False

        mock_tokenizer_instance = Mock()
        input_tensor = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_tokenizer_output = create_mock_tokenizer_output(input_tensor)
        mock_tokenizer_instance.return_value = mock_tokenizer_output
        mock_tokenizer_instance.decode.return_value = "Response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='gpt2')
        result = client.conv("Hello", system_message="You are helpful")

        # Verify system message was prepended to prompt
        call_args = mock_tokenizer_instance.call_args
        expected_prompt = "You are helpful\nHello"
        assert call_args[0][0] == expected_prompt

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_conv_without_system_message(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        """Test conversation without system message."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = False

        mock_tokenizer_instance = Mock()
        input_tensor = torch.tensor([[1, 2, 3]])
        mock_tokenizer_output = create_mock_tokenizer_output(input_tensor)
        mock_tokenizer_instance.return_value = mock_tokenizer_output
        mock_tokenizer_instance.decode.return_value = "Response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='gpt2')
        result = client.conv("Hello", system_message="")

        # Should use just the user message (no prepended system message)
        call_args = mock_tokenizer_instance.call_args
        assert call_args[0][0] == "Hello"

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_conv_with_temperature(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        """Test conversation with custom temperature."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = False

        mock_tokenizer_instance = Mock()
        input_tensor = torch.tensor([[1, 2, 3]])
        mock_tokenizer_output = create_mock_tokenizer_output(input_tensor)
        mock_tokenizer_instance.return_value = mock_tokenizer_output
        mock_tokenizer_instance.decode.return_value = "Response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='gpt2')
        result = client.conv("Hello", temperature=0.2)

        # Verify temperature was passed to generate
        gen_args = mock_model_instance.generate.call_args[1]
        assert gen_args['temperature'] == 0.2

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_conv_with_max_tokens(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        """Test conversation with custom max_tokens."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = False

        mock_tokenizer_instance = Mock()
        input_tensor = torch.tensor([[1, 2, 3]])
        mock_tokenizer_output = create_mock_tokenizer_output(input_tensor)
        mock_tokenizer_instance.return_value = mock_tokenizer_output
        mock_tokenizer_instance.decode.return_value = "Response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='gpt2')
        result = client.conv("Hello", max_tokens=50)

        # Verify max_new_tokens was passed to generate
        gen_args = mock_model_instance.generate.call_args[1]
        assert gen_args['max_new_tokens'] == 50

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_conv_strips_whitespace(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        """Test that response is stripped of whitespace."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = False

        mock_tokenizer_instance = Mock()
        input_tensor = torch.tensor([[1, 2, 3]])
        mock_tokenizer_output = create_mock_tokenizer_output(input_tensor)
        mock_tokenizer_instance.return_value = mock_tokenizer_output
        mock_tokenizer_instance.decode.return_value = "  \n  Response with whitespace  \n  "
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='gpt2')
        result = client.conv("Hello")

        # Verify whitespace was stripped
        assert result == "Response with whitespace"

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_conv_removes_input_tokens(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        """Test that input tokens are removed from output."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = False

        mock_tokenizer_instance = Mock()
        # Input has 4 tokens
        mock_input_ids = torch.tensor([[1, 2, 3, 4]])
        mock_tokenizer_output = create_mock_tokenizer_output(mock_input_ids)
        mock_tokenizer_instance.return_value = mock_tokenizer_output
        mock_tokenizer_instance.decode.return_value = "Generated text"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        # Output has 7 tokens (4 input + 3 new)
        mock_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        mock_model_instance.generate.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='gpt2')
        result = client.conv("Hello")

        # Verify only new tokens (after input) were decoded
        decode_call_args = mock_tokenizer_instance.decode.call_args
        decoded_tensor = decode_call_args[0][0]
        # Should be only the new tokens [5, 6, 7]
        assert len(decoded_tensor) == 3

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_conv_skip_special_tokens(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        """Test that special tokens are skipped in decoding."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = False

        mock_tokenizer_instance = Mock()
        input_tensor = torch.tensor([[1, 2, 3]])
        mock_tokenizer_output = create_mock_tokenizer_output(input_tensor)
        mock_tokenizer_instance.return_value = mock_tokenizer_output
        mock_tokenizer_instance.decode.return_value = "Response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='gpt2')
        result = client.conv("Hello")

        # Verify skip_special_tokens=True was passed to decode
        decode_call_args = mock_tokenizer_instance.decode.call_args
        assert decode_call_args[1]['skip_special_tokens'] is True

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_conv_with_unicode(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        """Test conversation with unicode characters."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = False

        mock_tokenizer_instance = Mock()
        input_tensor = torch.tensor([[1, 2, 3]])
        mock_tokenizer_output = create_mock_tokenizer_output(input_tensor)
        mock_tokenizer_instance.return_value = mock_tokenizer_output
        mock_tokenizer_instance.decode.return_value = "你好世界 🌍"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='gpt2')
        result = client.conv("Hello in Chinese")

        # Unicode should be preserved
        assert result == "你好世界 🌍"

    @patch('src.llms.huggingface_client.AutoModelForCausalLM')
    @patch('src.llms.huggingface_client.AutoTokenizer')
    @patch('src.llms.huggingface_client.torch.cuda.is_available')
    def test_model_id_used_for_loading(self, mock_cuda_available, mock_tokenizer_class, mock_model_class):
        """Test that model_id is used correctly when loading model."""
        from src.llms.huggingface_client import HuggingFaceLLM

        mock_cuda_available.return_value = False

        mock_tokenizer_instance = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance

        client = HuggingFaceLLM(model_id='meta-llama/Llama-2-7b', api_key='test-token')

        # Verify model_id was used to load tokenizer and model
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            'meta-llama/Llama-2-7b',
            token='test-token'
        )
        mock_model_class.from_pretrained.assert_called_once_with(
            'meta-llama/Llama-2-7b',
            token='test-token'
        )
        # Note: self.model becomes the actual model instance after __init__
        assert client.model == mock_model_instance
