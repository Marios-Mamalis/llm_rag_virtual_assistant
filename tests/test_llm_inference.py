import pytest
from unittest.mock import patch, MagicMock
from src.llm_inference import openai_inference


class TestOpenaiInference:
    def set_nontest_vals(self):
        self.api_key = "api_key"
        self.api_version = "api_version"
        self.endpoint = "endpoint"
        self.deployment = "deployment"

    @patch("src.llm_inference.AzureOpenAI")
    def test_openai_inference_valid(self, MockAzureOpenAI):
        self.set_nontest_vals()
        self.conversation_history = [
            {"role": "user", "content": "Input"},
            {"role": "assistant", "content": "Output"}
        ]
        self.user_query = "user_query"

        mock_client = MagicMock()
        MockAzureOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Response"))]

        result = openai_inference(self.conversation_history, self.user_query, self.api_key, self.api_version,
                                  self.endpoint, self.deployment)

        assert len(result) == 4  # assert correct number of dialogue truns
        assert result[-1]['role'] == 'assistant'  # assert that the last dialogue belongst to the assistant
        assert result[-1]['content'] == "Response"  # assert that the response was correctly inserted

    @patch("src.llm_inference.AzureOpenAI")
    def test_invalid_conversation_history(self, MockAzureOpenAI):
        self.set_nontest_vals()
        self.conversation_history = [{"role": "user", "content": "Input"}]
        self.user_query = "user_query"

        mock_client = MagicMock()
        MockAzureOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Response"))]

        with pytest.raises(AssertionError):
            openai_inference(self.conversation_history, self.user_query, self.api_key, self.api_version,
                             self.endpoint, self.deployment)
