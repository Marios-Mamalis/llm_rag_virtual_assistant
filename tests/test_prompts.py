import pytest
import string
from src.prompts import fill_rag_prompt


class TestFillRagPrompt:
    def test_fill_rag_prompt_valid(self):
        template = string.Template("Context: $context_pieces\nQuery: $user_query")
        context = ["Context1", "Context2"]
        query = "Question?"
        assert fill_rag_prompt(template, context, query) == "Context: Context1\nContext2\nQuery: Question?"

    def test_empty_user_query(self):
        template = string.Template("Context: $context_pieces\nQuery: $user_query")
        context = ["Context1", "Context2"]
        user_query = ""

        with pytest.raises(AssertionError, match="User query is missing"):
            fill_rag_prompt(template, context, user_query)

    def test_empty_context(self):
        template = string.Template("Context: $context_pieces\nQuery: $user_query")
        context = []
        user_query = "Question?"

        with pytest.raises(AssertionError, match="Context pieces are missing"):
            fill_rag_prompt(template, context, user_query)

    def test_only_empty_in_context(self):
        template = string.Template("Context: $context_pieces\nQuery: $user_query")
        context = ["   ", "\t", '\n']
        user_query = "Question?"

        with pytest.raises(AssertionError, match="Context pieces are missing"):
            fill_rag_prompt(template, context, user_query)

    def test_valid_input_single_context(self):
        template = string.Template("Context: $context_pieces\nQuery: $user_query")
        context = ["Context1"]
        user_query = "Question?"
        assert fill_rag_prompt(template, context, user_query) == "Context: Context1\nQuery: Question?"