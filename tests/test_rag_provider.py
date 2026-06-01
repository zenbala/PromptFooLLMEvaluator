"""
tests/test_rag_provider.py
──────────────────────────
Unit tests for the RAG provider — run with:
    pytest tests/test_rag_provider.py -v
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from providers.rag_provider import (
    retrieve_context,
    build_prompt,
    load_prompt_template,
    call_api,
)


# ── retrieve_context ─────────────────────────────────────────────────────────

class TestRetrieveContext:
    def test_returns_relevant_chunk_for_return_policy(self):
        result = retrieve_context("What is your return policy?")
        assert "30 days" in result

    def test_returns_relevant_chunk_for_pricing(self):
        result = retrieve_context("How much does the pro plan cost?")
        assert "$49" in result

    def test_returns_fallback_for_unknown_query(self):
        result = retrieve_context("xyzzy frobulate quux")
        assert "No specific documentation" in result

    def test_respects_top_k(self):
        result = retrieve_context("plan pricing support", top_k=1)
        # Should return only the single top chunk — no double newline separator
        assert result.count("\n\n") == 0

    def test_case_insensitive_matching(self):
        lower = retrieve_context("return policy")
        upper = retrieve_context("RETURN POLICY")
        assert lower == upper


# ── build_prompt ─────────────────────────────────────────────────────────────

class TestBuildPrompt:
    TEMPLATE = "Context: {{context}}\nHistory: {{history}}\nQ: {{question}}"

    def test_substitutes_all_vars(self):
        result = build_prompt(self.TEMPLATE, "ctx", "hist", "question?")
        assert "ctx" in result
        assert "hist" in result
        assert "question?" in result

    def test_empty_history_replaced(self):
        result = build_prompt(self.TEMPLATE, "ctx", "", "q")
        assert "{{history}}" not in result
        assert "(none)" in result

    def test_no_leftover_placeholders(self):
        result = build_prompt(self.TEMPLATE, "c", "h", "q")
        assert "{{" not in result


# ── load_prompt_template ─────────────────────────────────────────────────────

class TestLoadPromptTemplate:
    def test_returns_string(self):
        template = load_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0

    def test_fallback_on_missing_file(self):
        template = load_prompt_template("nonexistent_path.txt")
        assert "{{context}}" in template
        assert "{{question}}" in template


# ── call_api (integration-level, mocked LLM) ─────────────────────────────────

class TestCallApi:
    CONTEXT = {"vars": {"question": "What is the return policy?", "history": "", "context": "30 days return window."}}

    @patch("providers.rag_provider.call_openai", return_value="You can return within 30 days.")
    def test_openai_backend_called(self, mock_openai):
        result = call_api("", {"config": {"model": "openai"}}, self.CONTEXT)
        assert mock_openai.called
        assert "30 days" in result["output"]

    @patch("providers.rag_provider.call_anthropic", return_value="Returns are accepted within 30 days.")
    def test_anthropic_backend_called(self, mock_anthropic):
        result = call_api("", {"config": {"model": "anthropic"}}, self.CONTEXT)
        assert mock_anthropic.called
        assert "30" in result["output"]

    @patch("providers.rag_provider.call_openai", return_value="Some answer.")
    def test_metadata_included(self, _):
        result = call_api("", {}, self.CONTEXT)
        assert "metadata" in result
        assert "retrieved_context" in result["metadata"]

    @patch("providers.rag_provider.call_openai", side_effect=RuntimeError("API key missing"))
    def test_error_returned_gracefully(self, _):
        result = call_api("", {}, self.CONTEXT)
        assert "error" in result
        assert result["output"] == ""

    @patch("providers.rag_provider.call_openai", return_value="Answer.")
    def test_uses_context_var_over_retrieval(self, mock_openai):
        """When a test provides explicit 'context' var, it should be used as-is."""
        ctx = {"vars": {"question": "test", "history": "", "context": "EXPLICIT_CONTEXT_OVERRIDE"}}
        call_api("", {}, ctx)
        call_args = mock_openai.call_args[0][0]
        assert "EXPLICIT_CONTEXT_OVERRIDE" in call_args
