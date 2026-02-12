import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import cli_app


class TestCliBackends(unittest.TestCase):
    def test_parse_context_limit_error_extracts_token_counts(self):
        response = (
            "Error querying DSPy: OpenrouterException - "
            '{"error":{"message":"This endpoint\'s maximum context length is 256000 '
            "tokens. However, you requested about 293617 tokens (293617 of text input).\"}}"
        )
        self.assertEqual(
            cli_app._parse_context_limit_error(response),
            (256000, 293617),
        )

    def test_parse_context_limit_error_returns_none_when_not_present(self):
        self.assertIsNone(
            cli_app._parse_context_limit_error("Error querying DSPy: timed out after 180.0s")
        )

    def test_with_openrouter_middle_out_adds_transform(self):
        updated = cli_app._with_openrouter_middle_out(None)
        self.assertEqual(updated["extra_body"]["transforms"], ["middle-out"])

    def test_with_openrouter_middle_out_preserves_existing_transforms(self):
        updated = cli_app._with_openrouter_middle_out(
            {"extra_body": {"transforms": ["gzip", "middle-out"]}, "temperature": 0.1}
        )
        self.assertEqual(updated["extra_body"]["transforms"], ["gzip", "middle-out"])
        self.assertEqual(updated["temperature"], 0.1)

    def test_response_has_pdf_page_na_citation_detects_pdf_na(self):
        response = (
            '["quote" [source: cantikluka.pdf, page: n/a], '
            '"quote2" [source: notes.md, page: n/a]]'
        )
        self.assertTrue(
            cli_app._response_has_pdf_page_na_citation(
                response=response,
                pdf_doc_names={"cantikluka.pdf"},
            )
        )

    def test_response_has_pdf_page_na_citation_ignores_non_pdf_sources(self):
        response = '["quote" [source: notes.md, page: n/a]]'
        self.assertFalse(
            cli_app._response_has_pdf_page_na_citation(
                response=response,
                pdf_doc_names={"cantikluka.pdf"},
            )
        )

    def test_merge_run_metrics_with_retry(self):
        merged = cli_app._merge_run_metrics_with_retry(
            primary={
                "mode": "research_direct_fallback",
                "fallback_used": True,
                "execution_time": 4.0,
                "iterations": 1,
                "subagent_calls": 0,
                "total_input_tokens": 30,
                "total_output_tokens": 20,
                "total_tokens": 50,
            },
            retry={
                "mode": "direct_lm",
                "execution_time": 2.0,
                "iterations": 0,
                "subagent_calls": 0,
                "total_input_tokens": 10,
                "total_output_tokens": 10,
                "total_tokens": 20,
            },
            retry_label="citation_repair",
        )
        self.assertEqual(merged["mode"], "citation_repair_retry")
        self.assertEqual(merged["retry_used"], "citation_repair")
        self.assertTrue(merged["fallback_used"])
        self.assertEqual(merged["execution_time"], 6.0)
        self.assertEqual(merged["total_tokens"], 70)

    def test_should_use_research_direct_fallback_for_timeout(self):
        self.assertTrue(
            cli_app._should_use_research_direct_fallback(
                output_mode="research",
                use_subagents=True,
                response="Error querying DSPy: timed out after 120.0s",
                metrics={"error": "query_timeout"},
            )
        )

    def test_should_not_use_research_direct_fallback_in_concise_mode(self):
        self.assertFalse(
            cli_app._should_use_research_direct_fallback(
                output_mode="concise",
                use_subagents=True,
                response="Error querying DSPy: timed out after 120.0s",
                metrics={"error": "query_timeout"},
            )
        )

    def test_merge_run_metrics_with_fallback(self):
        merged = cli_app._merge_run_metrics_with_fallback(
            primary={
                "mode": "dspy_rlm",
                "error": "query_timeout",
                "execution_time": 10.0,
                "iterations": 3,
                "subagent_calls": 2,
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "total_tokens": 150,
            },
            fallback={
                "mode": "direct_lm",
                "execution_time": 2.0,
                "iterations": 0,
                "subagent_calls": 0,
                "total_input_tokens": 40,
                "total_output_tokens": 20,
                "total_tokens": 60,
            },
        )
        self.assertEqual(merged["mode"], "research_direct_fallback")
        self.assertTrue(merged["fallback_used"])
        self.assertEqual(merged["primary_error"], "query_timeout")
        self.assertEqual(merged["execution_time"], 12.0)
        self.assertEqual(merged["iterations"], 3)
        self.assertEqual(merged["subagent_calls"], 2)
        self.assertEqual(merged["total_tokens"], 210)

    def test_normalize_output_mode_aliases(self):
        self.assertEqual(cli_app._normalize_output_mode("short"), "concise")
        self.assertEqual(cli_app._normalize_output_mode("full"), "research")
        self.assertEqual(cli_app._normalize_output_mode("unknown"), "")

    @patch.dict(os.environ, {"DSPY_OUTPUT_MODE": "brief"}, clear=True)
    def test_resolve_default_output_mode_prefers_env(self):
        self.assertEqual(
            cli_app._resolve_default_output_mode({"output_mode": "research"}),
            "concise",
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_resolve_default_output_mode_uses_prompt_var(self):
        self.assertEqual(
            cli_app._resolve_default_output_mode({"output_mode": "long"}),
            "research",
        )

    def test_output_template_for_mode(self):
        concise_template = cli_app._output_template_for_mode("concise")
        research_template = cli_app._output_template_for_mode("research")
        self.assertIn("2-4 bullets", concise_template)
        self.assertIn("3-8 bullets", research_template)

    @patch.dict(os.environ, {"DSPY_RLM_MAX_ITERATIONS": "12"}, clear=True)
    def test_parse_positive_int_env_returns_value(self):
        self.assertEqual(cli_app._parse_positive_int_env("DSPY_RLM_MAX_ITERATIONS"), 12)

    @patch.dict(os.environ, {"DSPY_QUERY_TIMEOUT_SECONDS": "45.5"}, clear=True)
    def test_parse_positive_float_env_returns_value(self):
        self.assertEqual(
            cli_app._parse_positive_float_env("DSPY_QUERY_TIMEOUT_SECONDS"),
            45.5,
        )

    @patch.dict(os.environ, {"DSPY_QUERY_TIMEOUT_SECONDS": "abc"}, clear=True)
    def test_parse_positive_float_env_rejects_invalid(self):
        self.assertIsNone(cli_app._parse_positive_float_env("DSPY_QUERY_TIMEOUT_SECONDS"))

    @patch("src.cli_app.mp.get_all_start_methods", return_value=["fork", "spawn"])
    def test_preferred_mp_start_method_prefers_fork(self, _mock_methods):
        self.assertEqual(cli_app._preferred_mp_start_method(), "fork")

    @patch("src.cli_app.mp.get_all_start_methods", return_value=["spawn"])
    def test_preferred_mp_start_method_falls_back_to_spawn(self, _mock_methods):
        self.assertEqual(cli_app._preferred_mp_start_method(), "spawn")

    @patch.dict(os.environ, {"DSPY_RLM_MAX_ITERATIONS": "abc"}, clear=True)
    def test_parse_positive_int_env_rejects_invalid(self):
        self.assertIsNone(cli_app._parse_positive_int_env("DSPY_RLM_MAX_ITERATIONS"))

    @patch.dict(os.environ, {"DSPY_BACKEND": "unknown"}, clear=True)
    def test_resolve_backend_rejects_unsupported_backend(self):
        with self.assertRaises(ValueError):
            cli_app._resolve_backend_and_key()

    @patch.dict(
        os.environ,
        {
            "DSPY_BACKEND": "openrouter",
            "OPENROUTER_API_KEY": "or-key",
        },
        clear=True,
    )
    def test_resolve_backend_with_configured_dspy_backend(self):
        backend, api_key = cli_app._resolve_backend_and_key()
        self.assertEqual(backend, "openrouter")
        self.assertEqual(api_key, "or-key")

    @patch.dict(
        os.environ,
        {
            "DSPY_BACKEND": "anthropic",
        },
        clear=True,
    )
    def test_resolve_backend_requires_anthropic_key(self):
        with self.assertRaises(ValueError):
            cli_app._resolve_backend_and_key()

    @patch.dict(
        os.environ,
        {
            "DSPY_BACKEND": "azure_openai",
            "AZURE_OPENAI_API_KEY": "azure-key",
        },
        clear=True,
    )
    def test_resolve_backend_requires_azure_endpoint(self):
        with self.assertRaises(ValueError):
            cli_app._resolve_backend_and_key()

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "or-key",
            "OPENAI_API_KEY": "oa-key",
        },
        clear=True,
    )
    def test_autodetect_prefers_openrouter(self):
        backend, api_key = cli_app._resolve_backend_and_key()
        self.assertEqual(backend, "openrouter")
        self.assertEqual(api_key, "or-key")

    @patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "gem-key",
        },
        clear=True,
    )
    def test_autodetect_uses_gemini_when_it_is_only_available(self):
        backend, api_key = cli_app._resolve_backend_and_key()
        self.assertEqual(backend, "gemini")
        self.assertEqual(api_key, "gem-key")

    @patch.dict(os.environ, {}, clear=True)
    def test_resolve_root_model_default_uses_backend_default(self):
        self.assertEqual(
            cli_app._resolve_root_model_default("openrouter"),
            "openrouter/openai/gpt-5-mini",
        )

    @patch.dict(os.environ, {"DSPY_MODEL": "openrouter/free"}, clear=True)
    def test_resolve_root_model_default_prefers_dspy_model_env(self):
        self.assertEqual(
            cli_app._resolve_root_model_default("openrouter"),
            "openrouter/free",
        )

    @patch.dict(os.environ, {"OPENROUTER_MODEL": "openrouter/free"}, clear=True)
    def test_resolve_root_model_default_supports_backend_specific_model_env(self):
        self.assertEqual(
            cli_app._resolve_root_model_default("openrouter"),
            "openrouter/free",
        )

    @patch.dict(os.environ, {"DSPY_MODEL": "gpt-4.1-mini"}, clear=True)
    def test_resolve_root_model_default_normalizes_openrouter_shorthand(self):
        self.assertEqual(
            cli_app._resolve_root_model_default("openrouter"),
            "openrouter/openai/gpt-4.1-mini",
        )


if __name__ == "__main__":
    unittest.main()
