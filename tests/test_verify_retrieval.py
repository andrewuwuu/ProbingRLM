import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import pdf_utils
from src import rlm_handler


class TestPDFRetrieval(unittest.TestCase):
    def test_list_documents_includes_supported_and_text_like_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = {
                "paper.pdf": b"%PDF-1.7\nfake\n",
                "notes.md": b"# Notes\nhello\n",
                "readme.txt": b"plain text\n",
                "draft.docx": b"PK\x03\x04fake-docx-binary",
                "noext": b"just some text with no extension",
                "image.bin": b"\x00\x01\x02\x03\x04",
            }
            for name, content in files.items():
                with open(os.path.join(tmpdir, name), "wb") as handle:
                    handle.write(content)

            listed = pdf_utils.list_documents(tmpdir)

            self.assertIn("paper.pdf", listed)
            self.assertIn("notes.md", listed)
            self.assertIn("readme.txt", listed)
            self.assertIn("draft.docx", listed)
            self.assertIn("noext", listed)
            self.assertNotIn("image.bin", listed)

    def test_list_pdfs_sorted_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["b.pdf", "a.PDF", "ignore.txt"]:
                with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as handle:
                    handle.write("x")

            self.assertEqual(pdf_utils.list_pdfs(tmpdir), ["a.PDF", "b.pdf"])

    @patch("src.pdf_utils.PdfReader")
    def test_load_pdf(self, mock_pdf_reader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page text"

        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page, mock_page]
        mock_pdf_reader.return_value = mock_reader_instance

        text = pdf_utils.load_pdf("dummy.pdf")

        self.assertEqual(text, "Page text\nPage text")
        mock_pdf_reader.assert_called_with("dummy.pdf")

    @patch("src.pdf_utils.load_pdf")
    def test_load_document_routes_pdf(self, mock_load_pdf):
        mock_load_pdf.return_value = "pdf text"
        result = pdf_utils.load_document("sample.pdf")
        self.assertEqual(result, "pdf text")
        mock_load_pdf.assert_called_once_with("sample.pdf")

    @patch("src.pdf_utils._load_docx")
    def test_load_document_routes_docx(self, mock_load_docx):
        mock_load_docx.return_value = "docx text"
        result = pdf_utils.load_document("sample.docx")
        self.assertEqual(result, "docx text")
        mock_load_docx.assert_called_once_with("sample.docx")

    def test_load_document_reads_plain_text_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "sample.log")
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.write("hello from text file")

            result = pdf_utils.load_document(file_path)
            self.assertEqual(result, "hello from text file")

    @patch("src.rlm_handler.get_client")
    @patch("src.rlm_handler.rlm.RLM")
    def test_rlm_query_without_subagents(self, mock_rlm_class, mock_get_client):
        mock_client = MagicMock()
        mock_client.completion.return_value = "Mocked Response"
        mock_client.model_name = "openai/gpt-4.1-mini"
        mock_client.get_usage_summary.return_value = None
        mock_client.get_last_usage.return_value = SimpleNamespace(
            total_calls=1,
            total_input_tokens=120,
            total_output_tokens=45,
        )
        mock_get_client.return_value = mock_client

        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        response = handler.query("Test Prompt", "Test Context")

        self.assertEqual(response, "Mocked Response")
        mock_get_client.assert_called_once_with("openrouter", {})
        mock_rlm_class.assert_not_called()

        call_args = mock_client.completion.call_args.args
        self.assertEqual(len(call_args), 1)
        self.assertIn("Test Prompt", call_args[0])
        self.assertIn("Test Context", call_args[0])
        self.assertEqual(handler.last_metrics["mode"], "direct_lm")
        self.assertEqual(handler.last_metrics["total_input_tokens"], 120)
        self.assertEqual(handler.last_metrics["total_output_tokens"], 45)
        self.assertEqual(handler.last_metrics["total_tokens"], 165)
        self.assertFalse(handler.last_metrics["chunking"]["applied"])

    @patch("src.rlm_handler.get_client")
    def test_rlm_query_without_subagents_with_system_prompt(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.completion.return_value = "Mocked Response"
        mock_client.get_usage_summary.return_value = None
        mock_client.get_last_usage.return_value = SimpleNamespace(
            total_calls=1,
            total_input_tokens=10,
            total_output_tokens=4,
        )
        mock_get_client.return_value = mock_client

        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        handler.query("Prompt", "Context", system_prompt="System instructions")

        payload = mock_client.completion.call_args.args[0]
        self.assertEqual(payload[0]["role"], "system")
        self.assertEqual(payload[0]["content"], "System instructions")
        self.assertEqual(payload[1]["role"], "user")
        self.assertIn("Prompt", payload[1]["content"])

    @patch("src.rlm_handler.get_context_limit", return_value=2000)
    @patch("src.rlm_handler.get_client")
    def test_rlm_query_without_subagents_auto_chunks_large_context(
        self,
        mock_get_client,
        _mock_get_context_limit,
    ):
        mock_client = MagicMock()
        mock_client.model_name = "openai/gpt-4.1-mini"

        def _completion_side_effect(payload):
            text = payload if isinstance(payload, str) else payload[-1]["content"]
            if "Extract only information that is directly relevant" in text:
                return "chunk note"
            if "Compress these notes into a shorter list" in text:
                return "merged note"
            return "Final synthesized answer"

        mock_client.completion.side_effect = _completion_side_effect
        mock_client.get_usage_summary.return_value = SimpleNamespace(
            model_usage_summaries={
                "openai/gpt-4.1-mini": SimpleNamespace(
                    total_calls=5,
                    total_input_tokens=400,
                    total_output_tokens=120,
                )
            }
        )
        mock_get_client.return_value = mock_client

        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        large_context = "A" * 12000
        response = handler.query(
            "Summarize the content.",
            large_context,
            direct_chunk_overlap_tokens=0,
            direct_chunk_max_chunks=32,
            openrouter_middle_out_fallback=False,
        )

        self.assertEqual(response, "Final synthesized answer")
        self.assertTrue(handler.last_metrics["chunking"]["enabled"])
        self.assertTrue(handler.last_metrics["chunking"]["applied"])
        self.assertEqual(
            handler.last_metrics["chunking"]["strategy"],
            "map_reduce_chunking",
        )
        self.assertGreater(handler.last_metrics["chunking"]["chunks_processed"], 1)

    @patch("src.rlm_handler.get_context_limit", return_value=2000)
    @patch("src.rlm_handler.get_client")
    def test_rlm_query_without_subagents_uses_openrouter_middle_out_first(
        self,
        mock_get_client,
        _mock_get_context_limit,
    ):
        mock_client = MagicMock()
        mock_client.model_name = "openai/gpt-4.1-mini"
        mock_client.client = MagicMock()
        mock_client._track_cost = MagicMock()
        mock_client.get_usage_summary.return_value = SimpleNamespace(
            model_usage_summaries={
                "openai/gpt-4.1-mini": SimpleNamespace(
                    total_calls=1,
                    total_input_tokens=140,
                    total_output_tokens=30,
                )
            }
        )

        openrouter_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Middle-out answer"))],
            usage=SimpleNamespace(
                prompt_tokens=140,
                completion_tokens=30,
                total_tokens=170,
                model_extra={},
            ),
        )
        mock_client.client.chat.completions.create.return_value = openrouter_response
        mock_get_client.return_value = mock_client

        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        large_context = "B" * 12000
        response = handler.query(
            "Answer from context.",
            large_context,
            openrouter_middle_out_fallback=True,
        )

        self.assertEqual(response, "Middle-out answer")
        self.assertTrue(handler.last_metrics["chunking"]["middle_out_attempted"])
        self.assertTrue(handler.last_metrics["chunking"]["middle_out_applied"])
        self.assertEqual(
            handler.last_metrics["chunking"]["strategy"],
            "openrouter_middle_out",
        )
        self.assertTrue(handler.last_metrics["chunking"]["applied"])
        mock_client.client.chat.completions.create.assert_called_once()
        create_kwargs = mock_client.client.chat.completions.create.call_args.kwargs
        self.assertEqual(create_kwargs.get("extra_body"), {"transforms": ["middle-out"]})

    @patch("src.rlm_handler.rlm.RLM")
    def test_rlm_query_with_subagents(self, mock_rlm_class):
        mock_agent = MagicMock()
        completion = MagicMock()
        completion.response = "Mocked Recursive Response"
        completion.execution_time = 2.5
        completion.usage_summary = SimpleNamespace(
            model_usage_summaries={
                "openai/gpt-5-mini": SimpleNamespace(
                    total_calls=3,
                    total_input_tokens=220,
                    total_output_tokens=80,
                ),
                "gpt-4.1-mini": SimpleNamespace(
                    total_calls=2,
                    total_input_tokens=100,
                    total_output_tokens=40,
                ),
            }
        )
        mock_agent.completion.return_value = completion
        mock_rlm_class.return_value = mock_agent

        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        response = handler.query(
            "Prompt",
            "Context",
            model="openai/gpt-5-mini",
            use_subagents=True,
            system_prompt="SysPrompt",
            subagent_backend="openai",
            subagent_model="gpt-4.1-mini",
            max_iterations=12,
            max_subagent_calls=8,
        )

        self.assertEqual(response, "Mocked Recursive Response")
        mock_rlm_class.assert_called_once()

        rlm_init = mock_rlm_class.call_args.kwargs
        self.assertEqual(rlm_init.get("backend_kwargs", {}).get("model_name"), "openai/gpt-5-mini")
        self.assertEqual(rlm_init.get("max_depth"), 1)
        self.assertEqual(rlm_init.get("max_iterations"), 12)
        self.assertTrue(rlm_init.get("compaction"))
        self.assertEqual(rlm_init.get("compaction_threshold_pct"), 0.75)
        self.assertIn("SysPrompt", rlm_init.get("custom_system_prompt", ""))
        self.assertIn("Before FINAL, make at least one llm_query", rlm_init.get("custom_system_prompt", ""))
        self.assertEqual(rlm_init.get("other_backends"), ["openai"])
        self.assertEqual(
            rlm_init.get("other_backend_kwargs"),
            [{"model_name": "gpt-4.1-mini"}],
        )

        completion_call = mock_agent.completion.call_args.kwargs
        self.assertEqual(completion_call.get("prompt"), "Context")
        self.assertEqual(completion_call.get("root_prompt"), "Prompt")
        self.assertEqual(handler.last_metrics["mode"], "rlm_subagents")
        self.assertEqual(handler.last_metrics["configured_subagent_model"], "gpt-4.1-mini")
        self.assertEqual(handler.last_metrics["total_input_tokens"], 320)
        self.assertEqual(handler.last_metrics["total_output_tokens"], 120)
        self.assertEqual(handler.last_metrics["total_tokens"], 440)

    def test_guarded_logger_raises_on_subagent_budget_exceeded(self):
        logger = rlm_handler._GuardedRLMLogger(max_subagent_calls=1)
        iter_one = SimpleNamespace(
            code_blocks=[SimpleNamespace(result=SimpleNamespace(rlm_calls=["call-1"]))]
        )
        iter_two = SimpleNamespace(
            code_blocks=[SimpleNamespace(result=SimpleNamespace(rlm_calls=["call-2"]))]
        )

        logger.log(iter_one)
        self.assertEqual(logger.subagent_calls, 1)
        with self.assertRaises(RuntimeError):
            logger.log(iter_two)

    def test_inmemory_logger_clear_iterations_and_trajectory(self):
        logger = rlm_handler._InMemoryRLMLogger()
        metadata = SimpleNamespace(to_dict=lambda: {"root_model": "openai/gpt-5-mini"})
        iteration = SimpleNamespace(
            to_dict=lambda: {"response": "ok", "code_blocks": []},
            code_blocks=[],
        )

        logger.log_metadata(metadata)
        logger.log(iteration)

        trajectory = logger.get_trajectory()
        self.assertIsNotNone(trajectory)
        self.assertEqual(trajectory["run_metadata"]["root_model"], "openai/gpt-5-mini")
        self.assertEqual(len(trajectory["iterations"]), 1)
        self.assertEqual(logger.iteration_count, 1)

        logger.clear_iterations()
        self.assertEqual(logger.iteration_count, 0)
        self.assertEqual(logger.get_trajectory()["run_metadata"]["root_model"], "openai/gpt-5-mini")
        self.assertEqual(logger.get_trajectory()["iterations"], [])

    def test_guarded_logger_clear_iterations_resets_subagent_calls(self):
        logger = rlm_handler._GuardedRLMLogger(max_subagent_calls=5)
        iteration = SimpleNamespace(
            code_blocks=[SimpleNamespace(result=SimpleNamespace(rlm_calls=["call-1", "call-2"]))]
        )

        logger.log(iteration)
        self.assertEqual(logger.subagent_calls, 2)
        self.assertEqual(logger.iteration_count, 1)

        logger.clear_iterations()
        self.assertEqual(logger.subagent_calls, 0)
        self.assertEqual(logger.iteration_count, 0)

    @patch("src.rlm_handler.rlm.RLM")
    def test_rlm_query_reports_subagent_call_guard_violation(self, mock_rlm_class):
        mock_agent = MagicMock()
        mock_agent.completion.side_effect = RuntimeError(
            "Max subagent calls exceeded: 2 > 1"
        )
        mock_rlm_class.return_value = mock_agent

        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        response = handler.query(
            "Prompt",
            "Context",
            model="openai/gpt-5-mini",
            use_subagents=True,
            max_subagent_calls=1,
        )

        self.assertIn("Max subagent calls exceeded", response)
        self.assertTrue(handler.last_metrics.get("max_subagent_calls_exceeded"))
        self.assertEqual(handler.last_metrics.get("max_subagent_calls"), 1)

    def test_rlm_query_rejects_partial_subagent_config(self):
        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        response = handler.query(
            "Prompt",
            "Context",
            model="openai/gpt-5-mini",
            use_subagents=True,
            subagent_backend="openai",
            subagent_model=None,
        )
        self.assertIn("Both subagent_backend and subagent_model", response)

    def test_compose_rlm_system_prompt_keeps_core_instructions(self):
        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        prompt = handler._compose_rlm_system_prompt(
            user_system_prompt="Prioritize concise answers.",
            subagent_model="gpt-4.1-mini",
        )
        self.assertIn("You are tasked with answering a query with associated context.", prompt)
        self.assertIn("Prioritize concise answers.", prompt)
        self.assertIn("gpt-4.1-mini", prompt)
        self.assertIn("Before FINAL, make at least one llm_query", prompt)
        self.assertIn("Never call FINAL_VAR", prompt)

    def test_compose_rlm_system_prompt_without_alt_subagent_model(self):
        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        prompt = handler._compose_rlm_system_prompt(
            user_system_prompt=None,
            subagent_model=None,
        )
        self.assertIn(
            "llm_query calls use the root model by default when no alternate subagent backend/model is configured.",
            prompt,
        )
        self.assertIn("Before FINAL, make at least one llm_query", prompt)

    @patch("src.rlm_handler.rlm.RLM")
    def test_rlm_query_with_subagents_same_model_uses_default_routing(self, mock_rlm_class):
        mock_agent = MagicMock()
        completion = MagicMock()
        completion.response = "Recursive response on root model."
        completion.execution_time = 1.0
        completion.usage_summary = SimpleNamespace(model_usage_summaries={})
        mock_agent.completion.return_value = completion
        mock_rlm_class.return_value = mock_agent

        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        handler.query(
            "Prompt",
            "Context",
            model="openai/gpt-5-mini",
            use_subagents=True,
            subagent_backend=None,
            subagent_model=None,
        )

        rlm_init = mock_rlm_class.call_args.kwargs
        self.assertEqual(rlm_init.get("backend_kwargs", {}).get("model_name"), "openai/gpt-5-mini")
        self.assertNotIn("other_backends", rlm_init)
        self.assertIn(
            "llm_query calls use the root model by default",
            rlm_init.get("custom_system_prompt", ""),
        )
        self.assertIn(
            "Before FINAL, make at least one llm_query",
            rlm_init.get("custom_system_prompt", ""),
        )
        self.assertTrue(rlm_init.get("compaction"))
        self.assertEqual(rlm_init.get("compaction_threshold_pct"), 0.75)

    @patch("src.rlm_handler.rlm.RLM")
    def test_rlm_query_with_subagents_compaction_can_be_disabled(self, mock_rlm_class):
        mock_agent = MagicMock()
        completion = MagicMock()
        completion.response = "ok"
        completion.execution_time = 0.3
        completion.usage_summary = SimpleNamespace(model_usage_summaries={})
        mock_agent.completion.return_value = completion
        mock_rlm_class.return_value = mock_agent

        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        handler.query(
            "Prompt",
            "Context",
            model="openai/gpt-5-mini",
            use_subagents=True,
            subagent_root_compaction_enabled=False,
            subagent_compaction_threshold_pct=0.6,
        )

        rlm_init = mock_rlm_class.call_args.kwargs
        self.assertFalse(rlm_init.get("compaction"))
        self.assertEqual(rlm_init.get("compaction_threshold_pct"), 0.6)

    @patch("src.rlm_handler.get_client")
    @patch("src.rlm_handler.rlm.RLM")
    def test_rlm_query_recovers_from_final_var_error(self, mock_rlm_class, mock_get_client):
        bad_1 = MagicMock()
        bad_1.response = (
            "Error: Variable 'x' not found. Available variables: ['context']. "
            "You must create and assign a variable BEFORE calling FINAL_VAR on it."
        )
        bad_1.execution_time = 1.0
        bad_1.usage_summary = SimpleNamespace(model_usage_summaries={})

        bad_2 = MagicMock()
        bad_2.response = bad_1.response
        bad_2.execution_time = 1.2
        bad_2.usage_summary = SimpleNamespace(model_usage_summaries={})

        mock_agent = MagicMock()
        mock_agent.completion.side_effect = [bad_1, bad_2]
        mock_rlm_class.return_value = mock_agent

        fallback_client = MagicMock()
        fallback_client.model_name = "openai/gpt-5-mini"
        fallback_client.completion.return_value = "Fallback answer."
        fallback_client.get_last_usage.return_value = SimpleNamespace(
            total_calls=1,
            total_input_tokens=10,
            total_output_tokens=4,
        )
        mock_get_client.return_value = fallback_client

        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        response = handler.query(
            "Prompt",
            "Context",
            model="openai/gpt-5-mini",
            use_subagents=True,
            subagent_backend="openai",
            subagent_model="gpt-4.1-mini",
        )

        self.assertIn("[Recovered after RLM FINAL_VAR failure", response)
        self.assertIn("Fallback answer.", response)
        self.assertEqual(mock_agent.completion.call_count, 2)
        self.assertTrue(handler.last_metrics.get("retry_attempted"))
        self.assertTrue(handler.last_metrics.get("fallback_used"))
        self.assertTrue(handler.last_metrics.get("unresolved_final_var_error"))


if __name__ == "__main__":
    unittest.main()
