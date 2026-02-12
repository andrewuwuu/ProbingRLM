import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import pdf_utils
from src import rlm_handler


class _FakeLM:
    def __init__(self, model: str) -> None:
        self.model = model
        self.history: list[dict] = []
        self.last_messages = None

    def __call__(self, *, messages):
        self.last_messages = messages
        self.history.append(
            {
                "model": self.model,
                "usage": {"prompt_tokens": 120, "completion_tokens": 45},
            }
        )
        return ["Mocked Response"]


class TestPDFRetrieval(unittest.TestCase):
    def test_tracing_lm_proxy_is_baselm_and_forwards_messages_only(self):
        if rlm_handler.dspy is None:
            self.skipTest("dspy is not installed")

        class _CallOnlyLM:
            def __init__(self):
                self.model = "openrouter/openai/gpt-5-mini"
                self.model_type = "chat"
                self.temperature = 0.0
                self.max_tokens = 1000
                self.cache = True
                self.history = []
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                self.history.append(
                    {
                        "model": self.model,
                        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                    }
                )
                return ["ok"]

        raw_lm = _CallOnlyLM()
        proxy = rlm_handler._TracingLMProxy(raw_lm, "root:test")

        self.assertTrue(isinstance(proxy, rlm_handler.dspy.BaseLM))
        out = proxy(messages=[{"role": "user", "content": "hello"}])
        self.assertEqual(out, ["ok"])
        self.assertEqual(len(raw_lm.calls), 1)
        self.assertIn("messages", raw_lm.calls[0])
        self.assertNotIn("prompt", raw_lm.calls[0])

    def test_tracing_lm_proxy_forwards_prompt_only(self):
        if rlm_handler.dspy is None:
            self.skipTest("dspy is not installed")

        class _CallOnlyLM:
            def __init__(self):
                self.model = "openrouter/openai/gpt-5-mini"
                self.model_type = "chat"
                self.temperature = 0.0
                self.max_tokens = 1000
                self.cache = True
                self.history = []
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                self.history.append(
                    {
                        "model": self.model,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                    }
                )
                return ["ok"]

        raw_lm = _CallOnlyLM()
        proxy = rlm_handler._TracingLMProxy(raw_lm, "root:test")

        out = proxy(prompt="hi")
        self.assertEqual(out, ["ok"])
        self.assertEqual(len(raw_lm.calls), 1)
        self.assertIn("prompt", raw_lm.calls[0])
        self.assertNotIn("messages", raw_lm.calls[0])

    def test_list_documents_includes_supported_and_text_like_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = {
                "paper.pdf": b"%PDF-1.7\nfake\n",
                "notes.md": b"# Notes\nhello\n",
                "readme.txt": b"plain text\n",
                "config.json": b"{\"hello\": \"world\"}\n",
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
            self.assertIn("config.json", listed)
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

        self.assertEqual(text, "[Page 1]\nPage text\n[Page 2]\nPage text")
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

    @patch("src.pdf_utils._load_doc")
    def test_load_document_routes_doc(self, mock_load_doc):
        mock_load_doc.return_value = "doc text"
        result = pdf_utils.load_document("sample.doc")
        self.assertEqual(result, "doc text")
        mock_load_doc.assert_called_once_with("sample.doc")

    def test_load_document_reads_plain_text_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "sample.log")
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.write("hello from text file")

            result = pdf_utils.load_document(file_path)
            self.assertEqual(result, "hello from text file")

    def test_load_document_reads_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "sample.json")
            with open(file_path, "w", encoding="utf-8") as handle:
                handle.write("{\"name\": \"test\", \"ok\": true}")

            result = pdf_utils.load_document(file_path)
            self.assertIn("\"name\": \"test\"", result)

    @patch.object(rlm_handler, "dspy", new=SimpleNamespace(LM=object))
    @patch.object(rlm_handler.RLMHandler, "_build_lm")
    def test_query_without_subagents_uses_direct_lm(self, mock_build_lm):
        lm = _FakeLM("openai/gpt-4.1-mini")
        mock_build_lm.return_value = lm

        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        response = handler.query(
            "Test Prompt",
            "Test Context",
            system_prompt="System instructions",
            custom_prompt="Custom instructions",
        )

        self.assertEqual(response, "Mocked Response")
        self.assertIn("Test Prompt", lm.last_messages[1]["content"])
        self.assertIn("Test Context", lm.last_messages[1]["content"])
        self.assertEqual(lm.last_messages[0]["role"], "system")
        self.assertIn("System instructions", lm.last_messages[0]["content"])
        self.assertIn("Custom instructions", lm.last_messages[0]["content"])
        self.assertEqual(handler.last_metrics["mode"], "direct_lm")
        self.assertEqual(handler.last_metrics["total_input_tokens"], 120)
        self.assertEqual(handler.last_metrics["total_output_tokens"], 45)
        self.assertEqual(handler.last_metrics["total_tokens"], 165)

    @patch.object(rlm_handler.RLMHandler, "_build_signature")
    @patch.object(rlm_handler.RLMHandler, "_build_lm")
    def test_query_with_subagents_uses_dspy_rlm(
        self,
        mock_build_lm,
        mock_build_signature,
    ):
        root_lm = SimpleNamespace(model="openai/gpt-5-mini", history=[])
        sub_lm = SimpleNamespace(model="gpt-4.1-mini", history=[])

        mock_build_lm.side_effect = [root_lm, sub_lm]
        fake_signature = SimpleNamespace(
            input_fields={"context": object(), "query": object(), "guidance": object()}
        )
        mock_build_signature.return_value = fake_signature

        def fake_agent_call(*, context, query, guidance):
            self.assertEqual(context, "Context")
            self.assertEqual(query, "Prompt")
            self.assertIn("SysPrompt", guidance)
            self.assertIn("Depth policy", guidance)
            root_lm.history.append(
                {
                    "model": root_lm.model,
                    "usage": {"prompt_tokens": 220, "completion_tokens": 80},
                }
            )
            sub_lm.history.append(
                {
                    "model": sub_lm.model,
                    "usage": {"prompt_tokens": 100, "completion_tokens": 40},
                }
            )
            return SimpleNamespace(answer="Mocked Recursive Response", trajectory=["i1", "i2"])

        mock_agent = MagicMock(side_effect=fake_agent_call)
        mock_rlm_ctor = MagicMock(return_value=mock_agent)
        fake_dspy = SimpleNamespace(
            context=lambda **_: nullcontext(),
            RLM=mock_rlm_ctor,
        )

        with patch.object(rlm_handler, "dspy", new=fake_dspy):
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
                max_llm_calls=8,
                max_output_chars=5000,
                max_depth=3,
                rlm_signature="context, query, instructions -> answer",
            )

        self.assertEqual(response, "Mocked Recursive Response")
        mock_rlm_ctor.assert_called_once()
        rlm_init = mock_rlm_ctor.call_args.kwargs
        self.assertEqual(rlm_init.get("max_iterations"), 12)
        self.assertEqual(rlm_init.get("max_llm_calls"), 8)
        self.assertEqual(rlm_init.get("max_output_chars"), 5000)
        self.assertEqual(rlm_init.get("sub_lm"), sub_lm)
        self.assertEqual(rlm_init.get("signature"), fake_signature)

        self.assertEqual(handler.last_metrics["mode"], "dspy_rlm")
        self.assertEqual(handler.last_metrics["configured_subagent_model"], "gpt-4.1-mini")
        self.assertEqual(handler.last_metrics["subagent_calls"], 1)
        self.assertEqual(handler.last_metrics["iterations"], 2)
        self.assertEqual(handler.last_metrics["total_input_tokens"], 320)
        self.assertEqual(handler.last_metrics["total_output_tokens"], 120)
        self.assertEqual(handler.last_metrics["total_tokens"], 440)

    @patch.object(rlm_handler.RLMHandler, "_build_signature")
    @patch.object(rlm_handler.RLMHandler, "_build_lm")
    def test_query_with_subagent_prefetch_makes_sub_calls(
        self,
        mock_build_lm,
        mock_build_signature,
    ):
        class _FakeSubLM:
            def __init__(self):
                self.model = "openrouter/stepfun/step-3.5-flash:free"
                self.history = []

            def __call__(self, *, messages):
                self.history.append(
                    {
                        "model": self.model,
                        "usage": {"prompt_tokens": 40, "completion_tokens": 20},
                    }
                )
                return ["prefetch evidence"]

        root_lm = SimpleNamespace(model="openrouter/arcee-ai/trinity-large-preview:free", history=[])
        sub_lm = _FakeSubLM()

        mock_build_lm.side_effect = [root_lm, sub_lm]
        fake_signature = SimpleNamespace(
            input_fields={"context": object(), "query": object(), "guidance": object()}
        )
        mock_build_signature.return_value = fake_signature

        def fake_agent_call(*, context, query, guidance):
            root_lm.history.append(
                {
                    "model": root_lm.model,
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30},
                }
            )
            self.assertIn("Subagent pre-analysis", guidance)
            return SimpleNamespace(answer="ok", trajectory=["i1"])

        mock_rlm_ctor = MagicMock(return_value=MagicMock(side_effect=fake_agent_call))
        fake_dspy = SimpleNamespace(
            context=lambda **_: nullcontext(),
            RLM=mock_rlm_ctor,
        )

        with patch.object(rlm_handler, "dspy", new=fake_dspy):
            handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
            response = handler.query(
                "Prompt",
                "Context",
                model="arcee-ai/trinity-large-preview:free",
                use_subagents=True,
                subagent_backend="openrouter",
                subagent_model="stepfun/step-3.5-flash:free",
                subagent_prefetch_calls=2,
            )

        self.assertEqual(response, "ok")
        self.assertEqual(handler.last_metrics["subagent_calls"], 2)

    @patch.object(rlm_handler.RLMHandler, "_build_signature")
    @patch.object(rlm_handler.RLMHandler, "_build_lm")
    def test_query_with_strict_subagent_call_fails_when_no_subagent_calls(
        self,
        mock_build_lm,
        mock_build_signature,
    ):
        root_lm = SimpleNamespace(model="openrouter/openai/gpt-5-mini", history=[])
        sub_lm = SimpleNamespace(model="openrouter/stepfun/step-3.5-flash:free", history=[])

        mock_build_lm.side_effect = [root_lm, sub_lm]
        fake_signature = SimpleNamespace(
            input_fields={"context": object(), "query": object(), "guidance": object()}
        )
        mock_build_signature.return_value = fake_signature

        def fake_agent_call(*, context, query, guidance):
            self.assertEqual(context, "Context")
            self.assertEqual(query, "Prompt")
            self.assertIn("Depth policy", guidance)
            root_lm.history.append(
                {
                    "model": root_lm.model,
                    "usage": {"prompt_tokens": 180, "completion_tokens": 60},
                }
            )
            return SimpleNamespace(answer="Should fail strict mode", trajectory=["i1"])

        mock_agent = MagicMock(side_effect=fake_agent_call)
        mock_rlm_ctor = MagicMock(return_value=mock_agent)
        fake_dspy = SimpleNamespace(
            context=lambda **_: nullcontext(),
            RLM=mock_rlm_ctor,
        )

        with patch.object(rlm_handler, "dspy", new=fake_dspy):
            handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
            response = handler.query(
                "Prompt",
                "Context",
                model="openai/gpt-5-mini",
                use_subagents=True,
                system_prompt="SysPrompt",
                subagent_backend="openrouter",
                subagent_model="stepfun/step-3.5-flash:free",
                max_iterations=6,
                max_llm_calls=4,
                max_output_chars=4000,
                max_depth=2,
                require_subagent_call=True,
            )

        self.assertIn("strict subagent-call mode is enabled", response)
        self.assertEqual(handler.last_metrics["mode"], "dspy_rlm")
        self.assertEqual(handler.last_metrics["subagent_calls"], 0)
        self.assertEqual(
            handler.last_metrics["error"],
            "required_subagent_call_missing",
        )

    @patch.object(rlm_handler.RLMHandler, "_build_signature")
    @patch.object(rlm_handler.RLMHandler, "_build_lm")
    def test_query_builds_explicit_sub_lm_even_without_dedicated_subagent_config(
        self,
        mock_build_lm,
        mock_build_signature,
    ):
        root_lm = SimpleNamespace(model="openrouter/openai/gpt-5-mini", history=[])
        sub_lm = SimpleNamespace(model="openrouter/openai/gpt-5-mini", history=[])

        mock_build_lm.side_effect = [root_lm, sub_lm]
        fake_signature = SimpleNamespace(
            input_fields={"context": object(), "query": object(), "guidance": object()}
        )
        mock_build_signature.return_value = fake_signature

        def fake_agent_call(*, context, query, guidance):
            root_lm.history.append(
                {
                    "model": root_lm.model,
                    "usage": {"prompt_tokens": 80, "completion_tokens": 20},
                }
            )
            sub_lm.history.append(
                {
                    "model": sub_lm.model,
                    "usage": {"prompt_tokens": 60, "completion_tokens": 10},
                }
            )
            return SimpleNamespace(answer="ok", trajectory=["i1"])

        mock_rlm_ctor = MagicMock(return_value=MagicMock(side_effect=fake_agent_call))
        fake_dspy = SimpleNamespace(
            context=lambda **_: nullcontext(),
            RLM=mock_rlm_ctor,
        )

        with patch.object(rlm_handler, "dspy", new=fake_dspy):
            handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
            response = handler.query(
                "Prompt",
                "Context",
                model="openai/gpt-5-mini",
                use_subagents=True,
            )

        self.assertEqual(response, "ok")
        self.assertEqual(mock_build_lm.call_count, 2)
        sub_call = mock_build_lm.call_args_list[1]
        self.assertEqual(sub_call.kwargs["backend"], "openrouter")
        self.assertEqual(sub_call.kwargs["model"], "openai/gpt-5-mini")
        self.assertEqual(sub_call.kwargs["explicit_api_key"], "sk-test")
        self.assertEqual(handler.last_metrics["configured_subagent_model"], sub_lm.model)

    def test_query_rejects_partial_subagent_config(self):
        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        with self.assertRaises(ValueError):
            handler.query(
                "Prompt",
                "Context",
                model="openai/gpt-5-mini",
                use_subagents=True,
                subagent_backend="openai",
                subagent_model=None,
            )

    def test_compose_rlm_guidance_includes_user_custom_and_depth(self):
        prompt = rlm_handler.RLMHandler._compose_rlm_guidance(
            system_prompt="Prioritize concise answers.",
            custom_prompt="Use bullet points.",
            max_depth=4,
            subagent_model="gpt-4.1-mini",
        )
        self.assertIn("Prioritize concise answers.", prompt)
        self.assertIn("Use bullet points.", prompt)
        self.assertIn("gpt-4.1-mini", prompt)
        self.assertIn("Depth policy", prompt)

    def test_normalize_rlm_signature_rewrites_instructions_field(self):
        signature = rlm_handler.RLMHandler._normalize_rlm_signature(
            "context, query, instructions -> answer"
        )
        self.assertEqual(signature, "context, query, guidance -> answer")

    def test_resolve_model_name_supports_openrouter_free_aliases(self):
        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        self.assertEqual(
            handler._resolve_model_name("openrouter", "openrouter/free"),
            "openrouter/free",
        )
        self.assertEqual(
            handler._resolve_model_name("openrouter", "free"),
            "openrouter/free",
        )
        self.assertEqual(
            handler._resolve_model_name("openrouter", "arcee-ai/trinity-large-preview:free"),
            "openrouter/arcee-ai/trinity-large-preview:free",
        )
        self.assertEqual(
            handler._resolve_model_name("openrouter", "gpt-4.1-mini"),
            "openrouter/openai/gpt-4.1-mini",
        )

    def test_adapt_model_for_litellm_backend_handles_openrouter_router_ids(self):
        self.assertEqual(
            rlm_handler.RLMHandler._adapt_model_for_litellm_backend(
                "openrouter",
                "openrouter/free",
            ),
            "openrouter/openrouter/free",
        )
        self.assertEqual(
            rlm_handler.RLMHandler._adapt_model_for_litellm_backend(
                "openrouter",
                "openrouter/stepfun/step-3.5-flash:free",
            ),
            "openrouter/stepfun/step-3.5-flash:free",
        )
        self.assertEqual(
            rlm_handler.RLMHandler._adapt_model_for_litellm_backend(
                "openai",
                "openai/gpt-4.1-mini",
            ),
            "openai/gpt-4.1-mini",
        )

    @patch.object(rlm_handler, "dspy", new=None)
    def test_query_reports_missing_dspy_dependency(self):
        handler = rlm_handler.RLMHandler(backend="openrouter", api_key="sk-test")
        response = handler.query("Prompt", "Context")
        self.assertIn("dspy is not installed", response.lower())


if __name__ == "__main__":
    unittest.main()
