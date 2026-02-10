import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import cli_app


class TestCliBackends(unittest.TestCase):
    @patch.dict(os.environ, {"RLM_MAX_ITERATIONS": "12"}, clear=True)
    def test_parse_positive_int_env_returns_value(self):
        self.assertEqual(cli_app._parse_positive_int_env("RLM_MAX_ITERATIONS"), 12)

    @patch.dict(os.environ, {"RLM_MAX_ITERATIONS": "abc"}, clear=True)
    def test_parse_positive_int_env_rejects_invalid(self):
        self.assertIsNone(cli_app._parse_positive_int_env("RLM_MAX_ITERATIONS"))

    @patch.dict(os.environ, {"RLM_BACKEND": "unknown"}, clear=True)
    def test_resolve_backend_rejects_unsupported_backend(self):
        with self.assertRaises(ValueError):
            cli_app._resolve_backend_and_key()

    @patch.dict(
        os.environ,
        {
            "RLM_BACKEND": "openrouter",
            "OPENROUTER_API_KEY": "or-key",
        },
        clear=True,
    )
    def test_resolve_backend_with_configured_openrouter(self):
        backend, api_key = cli_app._resolve_backend_and_key()
        self.assertEqual(backend, "openrouter")
        self.assertEqual(api_key, "or-key")

    @patch.dict(
        os.environ,
        {
            "RLM_BACKEND": "anthropic",
        },
        clear=True,
    )
    def test_resolve_backend_requires_anthropic_key(self):
        with self.assertRaises(ValueError):
            cli_app._resolve_backend_and_key()

    @patch.dict(
        os.environ,
        {
            "RLM_BACKEND": "azure_openai",
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


if __name__ == "__main__":
    unittest.main()
