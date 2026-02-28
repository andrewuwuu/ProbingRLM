import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.context_controls import context_controls_from_env


class TestContextControls(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_context_controls_defaults(self):
        controls = context_controls_from_env()
        self.assertTrue(controls.direct_chunking_enabled)
        self.assertEqual(controls.direct_chunk_overlap_tokens, 256)
        self.assertEqual(controls.direct_chunk_max_chunks, 64)
        self.assertTrue(controls.openrouter_middle_out_fallback)
        self.assertTrue(controls.subagent_root_compaction_enabled)
        self.assertAlmostEqual(controls.subagent_compaction_threshold_pct, 0.75)

    @patch.dict(
        os.environ,
        {
            "RLM_DIRECT_CHUNKING_ENABLED": "false",
            "RLM_DIRECT_CHUNK_OVERLAP_TOKENS": "0",
            "RLM_DIRECT_CHUNK_MAX_CHUNKS": "12",
            "RLM_OPENROUTER_MIDDLE_OUT_FALLBACK": "no",
            "RLM_SUBAGENT_ROOT_COMPACTION_ENABLED": "0",
            "RLM_SUBAGENT_COMPACTION_THRESHOLD_PCT": "0.65",
        },
        clear=True,
    )
    def test_context_controls_env_override(self):
        controls = context_controls_from_env()
        self.assertFalse(controls.direct_chunking_enabled)
        self.assertEqual(controls.direct_chunk_overlap_tokens, 0)
        self.assertEqual(controls.direct_chunk_max_chunks, 12)
        self.assertFalse(controls.openrouter_middle_out_fallback)
        self.assertFalse(controls.subagent_root_compaction_enabled)
        self.assertAlmostEqual(controls.subagent_compaction_threshold_pct, 0.65)


if __name__ == "__main__":
    unittest.main()
