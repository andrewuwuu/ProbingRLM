import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.context_selector import build_query_context


class TestContextSelector(unittest.TestCase):
    def test_build_query_context_returns_full_text_when_under_budget(self):
        docs = [("a.txt", "alpha\n\nbeta"), ("b.txt", "gamma")]
        context, stats = build_query_context(
            documents=docs,
            query="alpha",
            max_chars=10_000,
            chunk_chars=1000,
            max_chunks=10,
        )
        self.assertIn("===== BEGIN FILE: a.txt =====", context)
        self.assertIn("===== END FILE: b.txt =====", context)
        self.assertEqual(stats["truncated"], 0)
        self.assertEqual(stats["selected_chunks"], 0)

    def test_build_query_context_raises_for_invalid_limits(self):
        with self.assertRaises(ValueError):
            build_query_context(
                documents=[("a.txt", "text")],
                query="q",
                max_chars=0,
                chunk_chars=100,
                max_chunks=10,
            )
        with self.assertRaises(ValueError):
            build_query_context(
                documents=[("a.txt", "text")],
                query="q",
                max_chars=100,
                chunk_chars=0,
                max_chunks=10,
            )
        with self.assertRaises(ValueError):
            build_query_context(
                documents=[("a.txt", "text")],
                query="q",
                max_chars=100,
                chunk_chars=100,
                max_chunks=0,
            )

    def test_build_query_context_selects_relevant_chunks_when_truncated(self):
        focused = "TCP congestion window grows quickly in slow start."
        filler = "lorem ipsum " * 200
        docs = [
            ("net.txt", f"{filler}\n\n{focused}\n\n{filler}"),
            ("other.txt", filler),
        ]
        context, stats = build_query_context(
            documents=docs,
            query="slow start congestion window",
            max_chars=1500,
            chunk_chars=400,
            max_chunks=3,
        )
        self.assertEqual(stats["truncated"], 1)
        self.assertGreater(stats["selected_chunks"], 0)
        self.assertIn("slow start", context.lower())


if __name__ == "__main__":
    unittest.main()
