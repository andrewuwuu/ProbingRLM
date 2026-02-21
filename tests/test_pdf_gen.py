import os
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.output_utils import save_markdown_response, save_pdf_response


class TestOutputUtils(unittest.TestCase):
    def test_markdown_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_output.md")
            save_markdown_response(output_file, "What is this?", "This is a test.")

            self.assertTrue(os.path.exists(output_file))
            with open(output_file, "r", encoding="utf-8") as handle:
                content = handle.read()
            self.assertIn("# Query: What is this?", content)
            self.assertIn("This is a test.", content)

    def test_pdf_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_output.pdf")
            save_pdf_response(output_file, "What is this?", "This is a test.")

            self.assertTrue(os.path.exists(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)


if __name__ == "__main__":
    unittest.main()
