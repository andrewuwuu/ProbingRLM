import unittest
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prompt_loader import (
    extract_template_variables,
    load_prompts,
    parse_prompt_variables,
    render_template,
)

class TestPromptLoader(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_prompts.md"
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write("## System\nSystem prompt content.\n\n# Query\nUser query content.")

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_load_prompts(self):
        prompts = load_prompts(self.test_file)
        self.assertIn("System", prompts)
        self.assertIn("Query", prompts)
        self.assertEqual(prompts["System"], "System prompt content.")
        self.assertEqual(prompts["Query"], "User query content.")

    def test_load_nonexistent_file(self):
        prompts = load_prompts("nonexistent.md")
        self.assertEqual(prompts, {})

    def test_extract_template_variables(self):
        vars_found = extract_template_variables("Q={{question}} scope={{scope}} {{question}}")
        self.assertEqual(vars_found, ["question", "scope"])

    def test_render_template(self):
        rendered = render_template(
            "Question: {{question}} | Scope: {{scope}} | Missing: {{missing}}",
            {"question": "What is TCP?", "scope": "RFC"},
        )
        self.assertEqual(
            rendered,
            "Question: What is TCP? | Scope: RFC | Missing: {{missing}}",
        )

    def test_parse_prompt_variables(self):
        parsed = parse_prompt_variables(
            "# comment\nquestion=What is TCP?\nscope=protocol behavior\ninvalid_line\n"
        )
        self.assertEqual(
            parsed,
            {"question": "What is TCP?", "scope": "protocol behavior"},
        )

if __name__ == '__main__':
    unittest.main()
