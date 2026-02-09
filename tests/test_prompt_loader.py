import unittest
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prompt_loader import load_prompts

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

if __name__ == '__main__':
    unittest.main()
