import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main


class TestMainQueryPrompt(unittest.TestCase):
    @patch("builtins.input", return_value="What is the abstract?")
    def test_prompt_query_without_default_uses_user_text(self, mock_input):
        result = main._prompt_query(None)
        self.assertEqual(result, "What is the abstract?")
        mock_input.assert_called_once()

    @patch("main._prompt_yes_no", return_value=True)
    @patch("builtins.input")
    def test_prompt_query_with_default_accepts_default(self, mock_input, mock_yes_no):
        default_query = "Summarize the paper in 5 bullets."
        result = main._prompt_query(default_query)
        self.assertEqual(result, default_query)
        mock_yes_no.assert_called_once()
        mock_input.assert_not_called()

    @patch("main._prompt_yes_no", return_value=False)
    @patch("builtins.input", return_value="Find all stated limitations.")
    def test_prompt_query_with_default_accepts_custom_query(self, mock_input, mock_yes_no):
        default_query = "Summarize the paper in 5 bullets."
        result = main._prompt_query(default_query)
        self.assertEqual(result, "Find all stated limitations.")
        mock_yes_no.assert_called_once()
        mock_input.assert_called_once()

    @patch("main._prompt_yes_no", return_value=False)
    @patch("builtins.input", side_effect=["y", ""])
    def test_prompt_query_guard_yes_no_falls_back_to_default(self, mock_input, mock_yes_no):
        default_query = "Summarize the paper in 5 bullets."
        result = main._prompt_query(default_query)
        self.assertEqual(result, default_query)
        self.assertEqual(mock_input.call_count, 2)
        mock_yes_no.assert_called_once()

    def test_should_continue_session_when_followups_enabled(self):
        self.assertTrue(main._should_continue_session(True, 0))
        self.assertTrue(main._should_continue_session(True, 1))

    def test_should_continue_session_when_one_time_mode(self):
        self.assertTrue(main._should_continue_session(False, 0))
        self.assertFalse(main._should_continue_session(False, 1))


if __name__ == "__main__":
    unittest.main()
