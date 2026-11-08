import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.env import load_credentials


class CredentialFileTests(unittest.TestCase):
    def test_allowlist_quotes_and_no_shell_execution_or_existing_env_override(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / ".env"
            path.write_text(
                'ALPACA_API_KEY="from-file"\n'
                'SEC_USER_AGENT="Researcher name research@example.com"\n'
                "IGNORED_COMMAND=$(touch /tmp/should-not-run)\n"
                'HF_TOKEN="$(never execute this)"\n'
            )
            with patch.dict(os.environ, {"ALPACA_API_KEY": "existing"}, clear=True):
                load_credentials(path)
                self.assertEqual(os.environ["ALPACA_API_KEY"], "existing")
                self.assertEqual(os.environ["HF_TOKEN"], "$(never execute this)")
                self.assertEqual(
                    os.environ["SEC_USER_AGENT"], "Researcher name research@example.com"
                )
                self.assertNotIn("IGNORED_COMMAND", os.environ)

    def test_blank_placeholders_are_skipped_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / ".env"
            path.write_text(
                "HF_TOKEN=\nALPHA_VANTAGE_API_KEY= # optional\n"
                'ALPACA_SECRET_KEY=""\nSEC_USER_AGENT="Name email@example.com"\n'
            )
            with patch.dict(os.environ, {"HF_TOKEN": "existing"}, clear=True):
                load_credentials(path)
                self.assertEqual(os.environ["HF_TOKEN"], "existing")
                self.assertNotIn("ALPHA_VANTAGE_API_KEY", os.environ)
                self.assertNotIn("ALPACA_SECRET_KEY", os.environ)
                self.assertEqual(os.environ["SEC_USER_AGENT"], "Name email@example.com")

    def test_invalid_quotes_or_unquoted_spaces_identify_key_without_exposing_value(self):
        for value in ("PRIVATE Name private@example.com", '"PRIVATE unterminated'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as root:
                path = Path(root) / ".env"
                path.write_text("# header\nSEC_USER_AGENT=" + value + "\n")
                with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError) as error:
                    load_credentials(path)
                self.assertIn("SEC_USER_AGENT", str(error.exception))
                self.assertIn("line 2", str(error.exception))
                self.assertNotIn("PRIVATE", str(error.exception))
                self.assertNotIn("private@example.com", str(error.exception))

    def test_example_template_can_be_loaded_as_data(self):
        with patch.dict(os.environ, {}, clear=True):
            load_credentials(Path(__file__).resolve().parents[1] / ".env.example")
            self.assertEqual(os.environ["SEC_USER_AGENT"], "Your Name your.email@example.com")
            self.assertNotIn("HF_TOKEN", os.environ)
