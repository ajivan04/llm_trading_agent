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
