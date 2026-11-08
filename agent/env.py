"""Load only known credentials as data; never execute a shell environment file."""

import os
import shlex
from pathlib import Path

_ALLOWED = {
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "SEC_USER_AGENT",
    "HF_TOKEN",
}


def load_credentials(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("credential file is missing")
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("export "):
            raw = raw[7:].strip()
        key, separator, value = raw.partition("=")
        key = key.strip()
        if key not in _ALLOWED:
            continue
        if not separator:
            raise ValueError("invalid credential assignment")
        try:
            parsed = shlex.split(value, comments=True, posix=True)
        except ValueError:
            raise ValueError(f"invalid quoting for {key} on line {number}") from None
        # Empty placeholders are common for optional providers. Required values
        # are checked by the consumer that actually needs that provider.
        if not parsed or parsed == [""]:
            continue
        if len(parsed) != 1:
            raise ValueError(f"{key} on line {number}: values with spaces must be quoted")
        os.environ.setdefault(key, parsed[0])
