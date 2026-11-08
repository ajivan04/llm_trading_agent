"""Small durable artifact helpers shared by hydration and frozen releases."""

import hashlib
import os
from pathlib import Path

from agent.contracts import canonical_json


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def check_files(root, hashes):
    root = Path(root).resolve()
    for name, expected in hashes.items():
        path = root / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError("unsafe artifact path")
        if file_hash(path) != expected:
            raise ValueError(f"artifact checksum differs: {name}")
