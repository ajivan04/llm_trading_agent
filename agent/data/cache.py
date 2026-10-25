"""Canonical response caching and content-addressed raw archives."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from agent.contracts import (
    FrozenDict, canonical_json, content_hash, normalize_timestamp, parse_timestamp,
)


_SENSITIVE_PARTS = ("api_key", "apikey", "token", "secret", "password", "authorization")


def _reject_sensitive(value: Any, path: str = "params") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).casefold().replace("-", "_")
            if any(part in name for part in _SENSITIVE_PARTS):
                raise ValueError(f"sensitive field is not permitted in cache identity: {path}.{key}")
            _reject_sensitive(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_sensitive(item, f"{path}[{index}]")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    # Validate leaves and reject NaN/non-JSON objects through canonical serialization.
    canonical_json(value)
    return value


@dataclass(frozen=True)
class CacheIdentity:
    source: str
    symbols: tuple[str, ...]
    params: Mapping[str, Any]
    snapshot_window: str
    transformation_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be nonempty")
        if not isinstance(self.snapshot_window, str) or not self.snapshot_window.strip():
            raise ValueError("snapshot_window must be nonempty")
        if not isinstance(self.transformation_version, str) or not self.transformation_version.strip():
            raise ValueError("transformation_version must be nonempty")
        if not isinstance(self.params, Mapping):
            raise TypeError("params must be a mapping")
        _reject_sensitive(self.params)
        frozen_params = _freeze_json(self.params)
        normalized = tuple(sorted({symbol.strip().upper() for symbol in self.symbols}))
        if not normalized or any(not symbol for symbol in normalized):
            raise ValueError("symbols must contain at least one nonempty symbol")
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(self, "symbols", normalized)
        object.__setattr__(self, "params", frozen_params)
        object.__setattr__(self, "snapshot_window", self.snapshot_window.strip())
        object.__setattr__(self, "transformation_version", self.transformation_version.strip())

    @property
    def key(self) -> str:
        return content_hash({
            "source": self.source,
            "symbols": self.symbols,
            "params": self.params,
            "snapshot_window": self.snapshot_window,
            "transformation_version": self.transformation_version,
        })


class SqliteResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS response_cache (
               cache_key TEXT PRIMARY KEY,
               fetched_at TEXT NOT NULL,
               expires_at TEXT NOT NULL,
               payload TEXT NOT NULL,
               payload_hash TEXT NOT NULL)"""
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def put(
        self, identity: CacheIdentity, payload: Any, *,
        fetched_at: str | datetime, ttl: timedelta,
    ) -> str:
        if not isinstance(ttl, timedelta) or ttl <= timedelta(0):
            raise ValueError("ttl must be positive")
        fetched = parse_timestamp(fetched_at)
        serialized = canonical_json(payload)
        payload_hash = content_hash(payload)
        with self.db:
            self.db.execute(
                """INSERT OR REPLACE INTO response_cache
                   (cache_key,fetched_at,expires_at,payload,payload_hash)
                   VALUES(?,?,?,?,?)""",
                (identity.key, normalize_timestamp(fetched),
                 normalize_timestamp(fetched + ttl), serialized, payload_hash),
            )
        return payload_hash

    def get(self, identity: CacheIdentity, *, now: str | datetime) -> Any | None:
        current = parse_timestamp(now)
        row = self.db.execute(
            "SELECT * FROM response_cache WHERE cache_key=?", (identity.key,)
        ).fetchone()
        if row is None:
            return None
        fetched = parse_timestamp(row["fetched_at"])
        expires = parse_timestamp(row["expires_at"])
        if current < fetched or current >= expires:
            return None
        payload = json.loads(row["payload"])
        if content_hash(payload) != row["payload_hash"]:
            raise ValueError("cached payload hash mismatch")
        return payload


class RawResponseArchive:
    """Write immutable response bytes and a secret-free sidecar manifest."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def put(
        self, raw: bytes, *, identity: CacheIdentity,
        retrieved_at: str | datetime, media_type: str = "application/json",
    ) -> str:
        if not isinstance(raw, bytes):
            raise TypeError("raw response must be bytes")
        if not raw:
            raise ValueError("raw response must not be empty")
        retrieved = normalize_timestamp(retrieved_at)
        digest = content_hash({"bytes_hex": raw.hex()})
        directory = self.root / "objects" / digest[:2]
        directory.mkdir(parents=True, exist_ok=True)
        data_path = directory / f"{digest}.bin"
        records_dir = self.root / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "content_hash": digest,
            "cache_key": identity.key,
            "source": identity.source,
            "symbols": identity.symbols,
            "snapshot_window": identity.snapshot_window,
            "transformation_version": identity.transformation_version,
            "retrieved_at": retrieved,
            "media_type": media_type,
            "size": len(raw),
        }
        encoded_manifest = canonical_json(manifest)
        record_id = content_hash(manifest)
        manifest_path = records_dir / f"{record_id}.json"
        if data_path.exists() and data_path.read_bytes() != raw:
            raise ValueError("archive hash collision")
        if manifest_path.exists() and manifest_path.read_text() != encoded_manifest:
            raise ValueError("archive record hash collision")
        if not data_path.exists():
            data_path.write_bytes(raw)
        if not manifest_path.exists():
            manifest_path.write_text(encoded_manifest)
        return digest

