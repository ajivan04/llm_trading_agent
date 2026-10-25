import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from agent.data.cache import CacheIdentity, RawResponseArchive, SqliteResponseCache


def identity(**changes):
    values = {
        "source": "alpha_vantage_news",
        "symbols": ("MSFT", "AAPL", "AAPL"),
        "params": {"limit": 50, "sort": "LATEST"},
        "snapshot_window": "2026-10-25",
        "transformation_version": "news-v1",
    }
    values.update(changes)
    return CacheIdentity(**values)


class CacheIdentityTests(unittest.TestCase):
    def test_symbol_and_parameter_order_are_canonical(self):
        first = identity()
        second = identity(
            symbols=("AAPL", "MSFT"), params={"sort": "LATEST", "limit": 50})
        self.assertEqual(first.symbols, ("AAPL", "MSFT"))
        self.assertEqual(first.key, second.key)
        self.assertNotEqual(first.key, identity(params={"limit": 10}).key)

    def test_sensitive_parameters_are_rejected(self):
        for params in ({"apikey": "secret"}, {"headers": {"Authorization": "secret"}}):
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, "sensitive"):
                identity(params=params)

    def test_nested_parameters_are_deeply_frozen(self):
        params = {"filters": {"topics": ["earnings"]}}
        cache_identity = identity(params=params)
        original_key = cache_identity.key
        params["filters"]["topics"].append("mergers")
        self.assertEqual(cache_identity.key, original_key)
        with self.assertRaises(TypeError):
            cache_identity.params["filters"]["topics"] += ("rates",)


class ResponseCacheTests(unittest.TestCase):
    def test_ttl_boundaries_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = SqliteResponseCache(Path(directory) / "cache.sqlite")
            try:
                cache.put(identity(), {"feed": [1]}, fetched_at="2026-10-25T12:00:00Z",
                          ttl=timedelta(hours=24))
                self.assertIsNone(cache.get(identity(), now="2026-10-25T11:59:59Z"))
                self.assertEqual(cache.get(identity(), now="2026-10-26T11:59:59Z"),
                                 {"feed": [1]})
                self.assertIsNone(cache.get(identity(), now="2026-10-26T12:00:00Z"))
                cache.db.execute(
                    "UPDATE response_cache SET payload='{}' WHERE cache_key=?",
                    (identity().key,),
                )
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    cache.get(identity(), now="2026-10-25T13:00:00Z")
            finally:
                cache.close()

    def test_archive_is_idempotent_and_manifest_is_secret_free(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = RawResponseArchive(Path(directory))
            raw = b'{"feed":[]}'
            first = archive.put(
                raw, identity=identity(), retrieved_at="2026-10-25T12:00:00Z")
            second = archive.put(
                raw, identity=identity(), retrieved_at="2026-10-25T12:00:00Z")
            self.assertEqual(first, second)
            archive.put(
                raw, identity=identity(snapshot_window="2026-10-26"),
                retrieved_at="2026-10-26T12:00:00Z")
            manifests = sorted((Path(directory) / "records").glob("*.json"))
            self.assertEqual(len(manifests), 2)
            for manifest_path in manifests:
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["content_hash"], first)
                self.assertNotIn("params", manifest)


if __name__ == "__main__":
    unittest.main()
