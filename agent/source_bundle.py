"""Market data sources, independent of any training labels or target files."""

import json
from pathlib import Path

from agent.artifacts import atomic_json, check_files, file_hash
from agent.contracts import content_hash, normalize_timestamp, parse_timestamp

_DATA_FILES = ("price-bars.jsonl", "evidence.jsonl")
_BOUNDARIES = ("train_end", "validation_end", "test_end")
_MARKERS = ("hydration-manifest.json", "price-basis-manifest.json", "compaction-manifest.json")


def _boundaries(values):
    # Validate ISO instants but preserve their spelling for compatibility with
    # existing training manifests and immutable optimizer contracts.
    result = {key: values[key] for key in _BOUNDARIES}
    for value in result.values():
        normalize_timestamp(value)
    if not (
        parse_timestamp(result["train_end"])
        < parse_timestamp(result["validation_end"])
        < parse_timestamp(result["test_end"])
    ):
        raise ValueError("source split boundaries must be strictly chronological")
    return result


def load_source(root):
    """Read a minimal bundle or an old archive in place; never load old targets."""
    root = Path(root)
    canonical = root / "source.json"
    compaction = None
    if canonical.is_file():
        saved = json.loads(canonical.read_text())
        manifest = saved["manifest"]
        if (
            manifest.get("version") != "research-source-v1"
            or saved.get("source_id") != content_hash(manifest)
            or manifest.get("price_basis") != "nominal"
            or manifest.get("scope") not in {"research", "heldout"}
            or set(manifest.get("files", {})) != set(_DATA_FILES)
            or ((manifest.get("scope") == "heldout") != bool(manifest.get("test_release_id")))
        ):
            raise ValueError("source manifest identity or format is invalid")
        universe = manifest.get("universe", [])
        if (
            not isinstance(universe, list)
            or not universe
            or len(set(universe)) != len(universe)
            or any(not isinstance(s, str) or not s or s != s.strip().upper() for s in universe)
        ):
            raise ValueError("source universe must contain unique normalized symbols")
        check_files(root, manifest["files"])
        result = {
            **_boundaries(manifest["boundaries"]),
            "universe": tuple(universe),
            "scope": manifest["scope"],
            "source_id": saved["source_id"],
            "dataset_hash": saved["source_id"],
            "price_hash": None,
            "evidence_count": None,
            "test_release_id": manifest.get("test_release_id"),
        }
        hashes = {**manifest["files"], "source.json": file_hash(canonical)}
    else:
        # Legacy split dates live in the dataset manifest, but its examples and
        # split targets are neither needed nor trusted as research source data.
        target_manifest = json.loads((root / "dataset/manifest.json").read_text())
        prep = json.loads((root / "preparation-manifest.json").read_text())
        hashes = {
            name: file_hash(root / name)
            for name in (*_DATA_FILES, "dataset/manifest.json", "preparation-manifest.json")
        }
        if prep.get("dataset_hash") != target_manifest.get("dataset_hash"):
            path = root / "compaction-manifest.json"
            if not path.is_file():
                raise ValueError("prepared dataset manifest differs without compaction lineage")
            row = json.loads(path.read_text())
            if (
                row.get("dataset_hash") != target_manifest.get("dataset_hash")
                or row.get("compaction_version") != "evidence-compaction-v2"
            ):
                raise ValueError("prepared dataset manifest differs from compaction lineage")
            hashes[path.name] = file_hash(path)
            linked = row.get("source_dataset_hash") == prep.get("dataset_hash")
            compaction = {
                "version": row["compaction_version"],
                "source_target_link_verified": linked,
                "preserved_price_hash_verified": True,
                "preserved_evidence_count_verified": True,
                "legacy_targets_consumed": False,
                "legacy_compaction_source_link_unverified": not linked,
            }
        price_hash = prep.get("source_hash")
        count = prep.get("counts", {}).get("evidence")
        if (
            not isinstance(price_hash, str)
            or len(price_hash) != 64
            or any(char not in "0123456789abcdef" for char in price_hash)
            or type(count) is not int
            or count < 0
            or not isinstance(target_manifest.get("dataset_hash"), str)
            or not target_manifest["dataset_hash"]
        ):
            raise ValueError(
                "legacy source requires its price hash, evidence count and dataset identity"
            )
        heldout = (
            prep.get("heldout_news_requested")
            or prep.get("archive_kind") == "frozen-heldout-source"
        )
        result = {
            **_boundaries(target_manifest),
            "universe": (),
            "scope": "heldout" if heldout else "research",
            "dataset_hash": target_manifest.get("dataset_hash"),
            "price_hash": prep.get("source_hash"),
            "evidence_count": prep.get("counts", {}).get("evidence"),
            "test_release_id": prep.get("test_release_id"),
            "source_id": content_hash(hashes),
        }
    for name in _MARKERS:
        path = root / name
        if path.is_file():
            hashes[name] = file_hash(path)
            if name == "hydration-manifest.json":
                hydration = json.loads(path.read_text())
                if hydration.get("heldout_news_requested"):
                    if canonical.is_file() and result["scope"] != "heldout":
                        raise ValueError("source scope differs from held-out hydration marker")
                    result["scope"] = "heldout"
                    if result.get("test_release_id") is not None and result[
                        "test_release_id"
                    ] != hydration.get("test_release_id"):
                        raise ValueError("source and hydration release identities differ")
                    result["test_release_id"] = hydration.get("test_release_id")
    return {**result, "file_hashes": hashes, "compaction_provenance": compaction}


def write_source(
    root, metadata, *, universe, scope=None, parent_source_id=None, test_release_id=None
):
    """Write source-only metadata after the two immutable data files exist."""
    root = Path(root)
    if metadata.get("scope") == "heldout" and scope == "research":
        raise ValueError("held-out sources cannot be relabeled for research")
    manifest = {
        "version": "research-source-v1",
        "boundaries": _boundaries(metadata),
        "universe": list(universe),
        "price_basis": "nominal",
        "files": {name: file_hash(root / name) for name in _DATA_FILES},
        "scope": scope or metadata.get("scope", "research"),
    }
    if parent_source_id is not None:
        manifest["parent_source_id"] = parent_source_id
    release = test_release_id or metadata.get("test_release_id")
    if release is not None:
        manifest["test_release_id"] = release
    if manifest["scope"] == "heldout" and not release:
        raise ValueError("held-out source requires a frozen release identity")
    if manifest["scope"] == "research" and release:
        raise ValueError("research source cannot carry a held-out release identity")
    atomic_json(root / "source.json", {"source_id": content_hash(manifest), "manifest": manifest})
    return load_source(root)
