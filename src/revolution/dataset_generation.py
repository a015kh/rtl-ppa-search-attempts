#!/usr/bin/env python3
"""Assemble the historical DC/PT PPA dataset from pre-measured JSONL inputs.

Deduplication, lineage splitting, report formatting, and label semantics are
preserved from the retained dataset builder. This module does not collect RTL
or launch synthesis. Input data is supplied separately by the user.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import dataset_row_helpers as resynth


PPA_FIELDS = (
    "SDC",
    "has_clock",
    "clock",
    "area_value",
    "delay_value",
    "wns",
    "tns",
    "total_power_value",
    "static_power_value",
    "switching_power_value",
    "synth",
    "area",
    "delay",
    "static_power",
    "switching_power",
    "synthesis_time",
    "design_name",
    "rtl",
)

PROVENANCE_FIELDS = (
    "record_id",
    "record_kind",
    "data_origin",
    "source_dataset",
    "source_revision",
    "source_split",
    "source_row_index",
    "source_match_method",
    "source_artifact",
    "source_artifact_sha256",
    "module_id",
    "candidate_id",
    "seed_sample_id",
    "generation_model",
    "optimization_direction",
    "generation_index",
    "candidate_index",
    "lineage_id",
    "seed_present_in_base",
)

ALL_FIELDS = PPA_FIELDS + PROVENANCE_FIELDS
NUMERIC_REQUIRED = (
    "area_value",
    "delay_value",
    "total_power_value",
    "static_power_value",
    "switching_power_value",
)
TRUE_TEST_KIND = "test"
AUGMENTATION_RE = re.compile(
    r"(?P<sample>sample_(?P<sample_index>\d{6})).*?"
    r"(?P<direction>area|delay|power)_gen(?P<generation>\d+)_cand(?P<candidate>\d+)",
    re.IGNORECASE,
)


class UnifiedDatasetError(RuntimeError):
    """Raised when an input or output violates a dataset invariant."""


@dataclass(frozen=True)
class Upstream:
    name: str
    revision: str
    split: str
    rtl_field: str
    priority: int


@dataclass(frozen=True)
class External:
    origin: str
    final: Path
    prepared: Path
    selection: Path


@dataclass(frozen=True)
class Augmentation:
    origin: str
    legacy_shard: str
    model: str
    forced_source: str


@dataclass(frozen=True)
class Config:
    output_root: Path
    legacy_final: Path
    seed_train: Path
    catalog_cache: Path
    split_seed: int
    valid_fraction: float
    split_tolerance: float
    upstream: tuple[Upstream, ...]
    external: tuple[External, ...]
    augmentation: tuple[Augmentation, ...]
    test_shards: tuple[str, ...]
    test_sources: dict[str, Path]


def load_config(path: Path) -> Config:
    path = path.expanduser().resolve()
    with path.open("rb") as source:
        raw = tomllib.load(source)
    dataset = raw["dataset"]

    def configured_path(value: str) -> Path:
        expanded = Path(value).expanduser()
        return (expanded if expanded.is_absolute() else path.parent / expanded).resolve()

    upstream = tuple(Upstream(**item) for item in raw.get("upstream", []))
    external = tuple(
        External(
            origin=item["origin"],
            final=configured_path(item["final"]),
            prepared=configured_path(item["prepared"]),
            selection=configured_path(item["selection"]),
        )
        for item in raw.get("external", [])
    )
    augmentation = tuple(Augmentation(**item) for item in raw.get("augmentation", []))
    tests = raw.get("tests", {})
    result = Config(
        output_root=configured_path(dataset["output_root"]),
        legacy_final=configured_path(dataset["legacy_final"]),
        seed_train=configured_path(dataset["seed_train"]),
        catalog_cache=configured_path(dataset["catalog_cache"]),
        split_seed=int(dataset["split_seed"]),
        valid_fraction=float(dataset["valid_fraction"]),
        split_tolerance=float(dataset["split_tolerance"]),
        upstream=upstream,
        external=external,
        augmentation=augmentation,
        test_shards=tuple(tests.get("shards", [])),
        test_sources={
            name: configured_path(value)
            for name, value in tests.get("sources", {}).items()
        },
    )
    if not 0.0 < result.valid_fraction < 1.0:
        raise UnifiedDatasetError("valid_fraction must be between zero and one")
    if not math.isfinite(result.split_tolerance) or result.split_tolerance < 0:
        raise UnifiedDatasetError("split_tolerance must be finite and nonnegative")
    priorities = [item.priority for item in result.upstream]
    if len(priorities) != len(set(priorities)):
        raise UnifiedDatasetError("upstream priorities must be unique")
    _validate_shards(result)
    return result


def _validate_shards(config: Config) -> None:
    def shard_name(value: str) -> None:
        if (
            not isinstance(value, str)
            or not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or ":" in value
            or Path(value).name != value
            or not value.endswith(".jsonl")
        ):
            raise UnifiedDatasetError(f"shards must be plain .jsonl filenames: {value!r}")

    reserved = {"train.jsonl", "valid.jsonl", "train_aug.jsonl", "valid_aug.jsonl"}
    for value in config.test_shards:
        shard_name(value)
        if value in reserved:
            raise UnifiedDatasetError(f"test shard would overwrite a training output: {value}")
    if len(config.test_shards) != len(set(config.test_shards)):
        raise UnifiedDatasetError("test shard names must be unique")
    for value in config.test_sources:
        shard_name(value)
        if value not in config.test_shards:
            raise UnifiedDatasetError(f"test source has no configured output shard: {value}")
    for item in config.augmentation:
        shard_name(item.legacy_shard)


def _overlaps(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _validate_build_locations(config: Config) -> None:
    """Keep generated output and optional catalog writes away from source inputs."""
    _validate_shards(config)
    module_root = Path(__file__).resolve().parents[1]
    checkout = module_root.parent
    code_root = checkout if (checkout / "pyproject.toml").is_file() else module_root
    protected_directories = (config.legacy_final, code_root)
    protected_files = [config.seed_train]
    for item in config.external:
        protected_files.extend((item.final, item.prepared, item.selection))
    protected_files.extend(config.test_sources.values())
    output, cache = config.output_root, config.catalog_cache
    for write_path, label in ((output, "output_root"), (cache, "catalog_cache")):
        for input_path in (*protected_directories, *protected_files):
            if _overlaps(write_path, input_path):
                raise UnifiedDatasetError(f"{label} overlaps an input or source location: {input_path}")
            if write_path.is_file() and input_path.is_file() and write_path.samefile(input_path):
                raise UnifiedDatasetError(f"{label} aliases an input file: {input_path}")
    if _overlaps(output, cache):
        raise UnifiedDatasetError("output_root and catalog_cache must be separate locations")
    if cache.exists() and not cache.is_file():
        raise UnifiedDatasetError("catalog_cache must be a file location")


def _require_fresh_output(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise UnifiedDatasetError(f"output_root already exists; select a new directory: {output}")


def jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        yield from resynth.jsonl_rows(path)
    except Exception as exc:
        raise UnifiedDatasetError(f"invalid JSONL input {path}: {exc}") from exc


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_rtl(rtl: str) -> str:
    return " ".join(rtl.split())


def rtl_lineage(rtl: str) -> str:
    return "sha256:" + sha256_text(normalized_rtl(rtl))


def canonical_clock(value: Any) -> str:
    if value in (None, ""):
        return ""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise UnifiedDatasetError(f"non-finite clock value: {value!r}")
    return format(numeric, ".17g")


def identity_digest(row: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(row["design_name"]).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(row["rtl"]).encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_clock(row.get("clock")).encode("ascii"))
    return digest.hexdigest()


def external_join_digest(rtl: Any, sdc: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(rtl).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(sdc).encode("utf-8"))
    return digest.hexdigest()


def parse_augmentation_id(value: str) -> dict[str, Any]:
    match = AUGMENTATION_RE.search(value)
    if not match:
        raise UnifiedDatasetError(f"cannot parse augmentation candidate name: {value!r}")
    return {
        "seed_sample_id": match.group("sample").lower(),
        "seed_row_index": int(match.group("sample_index")),
        "optimization_direction": match.group("direction").lower(),
        "generation_index": int(match.group("generation")),
        "candidate_index": int(match.group("candidate")),
    }


def normalize_ppa_row(raw: dict[str, Any], context: str) -> dict[str, Any]:
    missing = [field for field in PPA_FIELDS if field not in raw]
    if missing:
        raise UnifiedDatasetError(f"{context}: missing PPA fields {missing}")
    row = {field: raw[field] for field in PPA_FIELDS}
    synth = str(row["synth"] or "")
    area = str(row["area"] or "")
    delay = str(row["delay"] or "")
    if "Report : cell" in synth:
        synth, area = resynth.format_cell_reports(synth)
    if "Report : timing" in delay:
        delay = resynth.format_timing_report(delay)
    row["synth"], row["area"], row["delay"] = synth, area, delay
    if not all((row["design_name"], row["rtl"], synth, area, delay)):
        raise UnifiedDatasetError(f"{context}: empty identity or report field")
    for field in NUMERIC_REQUIRED:
        try:
            value = float(row[field])
        except (TypeError, ValueError) as exc:
            raise UnifiedDatasetError(f"{context}: {field} is not numeric") from exc
        if not math.isfinite(value) or value < 0:
            raise UnifiedDatasetError(f"{context}: invalid {field}={value!r}")
    for field in ("wns", "tns", "synthesis_time"):
        if row[field] is not None and row[field] != "":
            try:
                value = float(row[field])
            except (TypeError, ValueError) as exc:
                raise UnifiedDatasetError(f"{context}: {field} is not numeric") from exc
            if not math.isfinite(value):
                raise UnifiedDatasetError(f"{context}: non-finite {field}")
    if bool(row["has_clock"]):
        canonical_clock(row["clock"])
    return row


def complete_row(ppa: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in PROVENANCE_FIELDS if field not in metadata]
    if missing:
        raise UnifiedDatasetError(f"internal error: missing provenance fields {missing}")
    row = {field: ppa[field] for field in PPA_FIELDS}
    row.update({field: metadata[field] for field in PROVENANCE_FIELDS})
    if tuple(row) != ALL_FIELDS:
        raise UnifiedDatasetError("internal error: unstable output schema")
    return row


def write_jsonl_row(destination: Any, row: dict[str, Any]) -> None:
    destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS catalog (
            rtl_hash TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            source_split TEXT NOT NULL,
            source_row_index INTEGER NOT NULL,
            module_id TEXT NOT NULL,
            priority INTEGER NOT NULL,
            PRIMARY KEY (rtl_hash, source_dataset)
        );
        CREATE INDEX IF NOT EXISTS catalog_priority ON catalog(rtl_hash, priority);
        CREATE TABLE IF NOT EXISTS seed_index (
            row_index INTEGER PRIMARY KEY,
            rtl_hash TEXT NOT NULL,
            sample_id TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS seed_hash ON seed_index(rtl_hash, row_index);
        CREATE TABLE IF NOT EXISTS external_map (
            version TEXT NOT NULL,
            join_key TEXT NOT NULL,
            metadata TEXT NOT NULL,
            PRIMARY KEY (version, join_key)
        );
        CREATE TABLE IF NOT EXISTS external_identities (
            identity TEXT PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS seen (
            identity TEXT PRIMARY KEY,
            data_origin TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS lineages (
            lineage_id TEXT PRIMARY KEY,
            base_count INTEGER NOT NULL DEFAULT 0,
            aug_count INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    return connection


class Builder:
    def __init__(self, config: Config, staging: Path, *, allow_network: bool = False):
        self.config = config
        self.staging = staging
        self.allow_network = allow_network
        self.work = staging / ".work"
        self.metadata_dir = staging / "metadata"
        self.work.mkdir(parents=True)
        self.metadata_dir.mkdir(parents=True)
        self.db = database(self.work / "build.sqlite3")
        self.counts: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()
        self.model_counts: Counter[str] = Counter()
        self.direction_counts: Counter[str] = Counter()
        self.duplicate_counts: Counter[str] = Counter()
        self.artifacts: dict[str, dict[str, Any]] = {}
        self._artifact_hashes: dict[Path, str] = {}
        self.base_pool_path = self.work / "base.jsonl"
        self.aug_pool_path = self.work / "augmentation.jsonl"
        self.base_lineages: set[str] = set()
        self.unresolved: Counter[str] = Counter()

    def close(self) -> None:
        self.db.close()

    def artifact(self, label: str, path: Path) -> str:
        path = path.resolve()
        if path not in self._artifact_hashes:
            self._artifact_hashes[path] = sha256_file(path)
        digest = self._artifact_hashes[path]
        self.artifacts[label] = {
            "path": str(path),
            "sha256": digest,
            "size": path.stat().st_size,
        }
        return digest

    def _catalog_snapshot_candidate(self) -> Path | None:
        candidate = self.config.legacy_final / "metadata" / "source_catalog.jsonl"
        if candidate.is_file():
            return candidate
        return self.config.catalog_cache if self.config.catalog_cache.is_file() else None

    def build_catalog(self) -> None:
        snapshot_source = self._catalog_snapshot_candidate()
        snapshot_output = self.metadata_dir / "source_catalog.jsonl"
        expected = {item.name: item for item in self.config.upstream}
        loaded = 0
        if snapshot_source:
            with snapshot_output.open("w", encoding="utf-8") as output:
                for line_number, row in jsonl_rows(snapshot_source):
                    source = expected.get(str(row.get("source_dataset")))
                    if source is None or row.get("source_revision") != source.revision:
                        raise UnifiedDatasetError(
                            f"{snapshot_source}:{line_number}: catalog revision mismatch"
                        )
                    inserted = self.db.execute(
                        "INSERT OR IGNORE INTO catalog VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            row["rtl_hash"],
                            source.name,
                            source.revision,
                            source.split,
                            int(row["source_row_index"]),
                            row["module_id"],
                            source.priority,
                        ),
                    ).rowcount
                    if inserted:
                        write_jsonl_row(output, row)
                        loaded += 1
            self.artifact("source_catalog_previous", snapshot_source)
        else:
            if not self.allow_network:
                raise UnifiedDatasetError("source catalog is absent and network access is disabled")
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise UnifiedDatasetError(
                    "Hugging Face datasets is required to create the source catalog"
                ) from exc
            with snapshot_output.open("w", encoding="utf-8") as output:
                for source in sorted(self.config.upstream, key=lambda item: item.priority):
                    dataset = load_dataset(
                        source.name,
                        split=source.split,
                        revision=source.revision,
                        streaming=True,
                    )
                    for row_index, upstream_row in enumerate(dataset):
                        rtl = upstream_row.get(source.rtl_field)
                        if not isinstance(rtl, str) or not rtl.strip():
                            raise UnifiedDatasetError(
                                f"{source.name}:{source.split}:{row_index}: empty {source.rtl_field}"
                            )
                        rtl_hash = sha256_text(normalized_rtl(rtl))
                        module_id = (
                            f"hf:{source.name}@{source.revision}:"
                            f"{source.split}:{row_index}"
                        )
                        inserted = self.db.execute(
                            "INSERT OR IGNORE INTO catalog VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                rtl_hash,
                                source.name,
                                source.revision,
                                source.split,
                                row_index,
                                module_id,
                                source.priority,
                            ),
                        ).rowcount
                        if not inserted:
                            continue
                        write_jsonl_row(
                            output,
                            {
                                "rtl_hash": rtl_hash,
                                "source_dataset": source.name,
                                "source_revision": source.revision,
                                "source_split": source.split,
                                "source_row_index": row_index,
                                "module_id": module_id,
                            },
                        )
                        loaded += 1
                    self.db.commit()
            _validate_build_locations(self.config)
            cache = self.config.catalog_cache
            cache.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation protects existing user inputs and catalog snapshots.
            created = False
            try:
                with cache.open("xb") as destination:
                    created = True
                    with snapshot_output.open("rb") as source:
                        shutil.copyfileobj(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                fsync_directory(cache.parent)
            except BaseException:
                if created:
                    cache.unlink(missing_ok=True)
                raise
        self.db.commit()
        self.counts["catalog_entries"] = loaded

    def build_seed_index(self) -> None:
        path = self.config.seed_train
        if not path.is_file():
            raise UnifiedDatasetError(f"missing seed train file: {path}")
        self.artifact("dataset_v6.2.0_train", path)
        count = 0
        for line_number, row in jsonl_rows(path):
            rtl = row.get("rtl")
            if not isinstance(rtl, str) or not rtl:
                raise UnifiedDatasetError(f"{path}:{line_number}: missing rtl")
            row_index = line_number - 1
            self.db.execute(
                "INSERT INTO seed_index VALUES (?, ?, ?)",
                (row_index, sha256_text(normalized_rtl(rtl)), f"sample_{row_index:06d}"),
            )
            count += 1
            if count % 10000 == 0:
                self.db.commit()
        self.db.commit()
        self.counts["seed_rows"] = count

    def catalog_match(self, rtl_hash: str, forced_source: str = "") -> dict[str, Any]:
        if forced_source:
            row = self.db.execute(
                """SELECT source_dataset, source_revision, source_split,
                          source_row_index, module_id
                   FROM catalog WHERE rtl_hash=? AND source_dataset=?""",
                (rtl_hash, forced_source),
            ).fetchone()
        else:
            row = self.db.execute(
                """SELECT source_dataset, source_revision, source_split,
                          source_row_index, module_id
                   FROM catalog WHERE rtl_hash=? ORDER BY priority LIMIT 1""",
                (rtl_hash,),
            ).fetchone()
        if row is None:
            label = forced_source or "any upstream"
            raise UnifiedDatasetError(f"RTL hash {rtl_hash} is unresolved in {label}")
        return dict(zip(
            ("source_dataset", "source_revision", "source_split", "source_row_index", "module_id"),
            row,
        )) | {"source_match_method": "exact_rtl"}

    def seed_for_index(self, row_index: int) -> tuple[str, str]:
        row = self.db.execute(
            "SELECT rtl_hash, sample_id FROM seed_index WHERE row_index=?", (row_index,)
        ).fetchone()
        if row is None:
            raise UnifiedDatasetError(f"augmentation references missing seed row {row_index}")
        return str(row[0]), str(row[1])

    def sample_for_hash(self, rtl_hash: str) -> str | None:
        row = self.db.execute(
            "SELECT sample_id FROM seed_index WHERE rtl_hash=? ORDER BY row_index LIMIT 1",
            (rtl_hash,),
        ).fetchone()
        return str(row[0]) if row else None

    def prepare_external_maps(self) -> None:
        for item in self.config.external:
            for path in (item.final, item.prepared, item.selection):
                if not path.is_file():
                    raise UnifiedDatasetError(f"missing {item.origin} input: {path}")
            self.artifact(f"{item.origin}_prepared", item.prepared)
            self.artifact(f"{item.origin}_selection", item.selection)
            self.artifact(f"{item.origin}_final", item.final)
            inserted = 0
            collisions = 0
            for line_number, row in jsonl_rows(item.prepared):
                key = external_join_digest(row.get("rtl"), row.get("SDC"))
                metadata = {
                    "candidate_id": row.get("candidate_id"),
                    "source_dataset": row.get("source_dataset"),
                    "source_row_id": row.get("source_row_id"),
                    "source_origin": row.get("source_origin"),
                    "source_url": row.get("source_url"),
                    "prepared_row_index": line_number - 1,
                }
                if not metadata["candidate_id"] or not metadata["source_dataset"]:
                    raise UnifiedDatasetError(
                        f"{item.prepared}:{line_number}: incomplete external provenance"
                    )
                changed = self.db.execute(
                    "INSERT OR IGNORE INTO external_map VALUES (?, ?, ?)",
                    (item.origin, key, json.dumps(metadata, separators=(",", ":"))),
                ).rowcount
                if changed:
                    inserted += 1
                else:
                    collisions += 1
                if line_number % 10000 == 0:
                    self.db.commit()
            if collisions:
                raise UnifiedDatasetError(
                    f"{item.origin}: {collisions} duplicate (rtl, SDC) prepared inputs"
                )
            self.db.commit()
            self.counts[f"{item.origin}_prepared_rows"] = inserted

            final_count = 0
            for line_number, raw in jsonl_rows(item.final):
                ppa = normalize_ppa_row(raw, f"{item.final}:{line_number}")
                identity = identity_digest(ppa)
                self.db.execute(
                    "INSERT OR IGNORE INTO external_identities VALUES (?)", (identity,)
                )
                final_count += 1
            self.db.commit()
            self.counts[f"{item.origin}_successful_rows"] = final_count

    def _accept(self, row: dict[str, Any], pool: Any, pool_name: str) -> bool:
        identity = identity_digest(row)
        changed = self.db.execute(
            "INSERT OR IGNORE INTO seen VALUES (?, ?)", (identity, row["data_origin"])
        ).rowcount
        if not changed:
            previous = self.db.execute(
                "SELECT data_origin FROM seen WHERE identity=?", (identity,)
            ).fetchone()[0]
            self.duplicate_counts[f"{row['data_origin']}<-{previous}"] += 1
            return False
        row["record_id"] = "sha256:" + identity
        write_jsonl_row(pool, row)
        lineage = row["lineage_id"]
        base_delta = 1 if pool_name == "base" else 0
        aug_delta = 1 if pool_name == "augmentation" else 0
        self.db.execute(
            """INSERT INTO lineages(lineage_id, base_count, aug_count) VALUES (?, ?, ?)
               ON CONFLICT(lineage_id) DO UPDATE SET
                 base_count=base_count+excluded.base_count,
                 aug_count=aug_count+excluded.aug_count""",
            (lineage, base_delta, aug_delta),
        )
        self.counts[f"{pool_name}_accepted"] += 1
        self.source_counts[row["source_dataset"]] += 1
        if row["generation_model"]:
            self.model_counts[row["generation_model"]] += 1
        if row["optimization_direction"]:
            self.direction_counts[row["optimization_direction"]] += 1
        return True

    def process_original(self, pool: Any) -> None:
        artifact_paths = [self.config.legacy_final / name for name in ("train.jsonl", "valid.jsonl")]
        for path in artifact_paths:
            if not path.is_file():
                raise UnifiedDatasetError(f"missing original base shard: {path}")
            artifact_label = f"legacy_{path.stem}"
            artifact_hash = self.artifact(artifact_label, path)
            for line_number, raw in jsonl_rows(path):
                if raw.get("record_kind") and raw.get("record_kind") != "original":
                    continue
                ppa = normalize_ppa_row(raw, f"{path}:{line_number}")
                identity = identity_digest(ppa)
                if not raw.get("record_kind") and self.db.execute(
                    "SELECT 1 FROM external_identities WHERE identity=?", (identity,)
                ).fetchone():
                    self.counts["legacy_external_rows_removed"] += 1
                    continue
                rtl_hash = sha256_text(normalized_rtl(str(ppa["rtl"])))
                try:
                    source = self.catalog_match(rtl_hash)
                except UnifiedDatasetError:
                    self.unresolved["original"] += 1
                    if self.unresolved["original"] <= 20:
                        print(
                            f"unresolved original {path}:{line_number} "
                            f"{ppa['design_name']} {rtl_hash}",
                            file=sys.stderr,
                        )
                    continue
                lineage = "sha256:" + rtl_hash
                metadata = {
                    "record_id": "",
                    "record_kind": "original",
                    "data_origin": "dataset_v6.2.1_resynthesis",
                    **source,
                    "source_artifact": artifact_label,
                    "source_artifact_sha256": artifact_hash,
                    "candidate_id": None,
                    "seed_sample_id": self.sample_for_hash(rtl_hash),
                    "generation_model": None,
                    "optimization_direction": None,
                    "generation_index": None,
                    "candidate_index": None,
                    "lineage_id": lineage,
                    "seed_present_in_base": True,
                }
                if raw.get("record_kind") == "original":
                    for field in PROVENANCE_FIELDS:
                        if field in raw and field not in {
                            "record_id",
                            "lineage_id",
                            "seed_present_in_base",
                        }:
                            metadata[field] = raw[field]
                row = complete_row(ppa, metadata)
                if self._accept(row, pool, "base"):
                    self.base_lineages.add(lineage)
                if line_number % 10000 == 0:
                    self.db.commit()
        self.db.commit()
        if self.unresolved["original"]:
            raise UnifiedDatasetError(
                f"{self.unresolved['original']} original rows could not be mapped upstream"
            )

    def process_external(self, pool: Any) -> None:
        for item in self.config.external:
            artifact_label = f"{item.origin}_final"
            artifact_hash = self._artifact_hashes[item.final.resolve()]
            selection_hash = self._artifact_hashes[item.selection.resolve()]
            matched = 0
            for line_number, raw in jsonl_rows(item.final):
                ppa = normalize_ppa_row(raw, f"{item.final}:{line_number}")
                key = external_join_digest(ppa["rtl"], ppa["SDC"])
                match = self.db.execute(
                    "SELECT metadata FROM external_map WHERE version=? AND join_key=?",
                    (item.origin, key),
                ).fetchone()
                if match is None:
                    raise UnifiedDatasetError(
                        f"{item.final}:{line_number}: no prepared external provenance"
                    )
                provenance = json.loads(match[0])
                matched += 1
                rtl_hash = sha256_text(normalized_rtl(str(ppa["rtl"])))
                metadata = {
                    "record_id": "",
                    "record_kind": "external",
                    "data_origin": item.origin,
                    "source_dataset": provenance["source_dataset"],
                    "source_revision": "sha256:" + selection_hash,
                    "source_split": "selected",
                    "source_row_index": provenance["prepared_row_index"],
                    "source_match_method": "external_join",
                    "source_artifact": artifact_label,
                    "source_artifact_sha256": artifact_hash,
                    "module_id": f"external:{item.origin}:{provenance['candidate_id']}",
                    "candidate_id": provenance["candidate_id"],
                    "seed_sample_id": None,
                    "generation_model": None,
                    "optimization_direction": None,
                    "generation_index": None,
                    "candidate_index": None,
                    "lineage_id": "sha256:" + rtl_hash,
                    "seed_present_in_base": True,
                }
                self._accept(complete_row(ppa, metadata), pool, "base")
                if line_number % 10000 == 0:
                    self.db.commit()
            self.db.commit()
            expected = self.counts[f"{item.origin}_successful_rows"]
            if matched != expected:
                raise UnifiedDatasetError(
                    f"{item.origin}: mapped {matched} of {expected} successful rows"
                )

    def augmentation_sources(self) -> list[tuple[Augmentation, Path, bool]]:
        legacy = [
            (item, self.config.legacy_final / item.legacy_shard, False)
            for item in self.config.augmentation
        ]
        if all(path.is_file() for _, path, _ in legacy):
            return legacy
        combined = [
            self.config.legacy_final / "train_aug.jsonl",
            self.config.legacy_final / "valid_aug.jsonl",
        ]
        if not all(path.is_file() for path in combined):
            missing = [str(path) for path in combined if not path.is_file()]
            raise UnifiedDatasetError(
                "legacy augmentation shards are incomplete and combined shards are missing: "
                + ", ".join(missing)
            )
        return [
            (item, path, True)
            for item in self.config.augmentation
            for path in combined
        ]

    def process_augmentations(self, pool: Any) -> None:
        for item, path, filter_origin in self.augmentation_sources():
            label = f"augmentation_{item.origin}_{path.stem}"
            artifact_hash = self.artifact(label, path)
            for line_number, raw in jsonl_rows(path):
                if filter_origin and raw.get("data_origin") != item.origin:
                    continue
                ppa = normalize_ppa_row(raw, f"{path}:{line_number}")
                candidate_id = str(raw.get("candidate_id") or ppa["design_name"])
                parsed = parse_augmentation_id(candidate_id)
                seed_hash, sample_id = self.seed_for_index(parsed["seed_row_index"])
                try:
                    source = self.catalog_match(seed_hash, item.forced_source)
                except UnifiedDatasetError:
                    if not item.forced_source:
                        raise
                    upstream = next(
                        source
                        for source in self.config.upstream
                        if source.name == item.forced_source
                    )
                    seed_artifact_hash = self._artifact_hashes[
                        self.config.seed_train.resolve()
                    ]
                    source = {
                        "source_dataset": upstream.name,
                        "source_revision": upstream.revision,
                        "source_split": upstream.split,
                        "source_row_index": None,
                        "source_match_method": "declared_seed",
                        "module_id": (
                            "legacy:dataset_v6.2.0@sha256:"
                            f"{seed_artifact_hash}:train:{parsed['seed_row_index']}"
                        ),
                    }
                    self.counts[f"declared_seed_fallback:{item.origin}"] += 1
                lineage = "sha256:" + seed_hash
                metadata = {
                    "record_id": "",
                    "record_kind": "augmentation",
                    "data_origin": item.origin,
                    **source,
                    "source_artifact": label,
                    "source_artifact_sha256": artifact_hash,
                    "candidate_id": candidate_id,
                    "seed_sample_id": sample_id,
                    "generation_model": item.model,
                    "optimization_direction": parsed["optimization_direction"],
                    "generation_index": parsed["generation_index"],
                    "candidate_index": parsed["candidate_index"],
                    "lineage_id": lineage,
                    "seed_present_in_base": lineage in self.base_lineages,
                }
                if raw.get("record_kind") == "augmentation":
                    for field in (
                        "source_artifact",
                        "source_artifact_sha256",
                        "source_revision",
                        "source_split",
                        "source_row_index",
                        "source_match_method",
                        "module_id",
                    ):
                        if raw.get(field) is not None:
                            metadata[field] = raw[field]
                if lineage not in self.base_lineages:
                    self.counts["augmentation_rows_without_base_seed"] += 1
                self._accept(complete_row(ppa, metadata), pool, "augmentation")
                if line_number % 10000 == 0:
                    self.db.commit()
        self.db.commit()

    def _test_row(self, raw: dict[str, Any], shard: str, artifact_hash: str, line: int) -> dict[str, Any]:
        ppa = normalize_ppa_row(raw, f"{shard}:{line}")
        rtl_hash = sha256_text(normalized_rtl(str(ppa["rtl"])))
        origin = Path(shard).stem
        metadata = {
            "record_id": "",
            "record_kind": TRUE_TEST_KIND,
            "data_origin": origin,
            "source_dataset": f"local/{origin}",
            "source_revision": "sha256:" + artifact_hash,
            "source_split": "test",
            "source_row_index": line - 1,
            "source_match_method": "artifact",
            "source_artifact": f"test_{origin}",
            "source_artifact_sha256": artifact_hash,
            "module_id": f"test:{origin}:{rtl_hash}",
            "candidate_id": str(raw.get("candidate_id") or ppa["design_name"]),
            "seed_sample_id": None,
            "generation_model": None,
            "optimization_direction": None,
            "generation_index": None,
            "candidate_index": None,
            "lineage_id": "sha256:" + rtl_hash,
            "seed_present_in_base": None,
        }
        if raw.get("record_kind") == TRUE_TEST_KIND:
            for field in PROVENANCE_FIELDS:
                if field in raw and field != "record_id":
                    metadata[field] = raw[field]
        row = complete_row(ppa, metadata)
        row["record_id"] = "sha256:" + identity_digest(row)
        return row

    def process_tests(self) -> None:
        for shard in self.config.test_shards:
            source_path = self.config.test_sources.get(
                shard, self.config.legacy_final / shard
            )
            if not source_path.is_file():
                raise UnifiedDatasetError(f"missing test shard: {source_path}")
            label = f"test_{Path(shard).stem}"
            artifact_hash = self.artifact(label, source_path)
            output_path = self.staging / shard
            count = 0
            with output_path.open("w", encoding="utf-8") as output:
                for line_number, raw in jsonl_rows(source_path):
                    write_jsonl_row(
                        output,
                        self._test_row(raw, shard, artifact_hash, line_number),
                    )
                    count += 1
                output.flush()
                os.fsync(output.fileno())
            self.counts[f"test_rows:{shard}"] = count

    @staticmethod
    def split_lineages(
        groups: dict[str, tuple[int, int]], seed: int, fraction: float
    ) -> tuple[set[str], dict[str, Any]]:
        totals = (
            sum(counts[0] for counts in groups.values()),
            sum(counts[1] for counts in groups.values()),
        )
        targets = (totals[0] * fraction, totals[1] * fraction)

        def objective(base: int, aug: int) -> float:
            values = (base, aug)
            return sum(
                abs(values[index] - targets[index]) / max(totals[index], 1)
                for index in range(2)
                if totals[index]
            )

        valid: set[str] = set()
        current = [0, 0]
        ordered = sorted(
            groups,
            key=lambda lineage: hashlib.sha256(
                f"{seed}:{lineage}".encode("utf-8")
            ).digest(),
        )
        threshold = int(fraction * (1 << 256))
        for lineage in ordered:
            hashed = int.from_bytes(
                hashlib.sha256(f"{seed}:{lineage}".encode("utf-8")).digest(), "big"
            )
            if hashed < threshold:
                valid.add(lineage)
                current[0] += groups[lineage][0]
                current[1] += groups[lineage][1]

        improved = True
        passes = 0
        while improved and passes < 25:
            improved = False
            passes += 1
            for lineage in sorted(
                groups,
                key=lambda value: (-sum(groups[value]), value),
            ):
                base, aug = groups[lineage]
                sign = -1 if lineage in valid else 1
                before = objective(current[0], current[1])
                after = objective(current[0] + sign * base, current[1] + sign * aug)
                if after + 1e-15 < before:
                    if sign > 0:
                        valid.add(lineage)
                    else:
                        valid.remove(lineage)
                    current[0] += sign * base
                    current[1] += sign * aug
                    improved = True
        report = {
            "seed": seed,
            "valid_fraction_target": fraction,
            "lineage_count": len(groups),
            "valid_lineage_count": len(valid),
            "base_total": totals[0],
            "base_valid": current[0],
            "base_valid_fraction": current[0] / totals[0] if totals[0] else 0.0,
            "augmentation_total": totals[1],
            "augmentation_valid": current[1],
            "augmentation_valid_fraction": current[1] / totals[1] if totals[1] else 0.0,
            "improvement_passes": passes,
        }
        return valid, report

    def route_pool(
        self, source: Path, train_name: str, valid_name: str, valid_lineages: set[str]
    ) -> tuple[int, int]:
        train_path = self.staging / train_name
        valid_path = self.staging / valid_name
        train_count = valid_count = 0
        with train_path.open("w", encoding="utf-8") as train, valid_path.open(
            "w", encoding="utf-8"
        ) as valid:
            for _, row in jsonl_rows(source):
                if row["lineage_id"] in valid_lineages:
                    write_jsonl_row(valid, row)
                    valid_count += 1
                else:
                    write_jsonl_row(train, row)
                    train_count += 1
            for destination in (train, valid):
                destination.flush()
                os.fsync(destination.fileno())
        return train_count, valid_count

    def write_lineage_metadata(self, valid_lineages: set[str]) -> None:
        output_path = self.metadata_dir / "lineages.jsonl"
        with output_path.open("w", encoding="utf-8") as output:
            for lineage, base, aug in self.db.execute(
                "SELECT lineage_id, base_count, aug_count FROM lineages ORDER BY lineage_id"
            ):
                write_jsonl_row(
                    output,
                    {
                        "lineage_id": lineage,
                        "split": "valid" if lineage in valid_lineages else "train",
                        "base_row_count": base,
                        "augmentation_row_count": aug,
                    },
                )

    def build(self) -> dict[str, Any]:
        self.build_catalog()
        self.build_seed_index()
        self.prepare_external_maps()
        with self.base_pool_path.open("w", encoding="utf-8") as base_pool:
            self.process_original(base_pool)
            self.process_external(base_pool)
        with self.aug_pool_path.open("w", encoding="utf-8") as aug_pool:
            self.process_augmentations(aug_pool)
        groups = {
            str(lineage): (int(base), int(aug))
            for lineage, base, aug in self.db.execute(
                "SELECT lineage_id, base_count, aug_count FROM lineages"
            )
        }
        valid_lineages, split_report = self.split_lineages(
            groups, self.config.split_seed, self.config.valid_fraction
        )
        for pool in ("base", "augmentation"):
            actual = split_report[f"{pool}_valid_fraction"]
            if abs(actual - self.config.valid_fraction) > self.config.split_tolerance:
                raise UnifiedDatasetError(
                    f"{pool} validation fraction {actual:.6f} exceeds tolerance"
                )
        base_train, base_valid = self.route_pool(
            self.base_pool_path, "train.jsonl", "valid.jsonl", valid_lineages
        )
        aug_train, aug_valid = self.route_pool(
            self.aug_pool_path, "train_aug.jsonl", "valid_aug.jsonl", valid_lineages
        )
        self.process_tests()
        self.write_lineage_metadata(valid_lineages)
        output_counts = {
            "train.jsonl": base_train,
            "valid.jsonl": base_valid,
            "train_aug.jsonl": aug_train,
            "valid_aug.jsonl": aug_valid,
            **{
                shard: self.counts[f"test_rows:{shard}"]
                for shard in self.config.test_shards
            },
        }
        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "revolution.dataset_generation",
            "schema": list(ALL_FIELDS),
            "config": {
                "split_seed": self.config.split_seed,
                "valid_fraction": self.config.valid_fraction,
                "split_tolerance": self.config.split_tolerance,
                "dedup_identity": ["design_name", "exact_rtl", "normalized_clock"],
                "dedup_priority": [
                    "dataset_v6.2.1_resynthesis",
                    *[item.origin for item in self.config.external],
                    *[item.origin for item in self.config.augmentation],
                ],
                "orphan_augmentation_policy": "keep_and_flag",
            },
            "upstream": [item.__dict__ for item in self.config.upstream],
            "inputs": self.artifacts,
            "processing_counts": dict(sorted(self.counts.items())),
            "output_counts": output_counts,
            "source_counts": dict(sorted(self.source_counts.items())),
            "generation_model_counts": dict(sorted(self.model_counts.items())),
            "optimization_direction_counts": dict(sorted(self.direction_counts.items())),
            "duplicate_counts": dict(sorted(self.duplicate_counts.items())),
            "split": split_report,
        }
        manifest_path = self.staging / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_report = verify_dataset(self.staging, self.config, manifest=manifest)
        manifest["output_sha256"] = verify_report["output_sha256"]
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if os.name == "nt":
            # Windows locks the temporary SQLite file until its connection closes.
            self.close()
        shutil.rmtree(self.work)
        return manifest


def verify_dataset(
    root: Path, config: Config, *, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    root = root.resolve()
    if manifest is None:
        manifest_path = root / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise UnifiedDatasetError(f"missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = (
        "train.jsonl",
        "valid.jsonl",
        "train_aug.jsonl",
        "valid_aug.jsonl",
        *config.test_shards,
    )
    output_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    lineage_split: dict[str, str] = {}
    identities: set[str] = set()
    for shard in required:
        path = root / shard
        if not path.is_file():
            raise UnifiedDatasetError(f"missing output shard: {path}")
        count = 0
        is_train_valid = shard in {
            "train.jsonl", "valid.jsonl", "train_aug.jsonl", "valid_aug.jsonl"
        }
        split = "valid" if shard.startswith("valid") else "train"
        for line_number, row in jsonl_rows(path):
            if tuple(row) != ALL_FIELDS:
                raise UnifiedDatasetError(f"{path}:{line_number}: output schema mismatch")
            normalize_ppa_row(row, f"{path}:{line_number}")
            if row["optimization_direction"] not in (None, "area", "delay", "power"):
                raise UnifiedDatasetError(
                    f"{path}:{line_number}: invalid optimization_direction"
                )
            if row["record_id"] != "sha256:" + identity_digest(row):
                raise UnifiedDatasetError(f"{path}:{line_number}: record_id mismatch")
            if is_train_valid:
                identity = row["record_id"]
                if identity in identities:
                    raise UnifiedDatasetError(f"duplicate training identity {identity}")
                identities.add(identity)
                previous = lineage_split.setdefault(row["lineage_id"], split)
                if previous != split:
                    raise UnifiedDatasetError(
                        f"lineage leakage for {row['lineage_id']}: {previous}/{split}"
                    )
            count += 1
        expected = manifest.get("output_counts", {}).get(shard)
        if expected is not None and count != expected:
            raise UnifiedDatasetError(f"{shard}: expected {expected} rows, found {count}")
        counts[shard] = count
        output_hashes[shard] = sha256_file(path)
        recorded_hash = manifest.get("output_sha256", {}).get(shard)
        if recorded_hash is not None and output_hashes[shard] != recorded_hash:
            raise UnifiedDatasetError(f"{shard}: output SHA-256 mismatch")
    return {"counts": counts, "output_sha256": output_hashes}


def preflight(config: Config, *, allow_network: bool = False) -> dict[str, Any]:
    _validate_build_locations(config)
    missing: list[str] = []
    if not config.legacy_final.is_dir():
        missing.append(str(config.legacy_final))
    for path in (config.seed_train, config.legacy_final / "train.jsonl", config.legacy_final / "valid.jsonl"):
        if not path.is_file():
            missing.append(str(path))
    for item in config.external:
        for path in (item.final, item.prepared, item.selection):
            if not path.is_file():
                missing.append(str(path))
    for shard in config.test_shards:
        path = config.test_sources.get(shard, config.legacy_final / shard)
        if not path.is_file():
            missing.append(str(path))
    legacy_aug = [config.legacy_final / item.legacy_shard for item in config.augmentation]
    combined_aug = [
        config.legacy_final / "train_aug.jsonl",
        config.legacy_final / "valid_aug.jsonl",
    ]
    if not all(path.is_file() for path in legacy_aug) and not all(
        path.is_file() for path in combined_aug
    ):
        missing.extend(str(path) for path in legacy_aug if not path.is_file())
    catalog = config.legacy_final / "metadata" / "source_catalog.jsonl"
    catalog_available = catalog.is_file() or config.catalog_cache.is_file()
    if not catalog_available and not allow_network:
        missing.append(
            f"{catalog} or {config.catalog_cache} (or enable network catalog refresh)"
        )
    if missing:
        raise UnifiedDatasetError("missing inputs:\n  " + "\n  ".join(sorted(set(missing))))
    return {
        "status": "ok",
        "output_root": str(config.output_root),
        "catalog_mode": "snapshot" if catalog_available else "network",
        "augmentation_mode": "legacy" if all(path.is_file() for path in legacy_aug) else "combined",
    }


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # Windows does not expose directory descriptors through os.open.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(staging: Path, output: Path) -> None:
    """Publish into a newly reserved directory, without replacing user output."""
    _require_fresh_output(output)
    output.mkdir()  # Exclusive reservation; FileExistsError also guards concurrent builders.
    reservation = output.stat()
    try:
        if os.name == "nt":
            # Windows rename refuses an existing destination directory.
            output.rmdir()
            os.rename(staging, output)
        else:
            # POSIX can atomically replace our own still-empty reserved directory.
            os.replace(staging, output)
        fsync_directory(output.parent)
    except BaseException:
        if output.is_dir():
            current = output.stat()
            if (current.st_dev, current.st_ino) == (reservation.st_dev, reservation.st_ino):
                try:
                    output.rmdir()
                except OSError:
                    pass
        raise


def build_command(config: Config, *, allow_network: bool = False, publish_result: bool = True) -> dict[str, Any]:
    _require_fresh_output(config.output_root)
    preflight(config, allow_network=allow_network)
    output = config.output_root
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    output_mode = 0o775
    os.chmod(staging, output_mode)
    builder = Builder(config, staging, allow_network=allow_network)
    try:
        manifest = builder.build()
        builder.close()
        backup = publish(staging, output) if publish_result else None
        return {
            "status": "published" if publish_result else "built",
            "output_root": str(output if publish_result else staging),
            "backup": str(backup) if backup else None,
            "output_counts": manifest["output_counts"],
            "split": manifest["split"],
        }
    except BaseException:
        builder.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight", help="Validate configured inputs.")
    build_parser = subparsers.add_parser("build", help="Build and publish into a fresh output directory.")
    verify_parser = subparsers.add_parser("verify", help="Verify an existing unified dataset.")
    for command in (preflight_parser, build_parser, verify_parser):
        command.add_argument("--config", type=Path, required=True, help="Explicit input/output TOML configuration.")
    for command in (preflight_parser, build_parser):
        network = command.add_mutually_exclusive_group()
        network.add_argument("--allow-network", action="store_true", help="Allow pinned upstream downloads only if no catalog snapshot exists.")
        network.add_argument("--offline", action="store_true", help="Compatibility alias for the default offline mode.")
    build_parser.add_argument("--no-publish", action="store_true", help="Leave the validated result in its fresh staging directory.")
    verify_parser.add_argument("--dataset-root", type=Path, help="Existing result; defaults to the explicitly configured output_root.")
    return root


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "preflight":
            result = preflight(config, allow_network=args.allow_network)
        elif args.command == "build":
            result = build_command(config, allow_network=args.allow_network, publish_result=not args.no_publish)
        elif args.command == "verify":
            result = verify_dataset((args.dataset_root or config.output_root).expanduser().resolve(), config)
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (UnifiedDatasetError, OSError, KeyError, TypeError, ValueError) as exc:
        argument_parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
