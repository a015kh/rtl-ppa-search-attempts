"""Prepare an explicit, source-only Verilog dataset without measured PPA fields.

Inclusion policies are supplied by the operator. They do not establish ownership,
license compatibility, or permission to redistribute heterogeneous source records.
Every RTL string is preserved exactly; unsafe strings are withheld, never edited.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, OrderedDict
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
from typing import Any, Iterator
from urllib.parse import urlsplit


SCHEMA_VERSION = 1
PACKAGE_FILES = ("verilog.jsonl", "manifest.json", "data-source-distribution.csv", "split-distribution.csv")
RTL_FIELDS = {"rtl_id", "rtl", "source_categories", "record_kinds", "original_splits"}
KIND_MAP = {"original": "base", "base": "base", "external": "base", "augmentation": "generated", "generated": "generated",
            "test": "evaluation", "evaluation": "evaluation"}
KINDS = {"base", "generated", "evaluation"}
STATUSES = {"unverified", "conditional", "cleared"}
LABEL = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EXCLUSION_REASONS = {
    "unidentified_source", "unapproved_base", "unapproved_generated", "unapproved_evaluation",
    "unknown_record_kind", "missing_rtl", "invalid_rtl_utf8", "forbidden_literal",
    "private_identifier_or_path", "credential_marker", "synthesis_or_timing_report", "cell_library_artifact",
}
DISTRIBUTION_FIELDS = ("label", "measured_rows", "included_rows", "withheld_rows", "unique_rtl_texts")
COUNT_FIELDS = {"measured_rows", "included_rows", "withheld_rows", "unique_rtl_texts"}
SCHEMA = {
    "rtl_id": "sha256 of exact UTF-8 rtl bytes",
    "rtl": "Exact original Verilog text, including notices and comments",
    "source_categories": "Sorted unique approved public source labels",
    "record_kinds": "Sorted unique base, generated, or evaluation labels",
    "original_splits": "Sorted unique explicitly supplied public split labels",
}
PRIVACY_PATTERNS = (
    ("private_identifier_or_path", re.compile(r"(?i)(?:[a-z]:[\\/](?:users|documents and settings)[\\/]|/(?:home|users|mnt|scratch|storage|workspace|opt|usr/local)/)")),
    ("private_identifier_or_path", re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")),
    ("private_identifier_or_path", re.compile(r"(?im)(?:^\s*#?\s*SBATCH\s+.*--account|ssh\s+\S+@\S+)")),
    ("credential_marker", re.compile(r"(?i)(?:\bhf_[a-z0-9]{12,}\b|\b(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*[\"']?\S+|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bBearer\s+[a-z0-9._-]{12,})")),
    ("synthesis_or_timing_report", re.compile(r"(?i)(?:generated\s+by\s+(?:synopsys|design compiler|prime\s*time)|(?:design compiler|prime\s*time)\s+(?:version|report)|report_(?:timing|area|power)\s*-|\b(?:slack\s*\((?:MET|VIOLATED)\)|data arrival time|data required time|Total cell area|Cell Internal Power)\b)")),
    ("cell_library_artifact", re.compile(r"(?i)(?:\b(?:target_library|link_library|library\s*\()|[\"'\s][^\s\"']+\.(?:lib|db)(?:[\"'\s]|$)|\b(?:NAND|NOR|INV|BUF|AOI|OAI|AND|OR|XOR|XNOR|DFF|SDFF|DFFR|DFFS|MUX)[A-Z0-9_]*_(?:X|D)\d+\b)")),
)
MANIFEST_FIELDS = {
    "schema_version", "package_type", "release_status", "redistribution_note", "deduplication",
    "record_schema", "counts", "exclusion_reasons", "source_categories", "source_distribution",
    "split_distribution", "outputs",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keys(value: Any, allowed: set[str], required: set[str], message: str) -> None:
    if not isinstance(value, dict) or set(value) - allowed or not required.issubset(value):
        raise ValueError(message)


def _label(value: Any) -> str:
    if not isinstance(value, str) or not LABEL.fullmatch(value):
        raise ValueError("Public labels must use lowercase letters, digits, underscores, or hyphens")
    return value


def _regular_file(path: Path) -> Path:
    expanded = path.expanduser().absolute()
    if any(component.is_symlink() for component in (expanded, *expanded.parents)):
        raise ValueError("Symlink input paths are not supported")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ValueError("Every input must be an existing regular file")
    return resolved


def _url(value: Any) -> str:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ValueError("Registry URLs must be public HTTPS URLs")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Registry URLs must be well-formed public HTTPS URLs") from exc
    try:
        ipaddress.ip_address(parsed.hostname or "")
        literal_ip = True
    except ValueError:
        literal_ip = False
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or port not in (None, 443)
            or parsed.hostname == "localhost" or literal_ip or re.fullmatch(r"[\d.]+", parsed.hostname)
            or parsed.hostname.endswith((".local", ".internal"))):
        raise ValueError("Registry URLs must be public HTTPS URLs without credentials or private addresses")
    return value


@lru_cache(maxsize=16)
def _forbidden_regex(literals: tuple[str, ...]) -> re.Pattern[str] | None:
    """Compile private literal substrings once; no patterns enter public outputs."""
    if not literals:
        return None
    # Escaping is essential: user-supplied markers are literal substrings, never
    # regular expressions. Folding both sides preserves the original comparison.
    # Factor shared cell-name prefixes into a trie. A flat hundreds-way union
    # repeats those comparisons at each candidate character and can be slower
    # than individual substring checks. Iterative traversal supports long markers.
    trie: dict = {}
    for literal in literals:
        node = trie
        for char in literal.casefold():
            node = node.setdefault(char, {})
        node[None] = True
    patterns: dict[int, str] = {}
    pending = [(trie, False)]
    while pending:
        node, visited = pending.pop()
        children = sorted(char for char in node if char is not None)
        if not visited:
            pending.append((node, True))
            pending.extend((node[char], False) for char in children)
            continue
        branches = [re.escape(char) + patterns[id(node[char])] for char in children]
        pattern = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")" if branches else ""
        if None in node and branches:
            pattern = "(?:" + pattern + ")?"
        patterns[id(node)] = pattern
    return re.compile(patterns[id(trie)])


def _unsafe_rtl(rtl: str, forbidden: list[str]) -> str | None:
    folded = rtl.casefold()
    literal_pattern = _forbidden_regex(tuple(forbidden))
    if literal_pattern is not None and literal_pattern.search(folded):
        return "forbidden_literal"
    for reason, pattern in PRIVACY_PATTERNS:
        if pattern.search(rtl):
            return reason
    return None


def load_config(path: Path | str) -> dict[str, Any]:
    """Validate private configuration and resolve paths without exporting it."""
    config_path = _regular_file(Path(path))
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Configuration must be UTF-8 JSON") from exc
    _keys(config, {"schema_version", "inputs", "sources", "forbidden_literals"},
          {"schema_version", "inputs", "sources"}, "Unknown or missing configuration fields")
    if config["schema_version"] != SCHEMA_VERSION or isinstance(config["schema_version"], bool):
        raise ValueError("Unsupported configuration schema version")
    forbidden = config.get("forbidden_literals", [])
    if not isinstance(forbidden, list) or any(not isinstance(value, str) or not value.strip() for value in forbidden):
        raise ValueError("Forbidden literals must be nonempty strings")
    inputs = config["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("At least one explicitly named input split is required")
    prepared_inputs = []
    seen_labels: set[str] = set()
    for entry in inputs:
        _keys(entry, {"split", "path", "sha256"}, {"split", "path"}, "Unknown or missing input fields")
        label = _label(entry["split"])
        if label in seen_labels:
            raise ValueError("Input split labels must be unique")
        seen_labels.add(label)
        if not isinstance(entry["path"], str) or not entry["path"]:
            raise ValueError("Each input requires an explicit path")
        source = Path(entry["path"])
        if not source.is_absolute():
            source = config_path.parent / source
        source = _regular_file(source)
        expected = entry.get("sha256")
        if expected is not None and (not isinstance(expected, str) or not SHA256.fullmatch(expected)):
            raise ValueError("Optional input SHA256 must be a lowercase hexadecimal digest")
        if expected is not None and file_sha256(source) != expected:
            raise ValueError("Input SHA256 does not match the configured digest")
        prepared_inputs.append({"split": label, "path": source, "sha256": expected})
    sources = config["sources"]
    if not isinstance(sources, dict):
        raise ValueError("Sources must be an explicit source_dataset registry")
    prepared_sources = {}
    for identity, entry in sources.items():
        if not isinstance(identity, str) or not identity:
            raise ValueError("Source registry identities must be nonempty strings")
        _keys(entry, {"category", "upstream_url", "license", "license_url", "allow_base", "allow_generated",
                      "allow_evaluation", "redistribution_status"},
              {"category", "upstream_url", "license", "license_url"}, "Unknown or missing source registry fields")
        category = _label(entry["category"])
        status = entry.get("redistribution_status", "unverified")
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError("Redistribution status must be unverified, conditional, or cleared")
        upstream = _url(entry["upstream_url"]) if entry["upstream_url"] is not None else None
        license_url = _url(entry["license_url"]) if entry["license_url"] is not None else None
        license_label = entry["license"]
        if (not isinstance(license_label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.()+ -]{0,99}", license_label)
                or _unsafe_rtl(category + " " + (upstream or "") + " " + (license_url or "") + " " + license_label, forbidden)):
            raise ValueError("Registry public metadata contains unsafe or invalid text")
        flags = {f"allow_{kind}": entry.get(f"allow_{kind}", False) for kind in KINDS}
        if any(not isinstance(value, bool) for value in flags.values()):
            raise ValueError("Source inclusion approvals must be explicit booleans")
        if status == "cleared" and (upstream is None or license_url is None or license_label.casefold() in {"unresolved", "unknown", "noassertion"}):
            raise ValueError("Cleared sources require recorded upstream and license references")
        prepared_sources[identity] = {"category": category, "upstream_url": upstream, "license": license_label,
                                      "license_url": license_url, "redistribution_status": status, **flags}
    for entry in prepared_inputs:
        if _unsafe_rtl(entry["split"], forbidden):
            raise ValueError("Split label contains forbidden text")
    return {"inputs": prepared_inputs, "sources": prepared_sources, "forbidden_literals": forbidden,
            "config_path": config_path}


def _safe_output(output: Path | str, config: dict[str, Any]) -> Path:
    path = Path(output).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise ValueError("Output already exists; choose a new directory")
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Symlink output ancestors are not supported")
    path = path.resolve()
    package = Path(__file__).resolve().parent
    checkout = package.parents[1]
    if path.is_relative_to(package) or ((checkout / "pyproject.toml").is_file() and path.is_relative_to(checkout)):
        raise ValueError("Dataset output must be outside the source checkout")
    for source in [config["config_path"], *(entry["path"] for entry in config["inputs"])]:
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("Output must not overlap inputs or configuration")
    return path


def _read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL input at line {number}")
            try:
                record = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Malformed JSONL input at line {number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL rows must be objects; line {number}")
            yield record


@contextmanager
def _collect(config: dict[str, Any]) -> Iterator[tuple[sqlite3.Connection, dict[str, Any]]]:
    """Use temporary disk storage so input measurement rows need not fit in RAM."""
    with tempfile.TemporaryDirectory(prefix="verilog-source-stage-") as temporary:
        database = sqlite3.connect(str(Path(temporary) / "rows.sqlite"))
        try:
            database.executescript("""
                CREATE TABLE rtl (rtl_id TEXT PRIMARY KEY, rtl TEXT NOT NULL);
                CREATE TABLE membership (rtl_id TEXT, category TEXT, kind TEXT, split TEXT,
                    PRIMARY KEY (rtl_id, category, kind, split));
                CREATE INDEX category_members ON membership(category, rtl_id);
                CREATE INDEX split_members ON membership(split, rtl_id);
            """)
            total = Counter()
            reasons = Counter()
            by_source: dict[str, Counter] = {}
            by_split: dict[str, Counter] = {}
            included_sources: set[str] = set()
            seen_sources: set[str] = set()
            # The inputs repeat exact RTL at several measurement clocks. Cache
            # safety decisions by content hash, without retaining long RTL strings.
            # Source/kind inclusion policy and row counts remain checked per row.
            safety_cache: OrderedDict[str, str | None] = OrderedDict()
            for input_entry in sorted(config["inputs"], key=lambda item: item["split"]):
                split = input_entry["split"]
                split_counts = by_split.setdefault(split, Counter())
                for record in _read_rows(input_entry["path"]):
                    source_name = record.get("source_dataset")
                    source = config["sources"].get(source_name) if isinstance(source_name, str) else None
                    category = source["category"] if source else "unmapped"
                    source_counts = by_source.setdefault(category, Counter())
                    if source:
                        seen_sources.add(source_name)
                    for counter in (total, split_counts, source_counts):
                        counter["measured_rows"] += 1
                    reason = None
                    kind = KIND_MAP.get(record.get("record_kind")) if isinstance(record.get("record_kind"), str) else None
                    rtl = record.get("rtl")
                    if source is None:
                        reason = "unidentified_source"
                    elif kind is None:
                        reason = "unknown_record_kind"
                    elif not source[f"allow_{kind}"]:
                        reason = f"unapproved_{kind}"
                    elif not isinstance(rtl, str) or not rtl.strip():
                        reason = "missing_rtl"
                    else:
                        try:
                            raw = rtl.encode("utf-8")
                        except UnicodeEncodeError:
                            reason = "invalid_rtl_utf8"
                        else:
                            identity = hashlib.sha256(raw).hexdigest()
                            if identity in safety_cache:
                                reason = safety_cache[identity]
                                safety_cache.move_to_end(identity)
                            else:
                                reason = _unsafe_rtl(rtl, config["forbidden_literals"])
                                safety_cache[identity] = reason
                                if len(safety_cache) > 100_000:
                                    safety_cache.popitem(last=False)
                    if reason:
                        reasons[reason] += 1
                        for counter in (total, split_counts, source_counts):
                            counter["withheld_rows"] += 1
                        continue
                    for counter in (total, split_counts, source_counts):
                        counter["included_rows"] += 1
                    included_sources.add(source_name)
                    previous = database.execute("SELECT rtl FROM rtl WHERE rtl_id=?", (identity,)).fetchone()
                    if previous and previous[0] != rtl:
                        raise ValueError("SHA256 collision detected; no dataset was written")
                    database.execute("INSERT OR IGNORE INTO rtl VALUES (?,?)", (identity, rtl))
                    database.execute("INSERT OR IGNORE INTO membership VALUES (?,?,?,?)", (identity, category, kind, split))
                database.commit()
            total["unique_rtl_texts"] = database.execute("SELECT count(*) FROM rtl").fetchone()[0]
            def distribution(counters: dict[str, Counter], field: str) -> list[dict[str, Any]]:
                rows = []
                for label, counter in sorted(counters.items()):
                    unique = database.execute(f"SELECT count(DISTINCT rtl_id) FROM membership WHERE {field}=?", (label,)).fetchone()[0]
                    rows.append({"label": label, **{name: counter[name] for name in DISTRIBUTION_FIELDS[1:-1]}, "unique_rtl_texts": unique})
                return rows
            public_categories = {}
            for source_name in sorted(seen_sources):
                entry = config["sources"][source_name]
                reference = {key: entry[key] for key in ("upstream_url", "license", "license_url", "redistribution_status")}
                references = public_categories.setdefault(entry["category"], [])
                if reference not in references:
                    references.append(reference)
            categories = [{"category": label, "references": sorted(references, key=_json)}
                          for label, references in sorted(public_categories.items())]
            included_categories = {config["sources"][name]["category"] for name in included_sources}
            ready = bool(total["unique_rtl_texts"]) and all(
                config["sources"][name]["redistribution_status"] == "cleared"
                for name in seen_sources if config["sources"][name]["category"] in included_categories)
            manifest = {
                "schema_version": SCHEMA_VERSION, "package_type": "verilog-only",
                "release_status": "ready-for-redistribution" if ready else "staging-only",
                "redistribution_note": "Source inclusion and redistribution statuses are supplied by the operator; the exporter does not establish legal rights or license compatibility.",
                "deduplication": "Exact UTF-8 RTL text globally; repeated measurement/clock rows are not additional unique RTL texts. Category and split unique counts overlap and are not additive.",
                "record_schema": SCHEMA, "counts": {name: total[name] for name in sorted(COUNT_FIELDS)},
                "exclusion_reasons": dict(sorted(reasons.items())), "source_categories": categories,
                "source_distribution": distribution(by_source, "category"),
                "split_distribution": distribution(by_split, "split"), "outputs": {},
            }
            yield database, manifest
        finally:
            database.close()


def preflight(config_path: Path | str) -> dict[str, Any]:
    """Read every input, returning only the approved aggregate candidate summary."""
    config = load_config(config_path)
    with _collect(config) as (_, manifest):
        return manifest


def _records(database: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    for identity, rtl in database.execute("SELECT rtl_id, rtl FROM rtl ORDER BY rtl_id"):
        memberships = database.execute("SELECT category,kind,split FROM membership WHERE rtl_id=?", (identity,)).fetchall()
        yield {"rtl_id": identity, "rtl": rtl,
               "source_categories": sorted({row[0] for row in memberships}),
               "record_kinds": sorted({row[1] for row in memberships}),
               "original_splits": sorted({row[2] for row in memberships})}


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DISTRIBUTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def build_release(config_path: Path | str, output: Path | str) -> dict[str, Any]:
    """Create a new deterministic four-file candidate package atomically."""
    config = load_config(config_path)
    destination = _safe_output(output, config)
    with _collect(config) as (database, manifest):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".verilog-package-", dir=destination.parent))
        try:
            with (temporary / "verilog.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for record in _records(database):
                    stream.write(_json(record) + "\n")
            _write_csv(temporary / "data-source-distribution.csv", manifest["source_distribution"])
            _write_csv(temporary / "split-distribution.csv", manifest["split_distribution"])
            manifest["outputs"] = {name: {"sha256": file_sha256(temporary / name), "bytes": (temporary / name).stat().st_size}
                                   for name in PACKAGE_FILES if name != "manifest.json"}
            (temporary / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8", newline="\n")
            verify_release(temporary)
            if destination.exists() or destination.is_symlink():
                raise ValueError("Output appeared during preparation; no existing files were changed")
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return manifest


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Package counts must be nonnegative integers")
    return value


def _validate_distribution(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("Invalid distribution rows")
    last = None
    for row in rows:
        _keys(row, set(DISTRIBUTION_FIELDS), set(DISTRIBUTION_FIELDS), "Unexpected distribution fields")
        label = _label(row["label"])
        if last is not None and label <= last:
            raise ValueError("Distribution labels must be sorted and unique")
        last = label
        for name in COUNT_FIELDS:
            _count(row[name])
        if row["included_rows"] + row["withheld_rows"] != row["measured_rows"] or row["unique_rtl_texts"] > row["included_rows"]:
            raise ValueError("Distribution row counts are inconsistent")
    return rows


def _check_csv(path: Path, expected: list[dict[str, Any]]) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(DISTRIBUTION_FIELDS):
            raise ValueError("Unexpected distribution CSV fields")
        observed = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed distribution CSV")
            try:
                observed.append({"label": row["label"], **{field: int(row[field]) for field in DISTRIBUTION_FIELDS[1:]}})
            except ValueError as exc:
                raise ValueError("Distribution CSV counts must be integers") from exc
    if observed != expected:
        raise ValueError("Distribution CSV differs from the public manifest")


def verify_release(output: Path | str) -> dict[str, Any]:
    """Verify hashes, exact schema, deduplication, and public aggregate consistency."""
    root = Path(output).expanduser().absolute()
    if not root.is_dir() or root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise ValueError("Verification requires a regular package directory")
    if {path.name for path in root.iterdir()} != set(PACKAGE_FILES):
        raise ValueError("Package must contain exactly the four permitted files")
    for name in PACKAGE_FILES:
        _regular_file(root / name)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _keys(manifest, MANIFEST_FIELDS, MANIFEST_FIELDS, "Unexpected public manifest fields")
    if manifest["schema_version"] != SCHEMA_VERSION or isinstance(manifest["schema_version"], bool) or manifest["package_type"] != "verilog-only":
        raise ValueError("Unsupported public dataset schema")
    if (not isinstance(manifest["release_status"], str)
            or manifest["release_status"] not in {"staging-only", "ready-for-redistribution"}
            or manifest["record_schema"] != SCHEMA):
        raise ValueError("Invalid public release status or record schema")
    for field in ("redistribution_note", "deduplication"):
        if not isinstance(manifest[field], str) or _unsafe_rtl(manifest[field], []):
            raise ValueError("Unsafe public manifest description")
    _keys(manifest["counts"], COUNT_FIELDS, COUNT_FIELDS, "Unexpected aggregate count fields")
    for value in manifest["counts"].values():
        _count(value)
    counts = manifest["counts"]
    if counts["included_rows"] + counts["withheld_rows"] != counts["measured_rows"] or counts["unique_rtl_texts"] > counts["included_rows"]:
        raise ValueError("Aggregate row counts are inconsistent")
    if not isinstance(manifest["exclusion_reasons"], dict) or set(manifest["exclusion_reasons"]) - EXCLUSION_REASONS:
        raise ValueError("Unexpected withholding reason")
    if sum(_count(value) for value in manifest["exclusion_reasons"].values()) != counts["withheld_rows"]:
        raise ValueError("Withholding counts are inconsistent")
    sources = _validate_distribution(manifest["source_distribution"])
    splits = _validate_distribution(manifest["split_distribution"])
    for rows in (sources, splits):
        for field in ("measured_rows", "included_rows", "withheld_rows"):
            if sum(row[field] for row in rows) != counts[field]:
                raise ValueError("Distribution totals differ from package counts")
    categories = manifest["source_categories"]
    if not isinstance(categories, list):
        raise ValueError("Invalid category registry")
    category_labels = []
    category_cleared = {}
    for entry in categories:
        _keys(entry, {"category", "references"}, {"category", "references"}, "Unexpected public category fields")
        label = _label(entry["category"])
        category_labels.append(label)
        references = entry["references"]
        if not isinstance(references, list) or not references or references != sorted(references, key=_json):
            raise ValueError("Category references must be nonempty and sorted")
        statuses = []
        for reference in references:
            _keys(reference, {"upstream_url", "license", "license_url", "redistribution_status"},
                  {"upstream_url", "license", "license_url", "redistribution_status"}, "Unexpected public reference fields")
            if reference["upstream_url"] is not None:
                _url(reference["upstream_url"])
            if reference["license_url"] is not None:
                _url(reference["license_url"])
            if (not isinstance(reference["license"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.()+ -]{0,99}", reference["license"])
                    or not isinstance(reference["redistribution_status"], str)
                    or reference["redistribution_status"] not in STATUSES or _unsafe_rtl(_json(reference), [])):
                raise ValueError("Invalid or unsafe public source reference")
            statuses.append(reference["redistribution_status"])
            if reference["redistribution_status"] == "cleared" and (
                    reference["upstream_url"] is None or reference["license_url"] is None
                    or reference["license"].casefold() in {"unresolved", "unknown", "noassertion"}):
                raise ValueError("Cleared public references require actual upstream and license URLs")
        category_cleared[label] = all(status == "cleared" for status in statuses)
    if category_labels != sorted(set(category_labels)):
        raise ValueError("Public source categories must be sorted and unique")
    if set(row["label"] for row in sources) - (set(category_labels) | {"unmapped"}):
        raise ValueError("Unknown source distribution label")
    outputs = manifest["outputs"]
    expected_outputs = set(PACKAGE_FILES) - {"manifest.json"}
    _keys(outputs, expected_outputs, expected_outputs, "Unexpected output checksum entries")
    for name, metadata in outputs.items():
        _keys(metadata, {"sha256", "bytes"}, {"sha256", "bytes"}, "Unexpected output checksum fields")
        if not isinstance(metadata["sha256"], str) or not SHA256.fullmatch(metadata["sha256"]):
            raise ValueError("Invalid public output SHA256")
        if file_sha256(root / name) != metadata["sha256"] or (root / name).stat().st_size != _count(metadata["bytes"]):
            raise ValueError("Output checksum or byte count differs from manifest")
    _check_csv(root / "data-source-distribution.csv", sources)
    _check_csv(root / "split-distribution.csv", splits)
    category_counts, split_counts = Counter(), Counter()
    previous = None
    records_count = 0
    for record in _read_rows(root / "verilog.jsonl"):
        _keys(record, RTL_FIELDS, RTL_FIELDS, "Unexpected Verilog record fields; measured/private fields are forbidden")
        rtl, identity = record["rtl"], record["rtl_id"]
        if not isinstance(rtl, str) or not rtl.strip() or not isinstance(identity, str) or not SHA256.fullmatch(identity):
            raise ValueError("Invalid RTL text or RTL identity")
        if hashlib.sha256(rtl.encode("utf-8")).hexdigest() != identity:
            raise ValueError("RTL identity does not match its exact text")
        if previous is not None and identity <= previous:
            raise ValueError("RTL identities must be sorted and globally unique")
        previous = identity
        if _unsafe_rtl(rtl, []):
            raise ValueError("Unsafe RTL content found in package")
        for field, permitted in (("source_categories", set(category_labels)), ("record_kinds", KINDS),
                                 ("original_splits", {row["label"] for row in splits})):
            values = record[field]
            if (not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values)
                    or values != sorted(set(values)) or set(values) - permitted):
                raise ValueError("Invalid or unknown Verilog metadata labels")
        category_counts.update(record["source_categories"])
        split_counts.update(record["original_splits"])
        records_count += 1
    if records_count != counts["unique_rtl_texts"]:
        raise ValueError("Unique RTL count differs from package records")
    if set(category_counts) - {row["label"] for row in sources}:
        raise ValueError("Every observed RTL category requires a source distribution row")
    for rows, observed in ((sources, category_counts), (splits, split_counts)):
        if any(observed[row["label"]] != row["unique_rtl_texts"] for row in rows):
            raise ValueError("Unique RTL distribution differs from package records")
    if manifest["release_status"] == "ready-for-redistribution" and (not records_count or any(not category_cleared[label] for label in category_counts)):
        raise ValueError("Ready status conflicts with recorded source rights review")
    return {"status": "verified", "release_status": manifest["release_status"], "unique_rtl_texts": records_count,
            "output_files": len(PACKAGE_FILES)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a source-only Verilog candidate dataset from explicit inclusion policies.")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight", help="Scan inputs and print only aggregate inclusion/withholding counts")
    preflight_parser.add_argument("--config", required=True, type=Path, help="Private input/policy JSON configuration")
    build_parser = commands.add_parser("build", help="Write a new deterministic four-file candidate package")
    build_parser.add_argument("--config", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path, help="New output directory outside the source checkout")
    verify_parser = commands.add_parser("verify", help="Check an independent package's schema, hashes, and counts")
    verify_parser.add_argument("--package", "--output", dest="package", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.config)
        elif args.command == "build":
            manifest = build_release(args.config, args.output)
            result = {"status": "prepared", "release_status": manifest["release_status"], "counts": manifest["counts"]}
        else:
            result = verify_release(args.package)
        print(_json(result))
    except (ValueError, OSError, UnicodeError, sqlite3.Error) as exc:
        # Never print paths, input identities, forbidden literals, or measured rows.
        if isinstance(exc, ValueError) and not isinstance(exc, UnicodeError):
            print(f"Verilog preparation failed: {exc}", file=sys.stderr)
        else:
            print("Verilog preparation failed: input/output operation did not complete", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
