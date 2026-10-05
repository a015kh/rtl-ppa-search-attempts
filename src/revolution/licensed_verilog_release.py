"""Prepare an exact-file, record-scoped Verilog subset from reviewed evidence.

This workflow verifies supplied provenance evidence; it does not infer a license
from a collection label or determine whether an upstream licensor owns the code.
Original RTL bytes and complete supplied notices are preserved without editing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any

from . import verilog_release as core

REVISION = re.compile(r"[0-9a-f]{40}\Z")
LEDGER_FIELDS = {"rtl_id", "source_dataset", "record_kind", "upstream_url",
                 "upstream_revision", "upstream_file", "upstream_file_sha256",
                 "local_source_file", "notice_ids"}
PUBLIC_RIGHTS_FIELDS = {"rtl_id", "source_category", "record_kind", "upstream_url",
                       "upstream_revision", "upstream_file", "upstream_file_sha256", "notice_ids"}
NOTICE_FIELDS = {"id", "path", "sha256", "license", "upstream_url", "upstream_revision", "upstream_file"}
SELECTION_FIELDS = {"schema_version", "policy", "counts", "exclusion_reasons", "source_distribution",
                    "split_distribution", "notices", "limits"}
BASE_FILES = {"README.md", "NOTICES.md", "selection.json", "record-rights.jsonl", "checksums.json"}


def _relative(value: Any) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or "?" in value or "#" in value or any(char.isspace() for char in value)):
        raise ValueError("Upstream files require simple relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in value.split("/")):
        raise ValueError("Upstream file path must stay within its repository")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not core.SHA256.fullmatch(value):
        raise ValueError("Evidence requires a lowercase SHA256 digest")
    return value


def _origin(value: Any, revision: Any, filename: Any) -> tuple[str, str, str]:
    url = core._url(value)
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", url):
        raise ValueError("Exact-file evidence requires a public GitHub repository URL")
    if not isinstance(revision, str) or not REVISION.fullmatch(revision):
        raise ValueError("Evidence requires an immutable full Git revision")
    return url, revision, _relative(filename)


def _resolve(value: Any, parent: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Evidence requires an explicit local input path")
    path = Path(value)
    return core._regular_file(path if path.is_absolute() else parent / path)


def load_review(config_path: Path | str) -> dict[str, Any]:
    """Validate private evidence and check exact source and notice bytes."""
    path = core._regular_file(Path(config_path))
    value = json.loads(path.read_text(encoding="utf-8"))
    base_keys = {"schema_version", "inputs", "sources", "forbidden_literals"}
    core._keys(value, base_keys | {"rights_ledger", "notices", "historical_mixture", "recipes_document"},
               {"schema_version", "inputs", "sources", "rights_ledger", "notices"},
               "Unknown or missing licensed-release configuration fields")
    if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
        raise ValueError("Unsupported reviewed-evidence schema")
    base = {key: value[key] for key in base_keys if key in value}
    for item in base["inputs"]:
        item["path"] = str(_resolve(item["path"], path.parent))
    with tempfile.TemporaryDirectory(prefix="verilog-review-config-") as work:
        temporary = Path(work) / "config.json"
        temporary.write_text(core._json(base), encoding="utf-8")
        config = core.load_config(temporary)
    config["config_path"] = path
    categories = [source["category"] for source in config["sources"].values()]
    if len(categories) != len(set(categories)):
        raise ValueError("Reviewed inputs require a distinct public category for each source identity")
    notices = {}
    if not isinstance(value["notices"], list):
        raise ValueError("Notices must be an explicit list")
    for notice in value["notices"]:
        core._keys(notice, NOTICE_FIELDS, NOTICE_FIELDS, "Unknown or missing notice fields")
        identity = core._label(notice["id"])
        if identity in notices:
            raise ValueError("Duplicate notice identity")
        source = _resolve(notice["path"], path.parent)
        digest = _digest(notice["sha256"])
        if core.file_sha256(source) != digest:
            raise ValueError("Notice SHA256 differs from reviewed evidence")
        text = source.read_bytes().decode("utf-8")
        if not re.search(r"(?i)copyright", text):
            raise ValueError("Complete copyright-bearing notices are required")
        if core._unsafe_rtl(text, config["forbidden_literals"]):
            raise ValueError("Notice contains forbidden private information")
        url, revision, filename = _origin(notice["upstream_url"], notice["upstream_revision"], notice["upstream_file"])
        label = notice["license"]
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.()+ -]{0,99}", label):
            raise ValueError("Notice requires an explicit license label")
        notices[identity] = {"id": identity, "path": source, "sha256": digest, "license": label,
                             "upstream_url": url, "upstream_revision": revision, "upstream_file": filename}
    ledger = _resolve(value["rights_ledger"], path.parent)
    rights = {}
    for item in core._read_rows(ledger):
        core._keys(item, LEDGER_FIELDS, LEDGER_FIELDS, "Unknown or missing record-rights fields")
        identity = _digest(item["rtl_id"])
        if identity != _digest(item["upstream_file_sha256"]):
            raise ValueError("Only exact upstream-file SHA256 matches can be approved")
        source_name, kind = item["source_dataset"], item["record_kind"]
        if source_name not in config["sources"] or kind not in core.KINDS:
            raise ValueError("Evidence must name a configured source and canonical record kind")
        source_file = _resolve(item["local_source_file"], ledger.parent)
        if core.file_sha256(source_file) != identity:
            raise ValueError("Exact upstream source bytes differ from the approved RTL digest")
        url, revision, filename = _origin(item["upstream_url"], item["upstream_revision"], item["upstream_file"])
        ids = item["notice_ids"]
        if (not isinstance(ids, list) or not ids or ids != sorted(set(ids))
                or any(not isinstance(n, str) or n not in notices for n in ids)):
            raise ValueError("Evidence must reference complete configured notices")
        for n in ids:
            if (notices[n]["upstream_url"], notices[n]["upstream_revision"]) != (url, revision):
                raise ValueError("Source and notice evidence must use the same repository revision")
        public = {"rtl_id": identity, "source_category": config["sources"][source_name]["category"],
                  "record_kind": kind, "upstream_url": url, "upstream_revision": revision,
                  "upstream_file": filename, "upstream_file_sha256": identity, "notice_ids": ids}
        if core._unsafe_rtl(core._json(public), config["forbidden_literals"]):
            raise ValueError("Rights evidence contains forbidden private information")
        key = (source_name, kind, identity)
        if key in rights:
            raise ValueError("Duplicate source/kind/hash approval")
        rights[key] = public
    extras = {}
    for name in ("historical_mixture", "recipes_document"):
        if name in value:
            extras[name] = _resolve(value[name], path.parent)
            if core._unsafe_rtl(extras[name].read_text(encoding="utf-8"), config["forbidden_literals"]):
                raise ValueError("Public companion document contains forbidden information")
    if "historical_mixture" in extras:
        with extras["historical_mixture"].open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["source", "base_rows", "generated_rows", "historical_rows", "historical_percent"]:
                raise ValueError("Historical mixture must contain only the five numeric aggregate fields")
            for row in reader:
                if None in row or not row["source"]:
                    raise ValueError("Invalid historical mixture row")
                try:
                    a, b, total = (int(row[k]) for k in ("base_rows", "generated_rows", "historical_rows"))
                    percent = float(row["historical_percent"])
                except ValueError as exc:
                    raise ValueError("Historical mixture requires numeric aggregate values") from exc
                if min(a, b, total) < 0 or a + b != total or not 0 <= percent <= 100:
                    raise ValueError("Historical mixture counts are inconsistent")
    return {"core": config, "rights": rights, "notices": notices, "ledger_path": ledger, "extras": extras}


def _raw_notice_url(notice: dict) -> str:
    return notice["upstream_url"].replace("https://github.com/", "https://raw.githubusercontent.com/") + "/" + notice["upstream_revision"] + "/" + notice["upstream_file"]


def build_reviewed_release(config_path: Path | str, output: Path | str) -> dict[str, Any]:
    """Filter by source, kind and exact bytes before building the source-only core."""
    review = load_review(config_path)
    config = review["core"]
    destination = core._safe_output(output, config)
    protected = [review["ledger_path"], *(n["path"] for n in review["notices"].values()), *review["extras"].values()]
    for source in protected:
        if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("Output must not overlap evidence or companion inputs")
    counters, sources, splits, reasons = Counter(), {}, {}, Counter()
    unique, source_unique, split_unique, emitted_rights = set(), {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="verilog-reviewed-inputs-") as work:
        stage = Path(work)
        selected_inputs, approved_sources = [], {}
        for number, input_entry in enumerate(sorted(config["inputs"], key=lambda e: e["split"])):
            split = input_entry["split"]
            split_counter = splits.setdefault(split, Counter())
            split_ids = split_unique.setdefault(split, set())
            selected = stage / f"selected-{number}.jsonl"
            with selected.open("w", encoding="utf-8", newline="\n") as stream:
                for record in core._read_rows(input_entry["path"]):
                    name = record.get("source_dataset")
                    source = config["sources"].get(name) if isinstance(name, str) else None
                    category = source["category"] if source else "unmapped"
                    source_counter = sources.setdefault(category, Counter())
                    source_ids = source_unique.setdefault(category, set())
                    for count in (counters, source_counter, split_counter):
                        count["measured_rows"] += 1
                    kind = core.KIND_MAP.get(record.get("record_kind")) if isinstance(record.get("record_kind"), str) else None
                    rtl = record.get("rtl")
                    reason, identity, proof = None, None, None
                    if not source:
                        reason = "unidentified_source"
                    elif kind is None:
                        reason = "unknown_record_kind"
                    elif not source[f"allow_{kind}"]:
                        reason = f"unapproved_{kind}"
                    elif not isinstance(rtl, str) or not rtl.strip():
                        reason = "missing_rtl"
                    else:
                        try:
                            identity = hashlib.sha256(rtl.encode("utf-8")).hexdigest()
                        except UnicodeEncodeError:
                            reason = "invalid_rtl_utf8"
                        else:
                            proof = review["rights"].get((name, kind, identity))
                            reason = "unapproved_record" if proof is None else core._unsafe_rtl(rtl, config["forbidden_literals"])
                    if reason:
                        reasons[reason] += 1
                        for count in (counters, source_counter, split_counter):
                            count["withheld_rows"] += 1
                        continue
                    for count in (counters, source_counter, split_counter):
                        count["included_rows"] += 1
                    unique.add(identity); source_ids.add(identity); split_ids.add(identity)
                    emitted_rights[(identity, category, kind)] = proof
                    notices = [review["notices"][n] for n in proof["notice_ids"]]
                    key = "reviewed-" + hashlib.sha256(core._json([category, kind, proof["upstream_url"],
                        proof["upstream_revision"], proof["notice_ids"]]).encode()).hexdigest()
                    approved_sources[key] = {"category": category, "upstream_url": proof["upstream_url"],
                        "license": " + ".join(sorted({n["license"] for n in notices})),
                        "license_url": _raw_notice_url(notices[0]), "redistribution_status": "cleared",
                        "allow_base": kind == "base", "allow_generated": kind == "generated", "allow_evaluation": kind == "evaluation"}
                    stream.write(core._json({"rtl": rtl, "source_dataset": key, "record_kind": kind}) + "\n")
            selected_inputs.append({"split": split, "path": str(selected)})
        counters["unique_rtl_texts"] = len(unique)
        def distribution(counts, ids):
            return [{"label": label, **{k: counter[k] for k in core.DISTRIBUTION_FIELDS[1:-1]},
                     "unique_rtl_texts": len(ids[label])} for label, counter in sorted(counts.items())]
        used_notice_ids = sorted({n for proof in emitted_rights.values() for n in proof["notice_ids"]})
        public_notices = [{k: value for k, value in review["notices"][n].items() if k != "path"}
                          | {"package_file": f"licenses/{n}.txt"} for n in used_notice_ids]
        selection = {"schema_version": 1, "policy": "Exact UTF-8 upstream-file SHA256, explicit source and record kind, complete supplied notices, then confidentiality filtering. Collection labels never approve unmatched records.",
            "counts": {k: counters[k] for k in sorted(core.COUNT_FIELDS)}, "exclusion_reasons": dict(sorted(reasons.items())),
            "source_distribution": distribution(sources, source_unique), "split_distribution": distribution(splits, split_unique),
            "notices": public_notices,
            "limits": ["Upstream revisions are review evidence; original historical import revisions may be unknown.",
                "This is a selected source-code subset, not the full training mixture, a PPA dataset, or an evaluation benchmark.",
                "Original split memberships are historical and do not guarantee family independence.",
                "Review checks byte identity and supplied notices, not upstream ownership or legal compatibility beyond the recorded evidence."]}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".reviewed-verilog-", dir=destination.parent))
        try:
            cfg = stage / "selected-config.json"
            cfg.write_text(core._json({"schema_version": 1, "inputs": selected_inputs, "sources": approved_sources,
                                      "forbidden_literals": config["forbidden_literals"]}), encoding="utf-8")
            core.build_release(cfg, temporary / "data")
            with (temporary / "record-rights.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for key, proof in sorted(emitted_rights.items()):
                    stream.write(core._json(proof) + "\n")
            (temporary / "selection.json").write_text(core._json(selection) + "\n", encoding="utf-8")
            for n in used_notice_ids:
                target = temporary / f"licenses/{n}.txt"
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(review["notices"][n]["path"].read_bytes())
            notice_lines = ["# Notices for the selected Verilog subset", "", "The code repository's license does not relicense this dataset. Each record references its applicable complete upstream notices in `record-rights.jsonl`. Unmatched and uncleared source payloads are excluded.", ""]
            for notice in public_notices:
                notice_lines += [f"- {notice['upstream_url']} at `{notice['upstream_revision']}`: {notice['license']}; full notice in [{notice['package_file']}]({notice['package_file']})."]
            (temporary / "NOTICES.md").write_text("\n".join(notice_lines) + "\n", encoding="utf-8")
            for key, filename in (("historical_mixture", "historical-training-mixture.csv"), ("recipes_document", "SOURCE_RECIPES.md")):
                if key in review["extras"]:
                    shutil.copyfile(review["extras"][key], temporary / filename)
            text = f"""---
license: other
---
# Selected Verilog from RTL PPA research attempts

This unpublished preparation contains **{len(unique)} exact source strings**, representing **{counters['included_rows']} historical measurement rows**. Repeated clock measurements are not additional designs. Each included source string exactly matches a reviewed upstream file and retains its original text. See [record-level evidence](record-rights.jsonl) and [complete notices](NOTICES.md).

The source-only records are in `data/verilog.jsonl`: exact `rtl_id`, `rtl`, public `source_categories`, `record_kinds` and `original_splits`. No PPA labels, predictions, tool reports, cell-library data, weights or private archive identifiers are included. Recorded membership is not a new family-independent partition; this subset cannot reproduce the PPA results or train the retained model.

`selection.json` and its source/split tables describe what was selected and withheld from the supplied historical files. When present, `historical-training-mixture.csv` describes the original training/validation mixture, **not this released subset**. `SOURCE_RECIPES.md` gives upstream links and retrieval/preparation recipes for excluded sources; their payloads are not bundled.

The evidence uses current review revisions, not claimed original import revisions. Exact matching and complete notices do not establish ownership of every upstream sample. The rest of the combined candidate remains private. This preparation does not publish to Hugging Face or GitHub.

Verify with `python scripts/prepare_licensed_verilog_release.py verify --output /path/to/package` from the code checkout, or `rtl-ppa-licensed-verilog-release verify --output /path/to/package` after installation.
"""
            (temporary / "README.md").write_text(text, encoding="utf-8")
            files = {p.relative_to(temporary).as_posix(): {"sha256": core.file_sha256(p), "bytes": p.stat().st_size}
                     for p in sorted(temporary.rglob("*")) if p.is_file()}
            (temporary / "checksums.json").write_text(core._json(files) + "\n", encoding="utf-8")
            verify_reviewed_release(temporary)
            if destination.exists() or destination.is_symlink():
                raise ValueError("Output appeared during preparation")
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return selection


def verify_reviewed_release(output: Path | str) -> dict[str, Any]:
    root = Path(output).absolute()
    if not root.is_dir() or root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("Verification requires a regular package directory")
    files = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks are forbidden in the package")
    if not BASE_FILES.issubset(files):
        raise ValueError("Missing reviewed-package files")
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    if set(checksums) != set(files) - {"checksums.json"}:
        raise ValueError("Unexpected or missing package payload")
    for name, metadata in checksums.items():
        core._keys(metadata, {"sha256", "bytes"}, {"sha256", "bytes"}, "Unexpected checksum metadata")
        if core.file_sha256(files[name]) != _digest(metadata["sha256"]) or files[name].stat().st_size != core._count(metadata["bytes"]):
            raise ValueError("Package checksum mismatch")
    for path in files.values():
        if core._unsafe_rtl(path.read_text(encoding="utf-8"), []):
            raise ValueError("Package contains a private identifier, credential or tool artifact")
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    core._keys(selection, SELECTION_FIELDS, SELECTION_FIELDS, "Unexpected selection fields")
    if selection["schema_version"] != 1 or isinstance(selection["schema_version"], bool):
        raise ValueError("Unsupported selection schema")
    core._keys(selection["counts"], core.COUNT_FIELDS, core.COUNT_FIELDS, "Unexpected selection counts")
    counts = selection["counts"]
    for value in counts.values():
        core._count(value)
    if counts["included_rows"] + counts["withheld_rows"] != counts["measured_rows"]:
        raise ValueError("Selection row counts are inconsistent")
    if set(selection["exclusion_reasons"]) - (core.EXCLUSION_REASONS | {"unapproved_record"}):
        raise ValueError("Unknown selection exclusion reason")
    if sum(core._count(n) for n in selection["exclusion_reasons"].values()) != counts["withheld_rows"]:
        raise ValueError("Selection withholding counts are inconsistent")
    for field in ("source_distribution", "split_distribution"):
        distribution = core._validate_distribution(selection[field])
        for name in ("measured_rows", "included_rows", "withheld_rows"):
            if sum(r[name] for r in distribution) != counts[name]:
                raise ValueError("Selection distribution totals disagree")
    core.verify_release(root / "data")
    core_counts = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))["counts"]
    if core_counts["unique_rtl_texts"] != counts["unique_rtl_texts"] or core_counts["included_rows"] != counts["included_rows"]:
        raise ValueError("Selected core counts disagree")
    notices = {}
    for item in selection["notices"]:
        core._keys(item, NOTICE_FIELDS - {"path"} | {"package_file"}, NOTICE_FIELDS - {"path"} | {"package_file"}, "Unexpected public notice fields")
        n = core._label(item["id"])
        if n in notices or item["package_file"] != f"licenses/{n}.txt":
            raise ValueError("Invalid packaged notice identity")
        _origin(item["upstream_url"], item["upstream_revision"], item["upstream_file"])
        if item["package_file"] not in files or core.file_sha256(files[item["package_file"]]) != _digest(item["sha256"]):
            raise ValueError("Packaged notice differs from rights evidence")
        if not re.search(r"(?i)copyright", files[item["package_file"]].read_text(encoding="utf-8")):
            raise ValueError("Incomplete notice")
        notices[n] = item
    records = {r["rtl_id"]: r for r in core._read_rows(root / "data/verilog.jsonl")}
    # The unchanged five-field core stores categories and kinds independently.
    # The rights ledger preserves their actual pairs, never a Cartesian product.
    covered, used = set(), set()
    for item in core._read_rows(root / "record-rights.jsonl"):
        core._keys(item, PUBLIC_RIGHTS_FIELDS, PUBLIC_RIGHTS_FIELDS, "Unexpected public rights fields")
        if _digest(item["rtl_id"]) != _digest(item["upstream_file_sha256"]):
            raise ValueError("Public rights must identify exact source bytes")
        _origin(item["upstream_url"], item["upstream_revision"], item["upstream_file"])
        key = (item["rtl_id"], core._label(item["source_category"]), item["record_kind"])
        record = records.get(item["rtl_id"])
        if (item["record_kind"] not in core.KINDS or key in covered or record is None
                or item["source_category"] not in record["source_categories"]
                or item["record_kind"] not in record["record_kinds"]):
            raise ValueError("Invalid or unused record-level evidence")
        ids = item["notice_ids"]
        if not isinstance(ids, list) or not ids or ids != sorted(set(ids)) or any(n not in notices for n in ids):
            raise ValueError("Missing record-level notices")
        for n in ids:
            if (notices[n]["upstream_url"], notices[n]["upstream_revision"]) != (item["upstream_url"], item["upstream_revision"]):
                raise ValueError("Notice revision differs from source revision")
        covered.add(key); used.update(ids)
    for identity, record in records.items():
        pairs = {(source, kind) for rtl_id, source, kind in covered if rtl_id == identity}
        if {source for source, _ in pairs} != set(record["source_categories"]) or {kind for _, kind in pairs} != set(record["record_kinds"]):
            raise ValueError("Record-level rights coverage is incomplete")
    if used != set(notices):
        raise ValueError("Packaged notice has no emitted record")
    allowed = BASE_FILES | {"data/" + n for n in core.PACKAGE_FILES} | {n["package_file"] for n in notices.values()} | {"historical-training-mixture.csv", "SOURCE_RECIPES.md"}
    if set(files) - allowed:
        raise ValueError("Unexpected package payload")
    return {"status": "verified", "counts": counts, "rights_records": len(covered), "notice_files": len(notices),
            "redistribution_evidence": "exact-file subset verified; other source payloads withheld", "published": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build", help="Prepare a new exact-file subset from reviewed private evidence")
    build.add_argument("--config", type=Path, required=True); build.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="Verify record-level rights, notices, code and checksums")
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_reviewed_release(args.config, args.output) if args.action == "build" else verify_reviewed_release(args.output)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (OSError, UnicodeError, KeyError, TypeError):
        print("Reviewed Verilog release failed: invalid input or inaccessible file", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Reviewed Verilog release failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
