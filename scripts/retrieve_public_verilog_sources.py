#!/usr/bin/env python3
"""Write explicit upstream retrieval recipes, without downloading source payloads.

The collection-level license descriptions are evidence pointers, not grants to
redistribute every collected record. This script needs only the standard library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def hf_recipe(repo: str, revision: str | None) -> dict[str, Any]:
    """Keep a recorded dataset pin distinct from a mutable upstream reference."""
    reference = revision or "main"
    return {
        "upstream_url": f"https://huggingface.co/datasets/{repo}",
        "card_url": f"https://huggingface.co/datasets/{repo}/blob/{reference}/README.md",
        "file_inventory_url": f"https://huggingface.co/api/datasets/{repo}/tree/{reference}",
        "dataset_revision": revision,
        "revision_kind": "recorded-dataset-pin" if revision else "mutable-current-reference",
    }


SOURCES: dict[str, dict[str, Any]] = {
    "mg-verilog": {
        "display_name": "MG-Verilog",
        **hf_recipe("GaTech-EIC/MG-Verilog", None),
        "repository_url": "https://github.com/GATECH-EIC/mg-verilog",
        "historical_import_revision": None,
        "collection_license_declaration": "MIT",
        "rights_review": "record-rights-unverified",
        "review_required": [
            "The upstream card says licenses of individually collected online samples cannot be verified.",
            "Recover original repository/file notices; the collection declaration does not replace them.",
        ],
    },
    "metrex": {
        "display_name": "MetRex",
        **hf_recipe("scale-lab/MetRex", "b9cb51395445da1586e5b0b3ba6f7c91851823c5"),
        "repository_url": "https://github.com/scale-lab/MetRex",
        "historical_import_revision": "b9cb51395445da1586e5b0b3ba6f7c91851823c5",
        "collection_license_declaration": "BSD-3-Clause",
        "rights_review": "record-rights-unverified",
        "review_required": [
            "The recorded card declares BSD-3-Clause, but a complete copyright-bearing notice was not recovered.",
            "Recover applicable original notices; do not invent a copyright holder.",
            "Local augmentations are separate from the upstream dataset and require their seed provenance.",
        ],
    },
    "openrtlset": {
        "display_name": "OpenRTLSet",
        **hf_recipe("ESCAD/OpenRTLSet", None),
        "repository_url": "https://github.com/UIUC-ChenLab/OpenRTLSet",
        "historical_import_revision": None,
        "collection_license_declaration": "Apache-2.0",
        "rights_review": "record-rights-unverified",
        "review_required": [
            "Records include Repo_url and lic_name with multiple underlying licenses.",
            "Review the original file/repository license and preserve its notices per record.",
            "The collection's Apache-2.0 declaration is not a replacement for those licenses.",
        ],
    },
    "pyranet": {
        "display_name": "PyraNet-Verilog",
        **hf_recipe("bnadimi/PyraNet-Verilog", "87b56d5b3d014eb424200a96bc5f37feb6c8612f"),
        "repository_url": None,
        "historical_import_revision": "87b56d5b3d014eb424200a96bc5f37feb6c8612f",
        "collection_license_declaration": "CC-BY-NC-SA-4.0",
        "rights_review": "record-rights-unverified",
        "license_terms_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "review_required": [
            "The upstream card says licenses of individual scraped samples cannot be confirmed.",
            "Consider attribution, noncommercial and applicable share-alike conditions alongside original file rights.",
            "Generated modifications do not automatically remove seed-source obligations.",
        ],
    },
    "rtlcoder": {
        "display_name": "RTL-Coder",
        "upstream_url": "https://github.com/hkust-zhiyao/RTL-Coder/tree/main/dataset",
        "repository_url": "https://github.com/hkust-zhiyao/RTL-Coder",
        "card_url": "https://github.com/hkust-zhiyao/RTL-Coder/blob/main/README.md",
        "file_inventory_url": "https://api.github.com/repos/hkust-zhiyao/RTL-Coder/contents/dataset",
        "dataset_revision": None,
        "revision_kind": "mutable-current-reference",
        "historical_import_revision": None,
        "collection_license_declaration": None,
        "rights_review": "license-grant-not-recovered",
        "review_required": [
            "No explicit redistribution grant was recovered for the imported dataset.",
            "A model license or downstream mirror license is not evidence of permission for these records.",
        ],
    },
    "verilogos": {
        "display_name": "VeriLogos Augmented",
        **hf_recipe("97kjmin/VeriLogos_Augmented_Dataset", "caf3680677ef176445a7e7f3f86cb3e0c8124b5c"),
        "repository_url": "https://github.com/97kjmin/VeriLogos",
        "historical_import_revision": "caf3680677ef176445a7e7f3f86cb3e0c8124b5c",
        "collection_license_declaration": "BSD-3-Clause",
        "rights_review": "record-rights-unverified",
        "review_required": [
            "Preserve the complete BSD-3-Clause notice and applicable original notices.",
            "The upstream repository discusses scraped-code copyright concerns; anonymization alone does not clear rights.",
        ],
    },
}


def recipe_manifest(names: list[str]) -> dict[str, Any]:
    if not names:
        raise ValueError("Select at least one named upstream source")
    unknown = sorted(set(names) - SOURCES.keys())
    if unknown:
        raise ValueError("Unknown upstream source: " + ", ".join(unknown))
    return {
        "schema_version": 1,
        "content_kind": "upstream-retrieval-recipes-only",
        "contains_rtl": False,
        "contains_ppa": False,
        "redistribution_permission_established": False,
        "recreates_historical_dataset": False,
        "sources": [{"source_key": name, **SOURCES[name]} for name in sorted(set(names))],
    }


def write_recipes(names: list[str], output_dir: Path | str) -> dict[str, Any]:
    manifest = recipe_manifest(names)
    output = Path(output_dir).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[1]
    if output == checkout or output.is_relative_to(checkout):
        raise ValueError("Recipe output must be outside the source checkout")
    if output.exists():
        raise ValueError("Recipe output must be a new directory; existing files are never overwritten")
    output.mkdir(parents=True, exist_ok=False)
    (output / "sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Selected upstream source recipes", "",
        "This directory contains metadata and links only. No Verilog, measurements or model artifacts were downloaded.",
        "The collection license labels do not establish permission to redistribute each original design.",
        "Recorded dataset pins and mutable current references are identified separately in sources.json.", "",
        "For each selected source, inspect the official card, file inventory and original notices.",
        "Acquire files separately into a local working directory, record the resolved revision and SHA-256,",
        "and preserve source-to-record provenance before preparing a source-only export.", "",
        "Current downloads do not recreate the historical measured v6.3 dataset or its locally generated variants.", "",
    ]
    for source in manifest["sources"]:
        lines += [f"## {source['display_name']}", "", f"Upstream: {source['upstream_url']}", "",
                  f"Reference kind: {source['revision_kind']}", ""]
        lines += [f"- {note}" for note in source["review_required"]] + [""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List all available upstream metadata without network access or file writes")
    writer = commands.add_parser("write", help="Write explicitly selected retrieval links/metadata; never download datasets")
    writer.add_argument("--source", action="append", choices=sorted(SOURCES), required=True,
                        help="Named upstream to include; repeat for additional sources")
    writer.add_argument("--output-dir", type=Path, required=True, help="New directory outside the source checkout")
    args = parser.parse_args(argv)
    try:
        result = recipe_manifest(list(SOURCES)) if args.command == "list" else write_recipes(args.source, args.output_dir)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
