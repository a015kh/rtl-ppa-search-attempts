# Upstream Verilog retrieval and preparation recipes

The historical training mixture and the public Verilog subset serve different
purposes. The [historical source table](DATASET_GENERATION.md#recovered-source-distribution)
describes measured rows used by the project. A redistributable subset contains
only records with verified provenance, applicable licenses and preserved notices.
Withheld sources remain visible in the historical aggregate table; their omission
from a release does not change what the model was trained on.

This repository provides upstream links and preparation interfaces for the
remaining sources. It does not bundle their datasets or claim that current
downloads recreate the original v6.3 training files, local augmentations, PPA
labels or evaluation splits. Benchmark origins are listed separately in the
[benchmark reference guide](BENCHMARK_REFERENCES.md).

## Offline source selection

The recipe writer uses the Python standard library and works from any working
directory when invoked through the script's location. It makes no network
requests, downloads no source payloads and writes no files in listing mode.

```bash
python scripts/retrieve_public_verilog_sources.py list
python scripts/retrieve_public_verilog_sources.py write \
  --source metrex --source pyranet \
  --output-dir /path/to/new-source-recipes
```

`write` requires an explicit source selection and a new output directory outside
the checkout. It writes `sources.json` and a human-readable `README.md`, containing
public URLs, revision status and rights-review limitations. No automatic bulk
retrieval is offered. Acquire upstream files separately after inspecting the
official file inventory; record their resolved revision and SHA-256 locally.

## Source references and unresolved obligations

Recorded dataset revisions below are available evidence for the imports. A
`main` reference is mutable and is not a recovered historical import revision.
None of these collection-level declarations is a grant from every original RTL
author. Unknown rights are an evidence gap, not a finding that every record is
prohibited.

| Recipe key | Official source / available revision | Collection declaration and remaining review |
|---|---|---|
| `mg-verilog` | [MG-Verilog](https://huggingface.co/datasets/GaTech-EIC/MG-Verilog); historical import revision not recorded | The [MIT-labelled card](https://huggingface.co/datasets/GaTech-EIC/MG-Verilog/blob/main/README.md) says individual collected sample licenses cannot be verified. Recover original source notices. |
| `metrex` | [MetRex at `b9cb51395445da1586e5b0b3ba6f7c91851823c5`](https://huggingface.co/datasets/scale-lab/MetRex/tree/b9cb51395445da1586e5b0b3ba6f7c91851823c5) | The pinned card declares BSD-3-Clause; a complete copyright-bearing notice was not recovered in this review. Local augmentations are separate artifacts. |
| `openrtlset` | [OpenRTLSet](https://huggingface.co/datasets/ESCAD/OpenRTLSet); historical import revision not recorded | The collection declares Apache-2.0, while records expose `Repo_url` and `lic_name` for multiple underlying licenses. Review each original repository/file and retain its notices. |
| `pyranet` | [PyraNet-Verilog at `87b56d5b3d014eb424200a96bc5f37feb6c8612f`](https://huggingface.co/datasets/bnadimi/PyraNet-Verilog/tree/87b56d5b3d014eb424200a96bc5f37feb6c8612f) | The card declares CC-BY-NC-SA-4.0 and says individual scraped sample licenses cannot be confirmed. [Its terms](https://creativecommons.org/licenses/by-nc-sa/4.0/) require attribution, noncommercial use and applicable share-alike conditions; they do not prohibit all public sharing. |
| `rtlcoder` | [RTL-Coder's dataset directory](https://github.com/hkust-zhiyao/RTL-Coder/tree/main/dataset); historical import revision not recorded | An explicit redistribution grant for the imported data was not recovered. Do not substitute a model license or downstream mirror's license. |
| `verilogos` | [VeriLogos Augmented at `caf3680677ef176445a7e7f3f86cb3e0c8124b5c`](https://huggingface.co/datasets/97kjmin/VeriLogos_Augmented_Dataset/tree/caf3680677ef176445a7e7f3f86cb3e0c8124b5c) | BSD-3-Clause is declared; the [upstream repository](https://github.com/97kjmin/VeriLogos) also discusses scraped-code copyright concerns. Anonymization or augmentation alone does not establish individual record rights. |

Keep original copyright and license notices in each selected RTL file. The
research code's license does not relicense upstream data. Removing PPA fields
addresses the data release boundary; it does not remove source copyright
obligations. Generated variants require the same provenance review as their
seeds, including any applicable derivative conditions.

## Prepare locally supplied source records

For a source-only export, prepare local JSONL records compatible with the
[Verilog-only exporter](VERILOG_ONLY_RELEASE.md): use `rtl` for the exact original
source string, `source_dataset` for a configured source key and `record_kind` for
the appropriate provenance (`original`, `augmentation` or `test`). Supply an
explicit input split in the exporter configuration. Preserve the source record's
identity, file location, resolved revision and notices in your own provenance
ledger; do not add private locations or measurement fields to a public package.
Do not call an upstream augmented collection a locally generated augmentation.

Start with [the deny-by-default configuration](../configs/verilog-release.example.json).
Select records only after checking their applicable rights and retaining the
necessary notices. Follow the exporter's configuration and review requirements;
changing an inclusion flag is not evidence of permission.

```bash
python scripts/prepare_verilog_release.py preflight --config /path/to/verilog-release.json
python scripts/prepare_verilog_release.py build \
  --config /path/to/verilog-release.json --output /path/to/new-verilog-package
python scripts/prepare_verilog_release.py verify --output /path/to/new-verilog-package
```

The [measured dataset assembler](DATASET_GENERATION.md) is a separate workflow. It
requires compatible, already measured inputs and provenance metadata; upstream
RTL acquisition does not supply those labels. Neither workflow runs synthesis,
training, optimization search or publication.

Historical splits must not be presented as a newly verified independent
benchmark partition. Recovered Pyra augmentation records used an incorrect
seed association in historical lineage assignment. Reconstruct the actual seed
relationships before constructing any new family-disjoint split; this recipe
does not silently repair the retained files or change reported model metrics.
