# Verilog-only companion dataset

The companion dataset is intended to show the code mixture used by this work
without distributing the retained PPA measurements. It is a separate data
package, not a payload bundled with this source repository. The measured v6.3
dataset and trained checkpoint remain separate private artifacts.

## Release boundary

Keep only the exact original `rtl` string and a small, explicit set of metadata:

| Field | Meaning |
|---|---|
| `rtl_id` | SHA-256 of the UTF-8 Verilog text, without normalization |
| `rtl` | Original Verilog/SystemVerilog source, including its existing notices |
| `source_categories` | Public source attribution categories; a string can match more than one category |
| `record_kinds` | Base, locally generated or evaluation provenance |
| `original_splits` | Membership in the retained named files, not a newly constructed partition |

Do not copy PPA targets or predictions, clock-target metadata, SDC, synthesis or
timing reports, tool output, library names or paths, private design identifiers,
source-artifact locations, private input checksums, checkpoints or private
archive manifests. RTL ports and behavioral timing constructs are part of the
program; they are not the removed measurement metadata.

Exact identical source strings are stored once across all supplied files.
Membership metadata is combined. Different whitespace, comments or code remain
different strings; no code transformation or semantic-equivalence claim is
made. Existing copyright notices are preserved. Suspicious source text is
withheld and counted rather than silently edited.

The source-mixture tables report both original input rows and distinct source
strings. Six clock measurements of one implementation are not six different
designs. Source-category unique counts can overlap, so their sum need not equal
the global unique count. Use original row counts to understand the historical
training mixture and unique strings to understand the source-only package.
Locally generated augmentations inherit seed-source attribution; they are not
claimed to be original upstream records. Original split memberships are not a
guarantee of independence or a ready-to-use benchmark split.

## Prepare with explicit source policies

The exporter uses only the Python standard library. It reads an explicit JSON
configuration with named inputs, optional input checksums and a source registry.
Paths are relative to the configuration file unless absolute. Source policies
control base, generated and evaluation inclusion separately, defaulting to
withholding. A missing or unknown source never receives an assumed license.

Start from [the example configuration](../configs/verilog-release.example.json).
All its inclusion flags are false. Replace its input paths and decide the
source policies explicitly after checking the original notices and permissions.
Null upstream or license URLs mean unavailable provenance, not a rights grant.

```bash
python scripts/prepare_verilog_release.py preflight --config /path/to/verilog-release.json
python scripts/prepare_verilog_release.py build --config /path/to/verilog-release.json --output /path/to/new-verilog-package
python scripts/prepare_verilog_release.py verify --output /path/to/new-verilog-package
```

The installed alias is `rtl-ppa-verilog-release`. Select a new output directory
outside the checkout and all input locations. The exporter does not download
upstream data, synthesize RTL, compute labels, contact an API or publish files.

The package contains `verilog.jsonl`, a public manifest with output checksums,
`data-source-distribution.csv` and `split-distribution.csv`. Input paths and
measured-input file hashes are not included in the public manifest. Add a dataset
card and applicable upstream license/notice texts before publication.

## Rights and provenance

Public availability is separate from permission to redistribute a collected
record. Source-registry inclusion flags authorize local preparation; they are
not proof that all underlying licenses have been resolved. A package containing
an unverified or conditional source remains **staging-only**. Mark a source
cleared only after verifying applicable record-level rights and preserving the
required notices. Do not apply this code repository's license to the dataset.

For example, [PyraNet-Verilog](https://huggingface.co/datasets/bnadimi/PyraNet-Verilog/blob/87b56d5b3d014eb424200a96bc5f37feb6c8612f/README.md) declares CC-BY-NC-SA-4.0 and states that individual
collected sample licenses could not be confirmed. [OpenRTLSet/ESCAD](https://huggingface.co/datasets/ESCAD/OpenRTLSet) provides
per-record repository/license metadata with multiple underlying licenses.
Collection-level license labels therefore cannot replace record-level rights.
Local generated derivatives also require their seed-source obligations to be
considered. Benchmark citations establish origin, not redistribution permission.

The historical [source distribution](DATASET_GENERATION.md#recovered-source-distribution)
remains available even when a source must be withheld from a public data package.
The [benchmark reference guide](BENCHMARK_REFERENCES.md) identifies the retained
RTL-OPT and Pluto subsets. The exporter does not reconstruct missing licenses,
upstream commits, formal-equivalence proofs or the exact measured dataset from
this source-only package. It cannot reproduce the reported PPA evaluation or
train the retained PPA model without compatible privately supplied labels.
