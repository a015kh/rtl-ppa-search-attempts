# An offline synthetic walkthrough

This small example exercises the real dataset builder with entirely invented RTL,
PPA-shaped labels and a local source catalog. It needs Python 3.11 or newer and
only the standard library. It needs no dataset, checkpoint, GPU, API credentials,
network connection, simulator or synthesis tool.

From a source checkout, choose a new output directory outside the repository:

```bash
python scripts/synthetic_walkthrough.py --output /tmp/rtl-ppa-synthetic-example
```

Use an equivalent new directory on Windows. The script also works from another
working directory when invoked with its absolute path. After installing the
package, the same example is available as:

```bash
rtl-ppa-walkthrough --output /tmp/rtl-ppa-synthetic-example
```

The successful command exits with status zero and prints a JSON summary with
`"status": "passed"`. It writes:

| Location | Contents |
|---|---|
| `inputs/` | Fabricated base/augmentation records, test-result JSON, seed rows and local catalog |
| `test-dataset.json` | Explicit test-result conversion configuration |
| `prepared-tests/` | Converted test JSONL and preparation audit |
| `dataset.toml` | Explicit configuration, with paths relative to this file |
| `dataset/` | Assembled JSONL shards and builder metadata/manifest |
| `summary.json` | Expected counts, lineage checks and verified output checksums |

The assembled dataset has six training base rows, two validation base rows,
six training augmentation rows, two validation augmentation rows and two test
rows: **18 rows total**. A ninth input base row repeats an earlier identity with
a different label; the original label wins under the builder's existing priority.
There are six training lineages, two validation lineages and zero overlap.

The walkthrough first converts two fabricated test results with the preserved
`legacy_result_json` policy. It then calls the maintained dataset preflight,
build and verification functions.
It checks split counts, duplicate precedence, seed/augmentation lineage grouping,
unchanged numeric labels, test identity and manifest checksums. You can inspect
the generated configuration and rerun the supported verifier:

```bash
python scripts/generate_dataset.py verify --config /tmp/rtl-ppa-synthetic-example/dataset.toml
```

An existing output directory is rejected; repeated walkthroughs should use a new
directory. A failed run leaves its partial directory available for inspection.
Generated files are your local demo artifacts and do not belong in the release.

The fixture labels have no physical meaning. The example checks dataset plumbing;
it does not run learned inference, model training, correctness verification or
optimization search. Actual model workflows still require compatible artifacts
and their declared dependencies. These fixtures are unrelated to the recorded
v6.3 results. See [the workflow map](WORKFLOWS.md) for the next task to run.

This example does not exercise DC/PT report-tree filtering or prove that test
records are functionally correct or independent of training. The separate
[test preparation guide](TEST_DATASET_PREPARATION.md) describes those policy
interfaces and the historical provenance limits.
