# Test-set conversion and measured-result filtering

The release includes an offline stage for turning **supplied completed synthesis measurements** into test JSONL. It recovers the local DC/PT report parsers, synthesis-result filtering and historical result-to-row conversion. It does not download benchmarks, generate RTL, launch synthesis, establish functional correctness or recreate the seven retained v6.3 test sets from public sources.

```bash
python scripts/prepare_test_dataset.py --help
python scripts/prepare_test_dataset.py preflight --config /path/to/test-config.json
python scripts/prepare_test_dataset.py build --config /path/to/test-config.json
```

After installation, `rtl-ppa-test-dataset` exposes the same interface. The implementation uses only Python's standard library and packaged report-formatting helpers. All input/output locations are explicit. Relative paths in the configuration resolve from its directory; paths in a report-jobs JSON resolve from that JSON's directory.

Use [the example configuration](../configs/test-dataset.example.json), retaining only the shard entries needed for your inputs. `output_root` must be a new directory outside the package/source checkout and separate from all input/report locations. The output contains the requested JSONL files and `audit.json` with accepted/skipped counts, reasons, selected power-source counts and output SHA-256s. It contains no copied netlists or original path metadata. The JSONL necessarily contains user-supplied RTL, constraints and measurements: treat these generated artifacts separately from the public code repository.

## Named historical policies

| Policy | Supplied input | Preserved behavior |
|---|---|---|
| `legacy_result_json` | Ordered JSON files containing `Name`, `RTL`, `Syn` design records | Flatten every synthesis point using historical `extract_data_v3`; derive static/dynamic numeric targets from the last numeric token in their explanation strings. No extra quality, zero-label, deduplication or correctness filter. |
| `dcpt_inputs_reports` | Explicit jobs JSON pointing to existing DC/PT report trees | Recover `result_gen_dcpt.py` report acceptance: required SDC, area/QoR/timing reports and netlist; nonmissing, nonzero area; power, WNS, TNS and delay must parse. Prefer PT power; DC fallback only with explicitly selected `mixed`. Flatten accepted points with the same historical converter. |
| `resynthesis_success` | Source JSONL, read-only SQLite manifest, completed report work root | Recover `export_final_jsonls.py`'s default success-only export: skip unmapped source rows, jobs whose DC or PT status is not `success`, and report-collection errors. Parse replacement DC/PT labels and emit the historical compact no-netlist schema. Zero area and zero delay remain allowed, unlike the other report policy. |

These are different recorded policies, not interchangeable definitions of valid PPA data. Select the policy matching the input's actual history. No uniform new outlier threshold is applied. Negative or zero synthesis times are not repaired. A success marker or a parseable synthesis report does not establish simulation/formal correctness.

### Historical result JSON

Minimal configuration:

```json
{
  "output_root": "/path/to/new-test-dataset",
  "shards": [{
    "name": "test.jsonl",
    "policy": "legacy_result_json",
    "inputs": ["/path/to/completed-results.json"]
  }]
}
```

Each input is a JSON list of design objects. Each object has `Name` (string), `RTL` (string) and `Syn` (list). Compatible synthesis points carry `SDC`, `Has Clock`, `Clock`, `Total Area`, `Delay`, `WNS`, `TNS`, `Total Power`, `synth`, `area`, `delay`, `static_power`, `switching_power` and `synthesis_time`. For example, explanation strings ending in `0.10 mW.` and `0.15 mW.` yield numeric static/dynamic labels `0.10` and `0.15`. An explanation with no number fails conversion rather than silently inventing a target.

Inputs are concatenated in their explicit order; every `Syn` point is retained, preserving the historical clock-point expansion. This covers the conversion used by the earlier VerilogEval, RealBench and optimization-result helpers without relying on their hardcoded directories/file lists. It does not prove which of those helpers originally produced a particular retained shard. The legacy converter copies only its historical field map and adds the design name/RTL and extracted power targets; extra netlist, candidate or private-path fields are excluded.

### Existing raw report trees

`dcpt_inputs_reports` additionally requires an explicit `power_source`, either `primetime` or `mixed`. Its `jobs` file is a list of objects with the following shape; repeat a job entry for each measured clock/constraint point:

```json
[
  {
    "name": "example",
    "rtl": "/path/to/example.sv",
    "top_module": "example",
    "dc_tcl": "/path/to/design_dc.tcl",
    "sdc": "/path/to/design.sdc",
    "dc_report_dir": "/path/to/completed-dc-reports",
    "pt_report_dir": "/path/to/completed-pt-reports",
    "command_log": "/path/to/command.log"
  }
]
```

The parser uses the DC Tcl top when available, then the supplied top and existing report stems, matching the recovered selection order. DC files are `<stem>_area.log`, `_qor.log`, `_timing.log`, `_cell.log`, `_power.log` and `_syn.v`. PT power is `<stem>.report_power.rpt`, falling back to the supplied top's filename. `_cell.log` is optional under the historical policy; absent cell explanations may subsequently fail the unified builder's stricter complete-row validation.

`command_log` is optional; a missing start/end timestamp yields `null` synthesis time. Unlike the old command's accidental abort on some malformed reports, the release records those points under `report_parse_or_conversion_error`. Normal rejection reasons include `zero_area`, `missing_report_stem`, `missing_required_report_or_netlist`, `missing_power_report` and `missing_required_numeric_result`. These counts make filtering inspectable; the audit does not contain per-design log contents or identifiers.

The recovered DC parser takes numeric cell area and first data-arrival delay from the historical report patterns. PT power is converted to **mW** using its declared unit; missing units retain the historical **W** default. Unknown units retain that same historical default. The `mixed` DC fallback preserves the older last-line numeric total-power extraction and its explanation-unit assumptions; it does not introduce a corrected unit model. Area/time units and the measurement library/constraints must be recorded by the separate upstream measurement workflow. No cell-library location or synthesis installation is selected here.

### Completed resynthesis manifest

`resynthesis_success` requires explicit `input`, `manifest` and `work_root` paths. The SQLite `source_rows` table maps `(shard, line_number)` to `job_id`; `jobs` supplies `dc_status`, `pt_status`, `top_module`, `sdc`, `rtl`, `design_name` and `dc_elapsed`. The source shard name is the input JSONL filename. Existing report locations follow the recovered layout:

```text
<work_root>/jobs/<first-two-job-id-characters>/<job_id>/dc/report/<top>_area.log
<work_root>/jobs/<first-two-job-id-characters>/<job_id>/dc/report/<top>_qor.log
<work_root>/jobs/<first-two-job-id-characters>/<job_id>/dc/report/<top>_timing.log
<work_root>/jobs/<first-two-job-id-characters>/<job_id>/dc/report/<top>_power.log
<work_root>/jobs/<first-two-job-id-characters>/<job_id>/pt/report/<top>.report_power.rpt
```

The optional cell report uses the same DC report directory. This parser preserves the resynthesis special case `No paths.` as zero delay and formats its explanatory text. PT labels/explanations are normalized to mW. The manifest is opened read-only; no source file or report is rewritten. `input_designs` is `null` in this policy's audit because it counts source measurement rows, not distinct designs. The old `--include-failed` mode that could fall back to stale input labels is deliberately not exposed by this success-only stage.

## Connection to dataset assembly and reproduction limits

Point `tests.sources` in the [unified dataset configuration](DATASET_GENERATION.md) at these generated shards. The unified builder adds test provenance fields and validates complete labels/reports; it does not further invent a test-selection policy or resplit these files. Its training/validation lineage assignment is separate.

The original raw-report parser/filter, JSONL success exporter and result converter are now available as the named offline stages above. The historical `filter_testing.py` **port-match plus VerilogEval simulation overlap-removal workflow remains excluded**. It was a separate training-pool contamination check, not filtering of failed estimator predictions. Reproducing it requires separately supplied benchmark references/testbenches and simulation dependencies, and its exact use for these retained test sets was not established. This release makes no benchmark independence, equivalence or leakage-removal claim.

Exact historical commands/revisions creating the seven retained v6.3 shards were not recorded. The stages can prepare compatible user-supplied measured test data, but cannot guarantee identical retained v6.3 records from current benchmark downloads. Saved prediction-evaluation failed decodes are another separate issue; see [recorded v6.3 results](V63_RESULTS.md).
