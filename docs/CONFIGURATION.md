# Inputs, configuration and external dependencies

## PPA data and model compatibility

`scripts/ppa.py` requires an experiment JSON configuration and explicit dataset, model and output locations. Evaluation and prediction require an explicit split. A dataset directory contains JSONL files named after configured split labels, such as `train.jsonl`, `valid.jsonl` and `test.jsonl`. Each nonempty line is a JSON object.

The maintained multimetric formatter expects an RTL string under the configured `rtl_key`, a numeric clock period under `clock_key`, and finite numeric labels for the selected ordered metrics. Supported label aliases are `area_value`/`area`, `delay_value`/`delay`, `static_power_value`/`static_power`, `switching_power_value`/`switching_power` and `total_power_value`/`total_power`. Optional design/family identifiers support historical grouping attempts. Use units consistently across training labels, prompts and synthesis tools. The code does not convert between arbitrary library/tool unit conventions.

For example, the following is a **newly authored synthetic illustration with fabricated labels**, not a released research record or a measured circuit:

```json
{"rtl":"module example_buffer(input wire a, output wire y); assign y = a; endmodule","clock":1.0,"area":1.0,"delay":0.1,"total_power":0.001}
```

The formatter describes the clock in ns, formats selected targets in scientific notation in their configured order, tokenizes/truncates RTL, and appends target end-of-sequence tokens. Input length, target length, digit precision, sample count and split selection are experiment settings. A checkpoint used for inference must have been trained with compatible formatting and metric order.

Supply a local Hugging Face-style encoder-decoder model directory with configuration, complete weights and compatible tokenizer files supported by the retained T5Gemma workflow. The standalone backend loads a seq2seq model and decodes formatted numeric target text; an unrelated language-model checkpoint is not automatically a trained PPA predictor. The repository also preserves alternative OmniPred code with a custom numeric decoder/P10 tokenizer, but those architectures have a different checkpoint contract and are not interchangeable with the maintained full-seq2seq launcher. Historical adapter experiments additionally require their exact compatible base model. Model access and license requirements remain external.

The launcher validates requested split files and configuration compatibility. It does not require private research manifests or historical checkpoint identities. User-provided checksums can pin local inputs when explicitly configured; no historical artifact hash is shipped as a mandatory restriction. See the configuration example and launcher help for the supported checksum fields. Training requires all configured training/validation splits; standalone evaluation and prediction use only the requested split.

Split names contain letters, numbers, underscores or hyphens and omit `.jsonl`. Optional checksum settings map split names to SHA-256 values, and model filenames to their SHA-256 values:

```json
{"checksums":{"splits":{"train":"<64 hexadecimal SHA-256 characters>"},"model":{"model.safetensors":"<64 hexadecimal SHA-256 characters>"}}}
```

Replace the illustrative hash strings with hashes computed from your own files, or keep both mappings empty. A supplied hash must match. Only selected split checksums are verified; every supplied model-file checksum is verified. Complete sharded weights may be supplied with their Hugging Face index and all referenced shards. Dry-run checks the file layout and JSON metadata without loading model tensors; actual backend loading establishes tensor/tokenizer compatibility.

Output locations must be new and separate from datasets/checkpoints. Use a new directory for each experiment. Prediction and evaluation may create detailed local records from your supplied inputs; those output records are not part of this source release.

## Search benchmark and verification inputs

Population search requires an external benchmark root containing selected suites and each task's specifications, reference RTL when used, verification inputs and any manifest/sidecar files required by the chosen evaluator. EvolVE adapters require the corresponding external IC-RTL, VerilogEval or CVDP layout. Synthetic fixtures in this repository test interfaces and do not provide those benchmark suites.

Simulation and formal-check tools depend on the adapter. Install executable tools on `PATH` or provide their locations through the supported configuration. Selecting a pipeline that uses a model requires a compatible checkpoint and explicit inference settings. Selecting a synthesis pipeline requires its tool configuration and cell library. Missing external locations produce configuration errors rather than falling back to a research workstation installation.

The local EvolVE implementation includes idea-guided refinement and several evaluator adapters. Its generation model, prompts, score definitions, budgets and correctness checks must be specified when comparing runs. It is inspired by the EvolVE paper and is not certified as an exact reproduction. A simulation pass only certifies the tests executed by that run.

## Design Compiler / PrimeTime flow contract

The external commercial-flow directory must provide `resynthesize_dataset.py` and `synthesize_verilogeval_spec2rtl.py`. The retained adapter imports their existing APIs: `default_config`, `job_identity`, `job_dir`, `parse_dc_result`, `parse_pt_power`, `preflight`, `required_dc_outputs`, `required_pt_outputs`, `run_one_job`, `site_tool_environment`, and the latter module's `render_sdc`. A supplied tool-config JSON is loaded through the flow's `ToolConfig.from_json`.

Supply the cell-library Verilog model used for correctness checks and the synthesis/timing libraries required by that flow. Preserve constraints and units in your own configuration. This release provides neither the commercial executables nor their licenses nor a cell-library installation.

For the CVDP and VerilogEval EvolVE adapters, `paths.synthesis_flow`, `paths.synthesis_config` and `paths.cell_verilog` select that external flow, its JSON configuration and its simulation cell model. The IC-RTL adapter generates its own Tcl using configured tool-profile, database, synthesis-library and cell-model paths. Yosys-based EvolVE adapters additionally require explicit `paths.map_adders` and `paths.map_latches`. These are mapping sources from your own compatible installation, not bundled library payloads.

## Yosys/OpenROAD and PDK inputs

The retained synthesis templates support their original PDK conventions. Choose a template compatible with your externally installed library layout and supply its root explicitly with `--pdk_path`. Proxy preprocessing through Yosys also requires this location; raw RTL proxy inference does not. An optional `--dependency_root` can identify external helper dependencies. Do not infer that a different library or unit convention is equivalent merely because a tool launches. The bundled PDK payloads were excluded from this release. Check the selected template and its required Liberty/LEF/technology files before running a full synthesis search.

After valid external locations pass preflight, runtime Yosys preprocessing failures retain the historical fallback to parsed or original RTL. Inspect `used_yosys` and `fallback_reason` in the local preprocessing metadata; a fallback result does not establish that a netlist-trained model received normalized input.

The historical normalizer retains a `sky130` selection, but its referenced `ref.yosys.sky130.tcl` was not present in the source and is not supplied here. That selection cannot establish successful preprocessing with this release; use a supplied compatible template and inspect the preprocessing metadata.

## Secrets and dependencies

Generation API keys belong in the environment required by the selected backend, never in configuration committed to Git. The PPA dependency set provides training/model code; the experiment extra adds optional historical loss dependencies. GPU support depends on your PyTorch installation. The Docker recipe is an optional build recipe, not proof that every external EDA flow or proprietary tool has been validated in a container.
