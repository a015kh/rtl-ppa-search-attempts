from __future__ import annotations

import argparse
import time
from pathlib import Path

from .llm_client import DEFAULT_QWEN_MODEL, AugmentationLLM
from .models import AugmentationConfig
from .pipeline import DataAugmentationEngine
from .synthesis import PpaSynthesizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the EoH-style Verilog data augmentation pipeline."
    )
    parser.add_argument("--input", default="data/pyra_desc_yosys")
    parser.add_argument("--output", default="data/pyra_desc_augmented")
    parser.add_argument("--work-dir", default="outputs/augmention/eoh_data_aug")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["area", "power", "delay"],
        choices=["area", "power", "delay"],
    )
    parser.add_argument(
        "--target-selection-mode",
        choices=["all", "random"],
        default="all",
        help=(
            "Use 'all' to run every target in --targets for each sample, or "
            "'random' to choose one target from --targets per sample."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Optional seed for reproducible --target-selection-mode random choices.",
        default=42,
    )
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--candidates-per-generation", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Maximum number of in-flight augmentation candidates.",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=64,
        help="Maximum concurrent Qwen/vLLM generation or feedback requests.",
    )
    parser.add_argument(
        "--synthesis-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent Yosys/OpenROAD synthesis jobs.",
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Optional index to start processing from in the input dataset.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--api-backend",
        default="local-vllm",
        choices=["local-vllm", "vllm", "openai", "openrouter", "deepseek", "gemini"],
        help=(
            "Use local-vllm for an embedded vLLM engine, or vllm for a separate "
            "OpenAI-compatible vLLM server."
        ),
    )
    parser.add_argument("--model-name", default=DEFAULT_QWEN_MODEL)
    parser.add_argument(
        "--api-base-url",
        help=(
            "OpenAI-compatible API base URL. Used by --api-backend vllm and "
            "other hosted API backends."
        ),
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--local-vllm-batch-size", type=int, default=64)
    parser.add_argument("--local-vllm-batch-timeout-ms", type=int, default=50)
    parser.add_argument("--local-vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--local-vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--local-vllm-max-model-len", type=int)
    parser.add_argument("--local-vllm-max-num-batched-tokens", type=int)
    parser.add_argument("--local-vllm-max-num-seqs", type=int, default=64)
    parser.add_argument("--local-vllm-dtype", default="auto")
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code when loading the local vLLM model/tokenizer.",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Disable vLLM prefix caching for the embedded local-vllm backend.",
    )
    parser.add_argument("--yosys-bin", default="yosys")
    parser.add_argument("--openroad-bin", default="openroad")
    parser.add_argument("--clock-period-ns", type=float, default=0.01)
    parser.add_argument("--synth-timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=47.5,
        help=(
            "Stop scheduling new candidates before the process wall-time limit. "
            "Use <= 0 to disable."
        ),
    )
    parser.add_argument(
        "--shutdown-margin-minutes",
        type=float,
        default=15.0,
        help="Reserved time before --max-runtime-hours for graceful summary writes.",
    )
    parser.add_argument(
        "--synthesize-missing-baseline",
        action="store_true",
        help=(
            "If input rows do not already contain PPA metrics, synthesize the original "
            "Verilog before augmentation. The expected default is to run "
            "prepare_ppa_dataset.py first."
        ),
    )
    parser.add_argument(
        "--skip-missing-synthesis-report",
        "--skip-missing-baseline",
        action="store_true",
        dest="skip_missing_synthesis_report",
        help=(
            "Skip input records whose baseline synthesis/PPA metrics are missing "
            "instead of aborting the whole augmentation run."
        ),
    )
    parser.add_argument(
        "--max-summary-errors",
        type=int,
        default=2000,
        help=(
            "Maximum detailed error records to keep in evolution_summary.json. "
            "Use a negative value to keep every error."
        ),
    )
    return parser


def main() -> None:
    process_started_monotonic = time.monotonic()
    args = build_parser().parse_args()
    max_runtime_seconds = (
        None if args.max_runtime_hours <= 0 else args.max_runtime_hours * 3600
    )
    config = AugmentationConfig(
        input_dataset=Path(args.input),
        output_dir=Path(args.output),
        work_dir=Path(args.work_dir),
        targets=args.targets,
        target_selection_mode=args.target_selection_mode,
        random_seed=args.random_seed,
        num_generations=args.num_generations,
        candidates_per_generation=args.candidates_per_generation,
        concurrency=max(1, args.concurrency),
        llm_concurrency=max(1, args.llm_concurrency),
        synthesis_concurrency=max(1, args.synthesis_concurrency),
        start=args.start,
        limit=args.limit,
        resume=not args.no_resume,
        model_name=args.model_name,
        api_backend=args.api_backend,
        api_base_url=args.api_base_url,
        request_timeout_seconds=args.request_timeout_seconds,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        local_vllm_batch_size=max(1, args.local_vllm_batch_size),
        local_vllm_batch_timeout_ms=max(0, args.local_vllm_batch_timeout_ms),
        local_vllm_tensor_parallel_size=max(1, args.local_vllm_tensor_parallel_size),
        local_vllm_gpu_memory_utilization=args.local_vllm_gpu_memory_utilization,
        local_vllm_max_model_len=args.local_vllm_max_model_len,
        local_vllm_max_num_batched_tokens=args.local_vllm_max_num_batched_tokens,
        local_vllm_max_num_seqs=args.local_vllm_max_num_seqs,
        local_vllm_dtype=args.local_vllm_dtype,
        trust_remote_code=not args.no_trust_remote_code,
        enable_prefix_caching=not args.disable_prefix_caching,
        yosys_bin=args.yosys_bin,
        openroad_bin=args.openroad_bin,
        clock_period_ns=args.clock_period_ns,
        synth_timeout_seconds=args.synth_timeout_seconds,
        synthesize_missing_baseline=args.synthesize_missing_baseline,
        skip_missing_synthesis_report=args.skip_missing_synthesis_report,
        max_summary_errors=args.max_summary_errors,
        max_runtime_seconds=max_runtime_seconds,
        shutdown_margin_seconds=max(0.0, args.shutdown_margin_minutes * 60),
    )
    llm = AugmentationLLM(
        model_name=config.model_name,
        api_backend=config.api_backend,
        api_base_url=config.api_base_url,
        request_timeout_seconds=config.request_timeout_seconds,
        local_vllm_batch_size=config.local_vllm_batch_size,
        local_vllm_batch_timeout_ms=config.local_vllm_batch_timeout_ms,
        local_vllm_tensor_parallel_size=config.local_vllm_tensor_parallel_size,
        local_vllm_gpu_memory_utilization=config.local_vllm_gpu_memory_utilization,
        local_vllm_max_model_len=config.local_vllm_max_model_len,
        local_vllm_max_num_batched_tokens=config.local_vllm_max_num_batched_tokens,
        local_vllm_max_num_seqs=config.local_vllm_max_num_seqs,
        local_vllm_dtype=config.local_vllm_dtype,
        trust_remote_code=config.trust_remote_code,
        enable_prefix_caching=config.enable_prefix_caching,
    )
    synthesizer = PpaSynthesizer(
        work_root=config.work_dir,
        yosys_bin=config.yosys_bin,
        openroad_bin=config.openroad_bin,
        clock_period_ns=config.clock_period_ns,
        timeout_seconds=config.synth_timeout_seconds,
    )
    output_path = DataAugmentationEngine(
        config=config,
        llm=llm,
        synthesizer=synthesizer,
        started_monotonic=process_started_monotonic,
    ).run()
    print(f"Wrote augmentation dataset to {output_path}")
    print(
        "Wrote sample design-name map to "
        f"{config.work_dir / 'sample_design_names.json'}"
    )


if __name__ == "__main__":
    main()
