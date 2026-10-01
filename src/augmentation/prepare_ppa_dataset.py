from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .dataset_io import DatasetRecord, load_dataset_records, save_records
from .synthesis import PpaSynthesizer


_WORKER_SYNTHESIZER: PpaSynthesizer | None = None
_WORKER_DEFAULT_CLOCK_PERIOD_NS: float = 0.01


@dataclass(frozen=True)
class WorkerConfig:
    work_root: str
    yosys_bin: str
    openroad_bin: str
    clock_period_ns: float
    timeout_seconds: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 0: synthesize data/pyranet_with_desc with Yosys/OpenROAD and "
            "write a PPA-annotated dataset."
        )
    )
    parser.add_argument("--input", default="data/pyranet_with_desc")
    parser.add_argument("--output", default="data/pyra_desc_yosys")
    parser.add_argument("--work-dir", default="outputs/augmention/step0_yosys")
    parser.add_argument("--yosys-bin", default="yosys")
    parser.add_argument("--openroad-bin", default="openroad")
    parser.add_argument("--clock-period-ns", type=float, default=0.01)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help=(
            "Number of parallel Yosys/OpenROAD synthesis workers. "
            "Use 1 for sequential execution."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_dataset_records(args.input, require_description=False)
    if args.limit is not None:
        records = records[: args.limit]

    worker_config = WorkerConfig(
        work_root=args.work_dir,
        yosys_bin=args.yosys_bin,
        openroad_bin=args.openroad_bin,
        clock_period_ns=args.clock_period_ns,
        timeout_seconds=args.timeout_seconds,
    )
    workers = max(1, args.workers)

    if workers == 1:
        _init_worker(worker_config)
        output_rows = [
            _synthesize_record(record)
            for record in tqdm(records, desc="Synthesizing baseline PPA")
        ]
    else:
        output_rows = _synthesize_records_parallel(records, worker_config, workers)

    output_path = save_records(output_rows, Path(args.output))
    print(f"Wrote PPA dataset to {output_path}")


def _synthesize_records_parallel(
    records: list[DatasetRecord], worker_config: WorkerConfig, workers: int
) -> list[dict]:
    output_rows: list[dict | None] = [None] * len(records)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(worker_config,),
    ) as executor:
        futures = {
            executor.submit(_synthesize_record, record): idx
            for idx, record in enumerate(records)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Synthesizing baseline PPA ({workers} workers)",
        ):
            idx = futures[future]
            output_rows[idx] = future.result()

    return [row for row in output_rows if row is not None]


def _init_worker(config: WorkerConfig) -> None:
    global _WORKER_SYNTHESIZER, _WORKER_DEFAULT_CLOCK_PERIOD_NS
    _WORKER_DEFAULT_CLOCK_PERIOD_NS = config.clock_period_ns
    _WORKER_SYNTHESIZER = PpaSynthesizer(
        work_root=config.work_root,
        yosys_bin=config.yosys_bin,
        openroad_bin=config.openroad_bin,
        clock_period_ns=config.clock_period_ns,
        timeout_seconds=config.timeout_seconds,
    )


def _synthesize_record(record: DatasetRecord) -> dict:
    if _WORKER_SYNTHESIZER is None:
        raise RuntimeError("Worker synthesizer was not initialized.")

    _WORKER_SYNTHESIZER.evaluator.clk_period = _clock_period_for_record(
        record.raw, _WORKER_DEFAULT_CLOCK_PERIOD_NS
    )

    result = _WORKER_SYNTHESIZER.synthesize_text(
        verilog=record.verilog,
        sample_id=record.sample_id,
        top_module=record.top_module,
        variant="baseline",
    )
    row = dict(record.raw)
    row.update(
        {
            "sample_id": record.sample_id,
            "verilog": record.verilog,
            "top_module": record.top_module,
            "ppa_success": result.success,
            "ppa_metrics": result.metrics.to_dict(),
            "synthesis": result.to_record(),
        }
    )
    if "description" in record.raw:
        row["description"] = record.description
    return row


def _clock_period_for_record(row: dict, default: float) -> float:
    clock = row.get("clock")
    if clock is None or clock == "":
        return default
    try:
        return float(clock)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
