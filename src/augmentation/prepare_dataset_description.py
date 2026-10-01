from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .dataset_io import DatasetRecord, load_dataset_records, save_records
from .llm_client import DEFAULT_QWEN_MODEL, AugmentationLLM


@dataclass
class ModuleGroup:
    key: str
    top_module: str
    verilog: str
    records: list[DatasetRecord]
    description: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate missing text descriptions for Verilog dataset rows. "
            "Rows with the same top module and Verilog reuse one generated description, "
            "which is useful when the same module is synthesized under different clocks."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cache-file",
        type=Path,
        help=(
            "JSON cache for generated descriptions. Defaults to "
            "<output>/description_cache.json."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--api-backend",
        default="local-vllm",
        choices=["local-vllm", "vllm", "openai", "openrouter", "deepseek", "gemini"],
    )
    parser.add_argument("--model-name", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--api-base-url")
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--local-vllm-batch-size", type=int, default=64)
    parser.add_argument("--local-vllm-batch-timeout-ms", type=int, default=50)
    parser.add_argument("--local-vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--local-vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--local-vllm-max-model-len", type=int)
    parser.add_argument("--local-vllm-max-num-batched-tokens", type=int)
    parser.add_argument("--local-vllm-max-num-seqs", type=int, default=64)
    parser.add_argument("--local-vllm-dtype", default="auto")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Maximum number of concurrent description-generation requests.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Regenerate descriptions even when an input row already has one.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output)
    cache_file = args.cache_file or output_dir / "description_cache.json"

    records = load_dataset_records(args.input, require_description=False)
    if args.limit is not None:
        records = records[: args.limit]

    def llm_factory() -> AugmentationLLM:
        return AugmentationLLM(
            model_name=args.model_name,
            api_backend=args.api_backend,
            api_base_url=args.api_base_url,
            request_timeout_seconds=args.request_timeout_seconds,
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
        )

    cache = _load_cache(cache_file)
    output_rows = asyncio.run(
        _prepare_descriptions(
            records=records,
            llm_factory=llm_factory,
            cache=cache,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            concurrency=max(1, args.concurrency),
            overwrite_existing=args.overwrite_existing,
        )
    )

    output_path = save_records(output_rows, output_dir)
    _save_cache(cache_file, cache)
    print(f"Wrote description dataset to {output_path}")
    print(f"Wrote description cache to {cache_file}")


async def _prepare_descriptions(
    *,
    records: list[DatasetRecord],
    llm_factory: Callable[[], AugmentationLLM],
    cache: dict[str, dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    concurrency: int,
    overwrite_existing: bool,
) -> list[dict[str, Any]]:
    groups = _group_records_by_module(records)
    _load_group_descriptions_from_rows(
        groups,
        cache=cache,
        overwrite_existing=overwrite_existing,
    )
    missing_groups = [group for group in groups.values() if group.description is None]

    if missing_groups:
        llm = llm_factory()
        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            _describe_group(
                group=group,
                llm=llm,
                semaphore=semaphore,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            for group in missing_groups
        ]
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Generating descriptions",
        ):
            group, description = await task
            group.description = description
            cache[group.key] = _cache_entry(group, description)

    return _rows_with_group_descriptions(records, groups)


async def _describe_group(
    *,
    group: ModuleGroup,
    llm: AugmentationLLM,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[ModuleGroup, str]:
    async with semaphore:
        description = await llm.describe_module(
            verilog=group.verilog,
            top_module=group.top_module,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    return group, description


def _group_records_by_module(records: list[DatasetRecord]) -> dict[str, ModuleGroup]:
    groups: dict[str, ModuleGroup] = {}
    for record in records:
        key = _description_key(record)
        if key not in groups:
            groups[key] = ModuleGroup(
                key=key,
                top_module=record.top_module,
                verilog=record.verilog,
                records=[],
            )
        groups[key].records.append(record)
    return groups


def _load_group_descriptions_from_rows(
    groups: dict[str, ModuleGroup],
    *,
    cache: dict[str, dict[str, str]],
    overwrite_existing: bool,
) -> None:
    for group in groups.values():
        if not overwrite_existing:
            existing = _first_existing_description(group)
            if existing is not None:
                group.description = existing
                cache[group.key] = _cache_entry(group, existing)
                continue
        cached = cache.get(group.key, {}).get("description")
        if _has_description(cached):
            group.description = cached


def _first_existing_description(group: ModuleGroup) -> str | None:
    for record in group.records:
        if _has_description(record.description):
            return record.description
    return None


def _rows_with_group_descriptions(
    records: list[DatasetRecord],
    groups: dict[str, ModuleGroup],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        group = groups[_description_key(record)]
        if group.description is None:
            raise RuntimeError(f"Missing description for module group {group.key}.")
        rows.append(_row_with_description(record, group.description))
    return rows


def _row_with_description(record: DatasetRecord, description: str) -> dict[str, Any]:
    row = dict(record.raw)
    row.update(
        {
            "sample_id": record.sample_id,
            "description": description,
            "verilog": record.verilog,
            "top_module": record.top_module,
        }
    )
    return row


def _description_key(record: DatasetRecord) -> str:
    text = record.verilog.replace("\r\n", "\n").replace("\r", "\n").strip()
    payload = f"{record.top_module}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_entry(group: ModuleGroup, description: str) -> dict[str, str]:
    return {
        "description": description,
        "top_module": group.top_module,
        "sample_id": group.records[0].sample_id,
        "num_rows": str(len(group.records)),
    }


def _has_description(description: str | None) -> bool:
    return bool(description and description.strip())


def _load_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cache: dict[str, dict[str, str]] = {}
    for key, value in data.items():
        if isinstance(value, str):
            cache[str(key)] = {"description": value}
        elif isinstance(value, dict) and isinstance(value.get("description"), str):
            cache[str(key)] = {
                str(k): str(v)
                for k, v in value.items()
                if isinstance(v, (str, int, float))
            }
    return cache


def _save_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
