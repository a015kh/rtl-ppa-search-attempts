from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Literal

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from .models import OptimizationTarget, PpaMetrics

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-Coder-Next"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"

CompletionKind = Literal["description", "code", "feedback"]
GenerationMode = Literal["whole", "diff"]


@dataclass
class _LocalVllmRequest:
    messages: list[dict[str, str]]
    temperature: float
    top_p: float
    max_tokens: int
    n: int
    future: asyncio.Future[list[str]]


class _LocalVllmBatcher:
    """Small asyncio micro-batcher around vLLM's offline LLM API."""

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        batch_timeout_seconds: float,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int | None,
        max_num_batched_tokens: int | None,
        max_num_seqs: int | None,
        dtype: str,
        trust_remote_code: bool,
        enable_prefix_caching: bool,
    ) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "The local-vllm backend requires `vllm` and `transformers`. "
                "Install them on the H200 node or use --api-backend vllm with a "
                "separate OpenAI-compatible vLLM server."
            ) from exc

        engine_kwargs: dict[str, Any] = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "enable_prefix_caching": enable_prefix_caching,
        }
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len
        if max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = max_num_seqs

        self._llm = LLM(**engine_kwargs)
        self._sampling_params_cls = SamplingParams
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.batch_size = max(1, batch_size)
        self.batch_timeout_seconds = max(0.0, batch_timeout_seconds)
        self._queue: asyncio.Queue[_LocalVllmRequest] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker_task: asyncio.Task[None] | None = None

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int = 1,
    ) -> list[str]:
        loop = asyncio.get_running_loop()
        self._ensure_worker(loop)
        assert self._queue is not None
        future: asyncio.Future[list[str]] = loop.create_future()
        await self._queue.put(
            _LocalVllmRequest(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=n,
                future=future,
            )
        )
        return await future

    def _ensure_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError(
                "local-vllm clients cannot be shared across event loops."
            )
        self._loop = loop
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = loop.create_task(self._worker())

    async def _worker(self) -> None:
        assert self._queue is not None
        while True:
            first = await self._queue.get()
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.batch_timeout_seconds
            while len(batch) < self.batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(
                        await asyncio.wait_for(self._queue.get(), remaining)
                    )
                except asyncio.TimeoutError:
                    break

            try:
                results = await asyncio.to_thread(self._generate_batch_sync, batch)
            except Exception as exc:
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)
            else:
                for request, texts in zip(batch, results):
                    if not request.future.done():
                        request.future.set_result(texts)
            finally:
                for _ in batch:
                    self._queue.task_done()

    def _generate_batch_sync(
        self,
        batch: list[_LocalVllmRequest],
    ) -> list[list[str]]:
        prompts = [
            self._tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for request in batch
        ]
        sampling_params = [
            self._sampling_params_cls(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                n=request.n,
            )
            for request in batch
        ]
        try:
            outputs = self._llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        except TypeError:
            outputs = []
            for prompt, params in zip(prompts, sampling_params):
                outputs.extend(
                    self._llm.generate(
                        [prompt],
                        sampling_params=params,
                        use_tqdm=False,
                    )
                )
        return [[choice.text for choice in output.outputs] for output in outputs]


class AugmentationLLM:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_QWEN_MODEL,
        api_backend: str = "local-vllm",
        api_key: str | None = None,
        api_base_url: str | None = None,
        request_timeout_seconds: float = 600.0,
        max_retries: int = 3,
        base_delay: float = 2.0,
        local_vllm_batch_size: int = 64,
        local_vllm_batch_timeout_ms: int = 50,
        local_vllm_tensor_parallel_size: int = 1,
        local_vllm_gpu_memory_utilization: float = 0.92,
        local_vllm_max_model_len: int | None = None,
        local_vllm_max_num_batched_tokens: int | None = None,
        local_vllm_max_num_seqs: int | None = 64,
        local_vllm_dtype: str = "auto",
        trust_remote_code: bool = True,
        enable_prefix_caching: bool = True,
    ) -> None:
        self.model_name = model_name
        self.api_backend = api_backend
        self.max_retries = max(1, max_retries)
        self.base_delay = base_delay
        self.lock = asyncio.Lock()
        self.debug = os.getenv("AUGMENTATION_LLM_DEBUG", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.stats = {
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "description_calls": 0,
            "code_calls": 0,
            "feedback_calls": 0,
        }
        self.local_batcher: _LocalVllmBatcher | None = None
        self.client_args: dict[str, Any] | None = None

        if api_backend == "local-vllm":
            self.local_batcher = _LocalVllmBatcher(
                model_name=model_name,
                batch_size=local_vllm_batch_size,
                batch_timeout_seconds=local_vllm_batch_timeout_ms / 1000.0,
                tensor_parallel_size=local_vllm_tensor_parallel_size,
                gpu_memory_utilization=local_vllm_gpu_memory_utilization,
                max_model_len=local_vllm_max_model_len,
                max_num_batched_tokens=local_vllm_max_num_batched_tokens,
                max_num_seqs=local_vllm_max_num_seqs,
                dtype=local_vllm_dtype,
                trust_remote_code=trust_remote_code,
                enable_prefix_caching=enable_prefix_caching,
            )
        else:
            self.client_args = _client_args(
                api_backend=api_backend,
                api_key=api_key or api_key_from_env(api_backend),
                api_base_url=api_base_url,
                request_timeout_seconds=request_timeout_seconds,
            )

    async def describe_module(
        self,
        *,
        verilog: str,
        top_module: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        system_prompt = (
            "You are a Verilog design documentation assistant. "
            "Describe RTL behavior from source code accurately and concisely. "
            "Do not propose optimizations or replacement code."
        )
        user_prompt = build_description_prompt(
            verilog=verilog,
            top_module=top_module,
        )
        content = await self._chat_single(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            completion_type="description",
        )
        description = strip_code_fence_language(content)
        if not description:
            raise RuntimeError("LLM failed to generate a module description.")
        return description

    async def optimize_module(
        self,
        *,
        description: str,
        verilog: str,
        top_module: str,
        target: OptimizationTarget,
        original_ppa: PpaMetrics,
        feedback: str | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> tuple[str, str]:
        results = await self.optimize_modules(
            description=description,
            verilog=verilog,
            top_module=top_module,
            target=target,
            original_ppa=original_ppa,
            feedback=feedback,
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return results[0]

    async def optimize_modules(
        self,
        *,
        description: str,
        verilog: str,
        top_module: str,
        target: OptimizationTarget,
        original_ppa: PpaMetrics,
        feedback: str | None,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> list[tuple[str, str]]:
        prompt = build_optimization_prompt(
            description=description,
            verilog=verilog,
            top_module=top_module,
            target=target,
            original_ppa=original_ppa,
            feedback=feedback,
        )
        last_errors: list[str] = []
        for attempt in range(2):
            responses = await self.generate_n_responses(
                prompt,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                generation_mode="whole",
            )
            parsed: list[tuple[str, str]] = []
            last_errors = []
            for response_index, (thought, code) in enumerate(responses):
                if not thought or not code:
                    last_errors.append(
                        f"response {response_index}: missing thought or code block"
                    )
                    continue
                cleaned_code = strip_code_fence_language(code)
                validation_error = validate_generated_verilog(
                    cleaned_code,
                    top_module=top_module,
                )
                if validation_error:
                    last_errors.append(
                        f"response {response_index}: {validation_error}"
                    )
                    continue
                parsed.append((thought, cleaned_code))
            if len(parsed) >= n:
                return parsed[:n]
            prompt = build_optimization_retry_prompt(
                original_prompt=prompt,
                top_module=top_module,
                validation_errors=last_errors,
            )
        detail = "; ".join(last_errors[:3]) or "no valid code block"
        raise RuntimeError(f"LLM failed to generate valid optimized RTL: {detail}")

    async def feedback(
        self,
        *,
        description: str,
        original_verilog: str,
        generated_verilog: str,
        target: OptimizationTarget,
        original_ppa: PpaMetrics,
        generated_ppa: PpaMetrics,
        synthesis_log_excerpt: str,
        thought: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are a Verilog and physical design review assistant. "
            "Given an original module, a generated module, and Yosys/OpenROAD "
            "PPA reports, "
            "reason whether the generated module appears to preserve the original behavior "
            "and whether it improves the requested metric. Do not provide replacement code. "
            "If equivalence is not formally proven, say that explicitly and give "
            "evidence-based reasoning."
        )
        user_prompt = build_feedback_prompt(
            description=description,
            original_verilog=original_verilog,
            generated_verilog=generated_verilog,
            target=target,
            original_ppa=original_ppa,
            generated_ppa=generated_ppa,
            synthesis_log_excerpt=synthesis_log_excerpt,
            thought=thought,
        )
        content = await self._chat_single(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            completion_type="feedback",
        )
        return parse_feedback(content)

    async def generate_response(
        self,
        prompt: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        generation_mode: GenerationMode = "whole",
    ) -> tuple[str | None, str | None]:
        responses = await self.generate_n_responses(
            prompt,
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            generation_mode=generation_mode,
        )
        return responses[0] if responses else (None, None)

    async def generate_n_responses(
        self,
        prompt: str,
        n: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        generation_mode: GenerationMode = "whole",
    ) -> list[tuple[str | None, str | None]]:
        system_prompt_content = _generation_system_prompt(generation_mode)
        contents = await self._chat(
            messages=[
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            completion_type="code",
        )
        return [parse_thought_and_code(content) for content in contents[:n]]

    async def get_and_reset_usage_stats(self) -> dict[str, int]:
        async with self.lock:
            stats = dict(self.stats)
            for key in self.stats:
                self.stats[key] = 0
            return stats

    async def _chat_single(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        completion_type: CompletionKind,
    ) -> str:
        contents = await self._chat(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
            completion_type=completion_type,
        )
        return contents[0] if contents else ""

    async def _chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        completion_type: CompletionKind,
    ) -> list[str]:
        if self.local_batcher is not None:
            contents = await self.local_batcher.generate(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=n,
            )
            await self._update_stats(
                None,
                n_calls=max(1, len(contents)),
                completion_type=completion_type,
            )
            return [
                content.strip()
                for content in contents
                if content and content.strip()
            ]

        assert self.client_args is not None
        async with AsyncOpenAI(**self.client_args) as client:
            for attempt in range(self.max_retries):
                try:
                    completion_kwargs = self._get_completion_kwargs(
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        n=n,
                    )
                    if self.debug:
                        safe_kwargs = {
                            key: value
                            for key, value in completion_kwargs.items()
                            if key != "messages"
                        }
                        print(f"[DEBUG] LLM request: {safe_kwargs}")
                    completion = await client.chat.completions.create(
                        **completion_kwargs
                    )
                    await self._update_stats(
                        completion,
                        n_calls=n,
                        completion_type=completion_type,
                    )
                    contents = [
                        choice.message.content.strip()
                        for choice in completion.choices
                        if choice.message.content and choice.message.content.strip()
                    ]
                    if len(contents) >= n:
                        return contents
                    if contents and n == 1:
                        return contents
                    if contents:
                        remaining = n - len(contents)
                        fallback = await asyncio.gather(
                            *[
                                self._chat(
                                    messages=messages,
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_tokens=max_tokens,
                                    n=1,
                                    completion_type=completion_type,
                                )
                                for _ in range(remaining)
                            ]
                        )
                        return contents + [
                            item for group in fallback for item in group
                        ]
                    print(
                        "Warning: received empty LLM response on attempt "
                        f"{attempt + 1}/{self.max_retries}."
                    )
                except BadRequestError as exc:
                    if n > 1 and _is_n_unsupported_error(exc):
                        fallback = await asyncio.gather(
                            *[
                                self._chat(
                                    messages=messages,
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_tokens=max_tokens,
                                    n=1,
                                    completion_type=completion_type,
                                )
                                for _ in range(n)
                            ]
                        )
                        return [item for group in fallback for item in group]
                    raise
                except (
                    APIConnectionError,
                    RateLimitError,
                    APITimeoutError,
                    InternalServerError,
                ) as exc:
                    if attempt + 1 == self.max_retries:
                        raise RuntimeError(
                            "LLM request failed after "
                            f"{self.max_retries} attempts: {exc}"
                        ) from exc
                    delay = (self.base_delay * 2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
            raise RuntimeError("LLM request failed after multiple empty responses.")

    def _get_completion_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "model": self.model_name,
        }
        is_openai_reasoning = self.api_backend == "openai" and any(
            marker in self.model_name.lower() for marker in ("o1-", "o3-", "gpt-5")
        )
        if is_openai_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        if n > 1:
            kwargs["n"] = n
        return kwargs

    async def _update_stats(
        self,
        completion: Any,
        *,
        n_calls: int,
        completion_type: CompletionKind,
    ) -> None:
        async with self.lock:
            self.stats["api_calls"] += n_calls
            self.stats[f"{completion_type}_calls"] += n_calls
            usage = getattr(completion, "usage", None)
            if usage is not None:
                self.stats["prompt_tokens"] += int(
                    getattr(usage, "prompt_tokens", 0) or 0
                )
                self.stats["completion_tokens"] += int(
                    getattr(usage, "completion_tokens", 0) or 0
                )


def _client_args(
    *,
    api_backend: str,
    api_key: str | None,
    api_base_url: str | None,
    request_timeout_seconds: float,
) -> dict[str, Any]:
    client_args: dict[str, Any] = {
        "api_key": api_key,
        "timeout": request_timeout_seconds,
    }
    if api_backend == "openai":
        if not api_key:
            raise ValueError("API key is required for OpenAI backend.")
    elif api_backend == "openrouter":
        if not api_key:
            raise ValueError("API key is required for OpenRouter backend.")
        client_args["base_url"] = api_base_url or "https://openrouter.ai/api/v1"
    elif api_backend == "deepseek":
        if not api_key:
            raise ValueError("API key is required for DeepSeek backend.")
        client_args["base_url"] = api_base_url or "https://api.deepseek.com"
    elif api_backend == "gemini":
        if not api_key:
            raise ValueError("API key is required for Gemini backend.")
        client_args["base_url"] = (
            api_base_url or "https://generativelanguage.googleapis.com/v1beta/openai"
        )
    elif api_backend == "vllm":
        client_args["api_key"] = api_key or "EMPTY"
        client_args["base_url"] = (
            api_base_url
            or os.getenv("AUGMENTATION_VLLM_BASE_URL")
            or os.getenv("VLLM_BASE_URL")
            or DEFAULT_VLLM_BASE_URL
        )
    else:
        raise ValueError(
            "Unsupported API backend: "
            f"{api_backend}. Choose openai, openrouter, deepseek, gemini, vllm, "
            "or local-vllm."
        )
    return client_args


def api_key_from_env(api_backend: str) -> str | None:
    if api_backend == "openai":
        return os.getenv("RTL_OAI_KEY") or os.getenv("OPENAI_API_KEY")
    if api_backend == "openrouter":
        return os.getenv("OPENROUTER_API_KEY")
    if api_backend == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY")
    if api_backend == "gemini":
        return os.getenv("GEMINI_API_KEY")
    if api_backend in {"vllm", "local-vllm"}:
        return None
    raise ValueError(f"Unsupported API backend: {api_backend}")


def build_optimization_prompt(
    *,
    description: str,
    verilog: str,
    top_module: str,
    target: OptimizationTarget,
    original_ppa: PpaMetrics,
    feedback: str | None,
) -> str:
    target_text = {
        "area": "minimize synthesized area",
        "power": "minimize total power",
        "delay": "minimize effective clock period / timing delay",
    }[target]
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\nPrevious feedback to consider:\n"
            "```feedback\n"
            f"{feedback}\n"
            "```\n"
        )
    return (
        "Optimize this Verilog module for data augmentation.\n\n"
        "Hard requirements:\n"
        "- Preserve the same top-level module name and port interface.\n"
        "- Preserve the behavior described by the text description.\n"
        "- Return one complete synthesizable Verilog/SystemVerilog source file only in the code block.\n"
        "- The code block must contain the full final file, not a patch, diff, replacement statement, "
        "replacement always block, or explanation.\n"
        "- The final file must include `module "
        f"{top_module}` and the matching `endmodule`.\n"
        "- If the original file contains helper/submodules, include every helper module still required "
        "by the top module in the returned file. Do not omit unchanged required modules.\n"
        "- Do not use placeholders, ellipses (`...`), TODOs, or comments that stand in for omitted RTL.\n"
        "- Do not add testbench code.\n"
        "- Prefer conservative RTL transformations that Yosys can synthesize.\n\n"
        f"Optimization target: {target} ({target_text}).\n"
        f"Top module: {top_module}\n\n"
        "Text description:\n"
        "```text\n"
        f"{description}\n"
        "```\n\n"
        "Baseline Yosys/OpenROAD PPA metrics:\n"
        "```json\n"
        f"{json.dumps(original_ppa.to_dict(), indent=2)}\n"
        "```\n"
        f"{feedback_block}\n"
        "Original Verilog:\n"
        "```verilog\n"
        f"{verilog}\n"
        "```\n"
    )


def build_optimization_retry_prompt(
    *,
    original_prompt: str,
    top_module: str,
    validation_errors: list[str],
) -> str:
    errors = "\n".join(f"- {error}" for error in validation_errors)
    return (
        f"{original_prompt}\n\n"
        "Your previous answer was rejected before synthesis because it was not a "
        "complete Verilog/SystemVerilog source file.\n"
        "Validation errors:\n"
        f"{errors or '- unknown validation failure'}\n\n"
        "Regenerate the candidate now. The `code` block must contain the full final "
        f"source file, including `module {top_module}` and `endmodule`. Include all "
        "helper modules required by the top module. Do not return only changed lines, "
        "a replacement statement, a replacement always block, a diff, or ellipses."
    )


def build_description_prompt(
    *,
    verilog: str,
    top_module: str,
) -> str:
    return (
        "Write a concise text description of this RTL module for use as an "
        "augmentation prompt.\n\n"
        "Requirements:\n"
        "- Describe observable behavior, top-level inputs/outputs, control flow, "
        "state, and notable datapath operations.\n"
        "- Infer behavior only from the code. Do not invent unstated protocol details.\n"
        "- Do not mention synthesis metrics, clock constraints, PPA, or optimization.\n"
        "- Return plain text only, with no Markdown code block.\n\n"
        f"Top module: {top_module}\n\n"
        "Verilog:\n"
        "```verilog\n"
        f"{verilog}\n"
        "```\n"
    )


def build_feedback_prompt(
    *,
    description: str,
    original_verilog: str,
    generated_verilog: str,
    target: OptimizationTarget,
    original_ppa: PpaMetrics,
    generated_ppa: PpaMetrics,
    synthesis_log_excerpt: str,
    thought: str,
) -> str:
    return (
        "Review this generated Verilog augmentation candidate.\n\n"
        "Return exactly these XML-style sections:\n"
        "<EQUIVALENCE_REASONING>...</EQUIVALENCE_REASONING>\n"
        "<PPA_VERIFICATION>...</PPA_VERIFICATION>\n"
        "<FEEDBACK_THOUGHTS>...</FEEDBACK_THOUGHTS>\n\n"
        f"Requested optimization target: {target}\n\n"
        "Text description:\n"
        "```text\n"
        f"{description}\n"
        "```\n\n"
        "Generator thought:\n"
        "```thought\n"
        f"{thought}\n"
        "```\n\n"
        "Original PPA:\n"
        "```json\n"
        f"{json.dumps(original_ppa.to_dict(), indent=2)}\n"
        "```\n\n"
        "Generated PPA:\n"
        "```json\n"
        f"{json.dumps(generated_ppa.to_dict(), indent=2)}\n"
        "```\n\n"
        "Original Verilog:\n"
        "```verilog\n"
        f"{original_verilog}\n"
        "```\n\n"
        "Generated Verilog:\n"
        "```verilog\n"
        f"{generated_verilog}\n"
        "```\n\n"
        "Yosys/OpenROAD log excerpt:\n"
        "```log\n"
        f"{synthesis_log_excerpt}\n"
        "```\n"
    )


def parse_feedback(text: str) -> dict[str, Any]:
    return {
        "equivalence_reasoning": _tag(text, "EQUIVALENCE_REASONING"),
        "ppa_verification": _tag(text, "PPA_VERIFICATION"),
        "feedback_thoughts": _tag(text, "FEEDBACK_THOUGHTS"),
        "raw": text,
    }


def parse_thought_and_code(response_text: str) -> tuple[str | None, str | None]:
    thought_match = re.search(
        r"```thought\s*\n(.*?)\n```",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    code_match = re.search(
        r"```(?:code|verilog|systemverilog|sv)\s*\n(.*?)\n```",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    thought = thought_match.group(1).strip() if thought_match else None
    code = code_match.group(1).strip() if code_match else None
    if thought is None and code is not None:
        thought = "Generated RTL candidate."
    if code is None:
        code = _first_code_fence(response_text)
    if thought is None or code is None:
        warning = "\n\n--- WARNING: Parsing Issues ---\n"
        if thought is None:
            warning += "Could not parse a thought block.\n"
        if code is None:
            warning += "Could not parse a code block.\n"
        return response_text + warning, response_text + warning
    return thought, code


def strip_code_fence_language(code: str) -> str:
    cleaned = code.strip()
    cleaned = re.sub(
        r"^```(?:code|verilog|systemverilog|sv)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def validate_generated_verilog(code: str, *, top_module: str) -> str | None:
    stripped = code.strip()
    if not stripped:
        return "empty generated Verilog"
    if "..." in stripped or "\u2026" in stripped:
        return "contains an ellipsis/placeholders instead of complete RTL"
    if re.search(r"(?m)^\s*(?:---|\+\+\+|@@)\s", stripped):
        return "looks like a diff instead of a complete Verilog file"
    if "```" in stripped:
        return "contains Markdown fences inside the code block"

    module_decl = re.compile(
        rf"(?m)^\s*module\s+{re.escape(top_module)}\b"
    )
    if not module_decl.search(stripped):
        return f"missing expected top module declaration: module {top_module}"

    module_count = len(re.findall(r"(?m)^\s*module\s+", stripped))
    endmodule_count = len(re.findall(r"\bendmodule\b", stripped))
    if endmodule_count < module_count:
        return (
            "has fewer endmodule tokens than module declarations "
            f"({endmodule_count} endmodule for {module_count} module)"
        )
    return None


def _generation_system_prompt(generation_mode: GenerationMode) -> str:
    if generation_mode == "whole":
        return (
            "You are an expert Verilog design assistant. "
            "Your role is to address Verilog-related problems posed by the user. "
            "For each problem, you must provide both a 'thought' and the corresponding 'code'. "
            "The 'thought' is your concise design idea. "
            "The 'code' is the complete final Verilog/SystemVerilog source file. "
            "Even when making a small edit, never return only the changed statement, "
            "changed always block, a patch, or an excerpt. Multi-module inputs require "
            "a complete returned file with the top module and all helper modules still "
            "needed by that top module. Do not use ellipses or placeholders.\n"
            "Strictly format your response as follows:\n"
            "```thought\n"
            "[Your concise design idea here]\n"
            "```\n"
            "```code\n"
            "[Your complete synthesizable Verilog/SystemVerilog file here]\n"
            "```"
        )
    return (
        "You are an expert Verilog design assistant that modifies code based on user requests.\n"
        "You must format your response with a `thought` block and a `code` block. "
        "The `code` block must contain only the requested diff."
    )


def _is_n_unsupported_error(exc: BadRequestError) -> bool:
    message = str(exc).lower()
    return (
        "invalid n value" in message
        or "only n = 1 is supported" in message
        or ("invalid" in message and "'n'" in message)
        or "invalid value of n" in message
    )


def _first_code_fence(text: str) -> str | None:
    match = re.search(r"```\w*\s*\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _tag(text: str, name: str) -> str | None:
    match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text, re.DOTALL)
    return match.group(1).strip() if match else None
