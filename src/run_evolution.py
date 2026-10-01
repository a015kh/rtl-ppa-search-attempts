import argparse
import datetime
import multiprocessing
import os
import sys
import time

from pathlib import Path

from tqdm import tqdm

# Import the core logic from src package
from revolution.algorithm import EoHEngine
from revolution.dcpt import DEFAULT_CELL_LIBRARY, DEFAULT_FLOW_ROOT, DCPTFlow
from revolution.dcpt_evaluation import DCPTEvaluator
from revolution.evaluation import (
    PPAProxyEvaluator,
    PPAProxyEvaluatorWrapper,
    SynthesisEvaluator,
    VerilogEvaluator,
    VerilogEvaluatorICRTL,
)
from revolution.llm import LLMInterface
from revolution.utils import StreamRedirector


def build_synthesis_evaluator(args, benchmark):
    if args.synthesis_evaluator == "proxy":
        if not args.proxy_checkpoint:
            raise ValueError("--proxy_checkpoint is required for proxy evaluation")
        proxy_eval = PPAProxyEvaluator(
            checkpoint_path=args.proxy_checkpoint,
            encoder_model_name=args.proxy_encoder,
            metrics=args.metrics,
            device=args.proxy_device,
        )
        return PPAProxyEvaluatorWrapper(
            proxy_eval,
            input_preprocess=args.proxy_input_preprocess,
            yosys_bin=args.proxy_yosys_bin,
            yosys_timeout_seconds=args.proxy_yosys_timeout_seconds,
            yosys_pdk=args.proxy_yosys_pdk,
            yosys_pdk_path=args.pdk_path,
            dependency_root=args.dependency_root,
        )
    if args.synthesis_evaluator == "dcpt":
        return DCPTEvaluator(
            benchmark_path=os.path.join(args.benchmark_root, benchmark),
            flow_root=args.dcpt_flow_root,
            config_path=args.dcpt_config,
            cell_library=args.dcpt_cell_library,
            clocks=args.dcpt_clocks,
            constraint_tolerance_ns=args.dcpt_constraint_tolerance_ns,
        )
    return SynthesisEvaluator(pdk_path=args.pdk_path)


# Wrapper function for multiprocessing
def run_problem_worker(args_tuple):
    """
    Wrapper function to run a single problem instance.
    This function will be executed by each worker process.
    All stdout/stderr from this process is redirected to a problem-specific log file.
    """
    # Unpack arguments
    benchmark, problem, args = args_tuple

    # Individual Log Setup
    model_name_cleaned = args.model_name.replace("/", "_")
    problem_log_dir = os.path.join(
        args.save_path, model_name_cleaned, benchmark, problem
    )
    # The EoHEngine will create this directory, but we ensure it exists early.
    os.makedirs(problem_log_dir, exist_ok=True)
    individual_log_path = os.path.join(problem_log_dir, "problem_run.log")

    # Redirect all output from this worker to the individual log file
    with StreamRedirector(filepath=individual_log_path):
        print(
            f"\n[Worker PID: {os.getpid()}] Starting problem: {benchmark}/{problem}\n"
        )

        # Initialize objects within the worker process to avoid pickling issues
        # Determine the API key based on the selected backend
        api_key = None
        if args.api_backend == "openai":
            api_key = os.getenv("RTL_OAI_KEY")
        elif args.api_backend == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
        elif args.api_backend == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")

        if args.api_backend not in [
            "vllm",
            "unsloth",
            "codex",
        ]:  # Local/CLI backends do not require an API key
            if not api_key:
                raise ValueError(
                    f"API key for backend '{args.api_backend}' not found. "
                    f"Please set the corresponding environment variable (e.g., RTL_OAI_KEY, OPENROUTER_API_KEY, DEEPSEEK_API_KEY)."
                )
        llm_interface = LLMInterface(
            api_key=api_key,
            model_name=args.model_name,
            api_backend=args.api_backend,
            codex_reasoning_effort=args.codex_reasoning_effort,
            codex_timeout_seconds=args.codex_timeout_seconds,
        )
        if benchmark == "ICRTL":
            verilog_evaluator = VerilogEvaluatorICRTL(
                iverilog_executable_path="iverilog", vvp_executable_path="vvp"
            )
        else:
            verilog_evaluator = VerilogEvaluator(
                iverilog_executable_path="iverilog", vvp_executable_path="vvp"
            )
        synthesis_evaluator = build_synthesis_evaluator(args, benchmark)

        eoh_engine = EoHEngine(
            problem_name=problem,
            benchmark_name=benchmark,
            llm_interface=llm_interface,
            verilog_evaluator=verilog_evaluator,
            synthesis_evaluator=synthesis_evaluator,
            population_size=args.population_size,
            num_generations=args.num_generations,
            base_save_path=args.save_path,
            default_llm_temp=args.temperature,
            default_llm_top_p=args.top_p,
            default_llm_max_tokens=args.max_tokens,
            strategy_selection_method=args.strategy_selection,
            epsilon=args.epsilon,
            ucb_c=args.ucb_c,
            generation_mode=args.generation_mode,
            start_from_reference=args.start_from_reference,
            best_record_require_constraints=args.best_record_require_constraints,
            debug_bypass_path=args.debug_bypass_path,
            benchmark_root=args.benchmark_root,
        )
        result_str = eoh_engine.run()
        # Return the result string and the path to the individual log file created for this problem
        print(f"[Worker PID: {os.getpid()}] Finished problem: {benchmark}/{problem}\n")
    return result_str, individual_log_path


# Wrapper function for multiprocessing
def run_indexed_problem_worker(indexed_task):
    """
    Wrapper function to run a single problem instance.
    This function will be executed by each worker process.
    All stdout/stderr from this process is redirected to a problem-specific log file.

    This version includes the index of the task for better tracking.
    To support parallel processing with imap_unordered, while preserving the order of tasks,
    we need to index the tasks before passing them to the pool.

    :param indexed_task: A tuple containing the index and a tuple of (benchmark, problem, args).
    :type indexed_task: tuple
    :return: A tuple containing the index and the result tuple(result_str, individual_log_path) from the EoHEngine run.
    :rtype: tuple

    """
    # Unpack arguments
    index, args_tuple = indexed_task
    benchmark, problem, args = args_tuple

    # Individual Log Setup
    model_name_cleaned = args.model_name.replace("/", "_")
    problem_log_dir = os.path.join(
        args.save_path, model_name_cleaned, benchmark, problem
    )
    # The EoHEngine will create this directory, but we ensure it exists early.
    os.makedirs(problem_log_dir, exist_ok=True)
    individual_log_path = os.path.join(problem_log_dir, "problem_run.log")

    # Redirect all output from this worker to the individual log file
    with StreamRedirector(filepath=individual_log_path):
        print(
            f"\n[Worker PID: {os.getpid()}] Starting problem: {benchmark}/{problem}\n"
        )

        # Initialize objects within the worker process to avoid pickling issues
        # Determine the API key based on the selected backend
        api_key = None
        if args.api_backend == "openai":
            api_key = os.getenv("RTL_OAI_KEY")
        elif args.api_backend == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
        elif args.api_backend == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
        elif args.api_backend == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")

        if args.api_backend not in [
            "vllm",
            "unsloth",
            "codex",
        ]:  # Local/CLI backends do not require an API key
            if not api_key:
                raise ValueError(
                    f"API key for backend '{args.api_backend}' not found. "
                    f"Please set the corresponding environment variable (e.g., RTL_OAI_KEY, OPENROUTER_API_KEY, DEEPSEEK_API_KEY)."
                )
        llm_interface = LLMInterface(
            api_key=api_key,
            model_name=args.model_name,
            api_backend=args.api_backend,
            codex_reasoning_effort=args.codex_reasoning_effort,
            codex_timeout_seconds=args.codex_timeout_seconds,
        )
        if benchmark == "ICRTL":
            verilog_evaluator = VerilogEvaluatorICRTL(
                iverilog_executable_path="iverilog", vvp_executable_path="vvp"
            )
        else:
            verilog_evaluator = VerilogEvaluator(
                iverilog_executable_path="iverilog", vvp_executable_path="vvp"
            )
        synthesis_evaluator = build_synthesis_evaluator(args, benchmark)

        eoh_engine = EoHEngine(
            problem_name=problem,
            benchmark_name=benchmark,
            llm_interface=llm_interface,
            verilog_evaluator=verilog_evaluator,
            synthesis_evaluator=synthesis_evaluator,
            population_size=args.population_size,
            num_generations=args.num_generations,
            base_save_path=args.save_path,
            default_llm_temp=args.temperature,
            default_llm_top_p=args.top_p,
            default_llm_max_tokens=args.max_tokens,
            strategy_selection_method=args.strategy_selection,
            epsilon=args.epsilon,
            ucb_c=args.ucb_c,
            generation_mode=args.generation_mode,
            start_from_reference=args.start_from_reference,
            best_record_require_constraints=args.best_record_require_constraints,
            debug_bypass_path=args.debug_bypass_path,
            benchmark_root=args.benchmark_root,
        )
        result_str = eoh_engine.run()
        # Return the result string and the path to the individual log file created for this problem
        print(f"[Worker PID: {os.getpid()}] Finished problem: {benchmark}/{problem}\n")
        result = (result_str, individual_log_path)
    return (index, result)


def main():
    """Main function to parse arguments and orchestrate the evolutionary run."""
    # Use argparse to make the script configurable
    parser = argparse.ArgumentParser(
        description="Run the EoH framework on specified Verilog benchmarks."
    )

    default_benchmark_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "bench")
    )
    parser.add_argument(
        "--benchmark_root",
        type=str,
        required=True,
        help="Directory containing benchmark suites.",
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmark suites to run. Default: every suite under --benchmark_root.",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        help="A list of specific problem names to run. If not provided, all problems in the suite will be run.",
    )
    parser.add_argument(
        "--api_backend",
        type=str,
        default="openai",
        choices=[
            "openai",
            "openrouter",
            "deepseek",
            "gemini",
            "vllm",
            "unsloth",
            "codex",
        ],
        help="The API backend to use for LLM calls.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4.1-mini",
        help="Name of the OpenAI model to use.",
    )
    parser.add_argument(
        "--codex_reasoning_effort",
        choices=["low", "medium", "high", "xhigh", "max", "ultra"],
        default="medium",
        help="Reasoning effort passed to Codex CLI.",
    )
    parser.add_argument(
        "--codex_timeout_seconds",
        type=int,
        default=3600,
        help="Timeout for each non-interactive Codex CLI turn.",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=5,
        help="Number of candidates in each generation.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=5,
        help="Number of evolutionary generations to run.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Base path to save results.",
    )  # Default is ./exp, defined relative to main.py
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of parallel processes to use.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument(
        "--strategy_selection",
        type=str,
        default="random",
        choices=["random", "epsilon-greedy", "ucb"],
        help="The meta-strategy for selecting genetic operators.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="The exploration factor for the epsilon-greedy strategy.",
    )
    parser.add_argument(
        "--ucb_c",
        type=float,
        default=2.0,
        help="The exploration constant (c) for the UCB strategy.",
    )
    parser.add_argument(  # Choice of generation mode either "whole" or "diff"
        "--generation_mode",
        type=str,
        default="whole",
        choices=["whole", "diff"],
        help="Mode of generation, either 'whole' (full code) or 'diff' (code diffs). "
        "In 'whole' mode, the entire code is generated. "
        "In 'diff' mode, only the differences from the original code are generated.",
    )
    parser.add_argument(
        "--start_from_reference",
        action="store_true",
        help="Start evolution from the benchmark reference Verilog (<problem_name>_ref.sv) instead of LLM-generated Gen0.",
    )
    # Synthesis Evaluator Options
    parser.add_argument(
        "--synthesis_evaluator",
        type=str,
        default="synthesis",
        choices=["synthesis", "proxy", "dcpt"],
        help="Use Yosys/OpenROAD, the PPA proxy, or Design Compiler/PrimeTime.",
    )
    parser.add_argument("--pdk_path", default=None, help="External PDK root for Yosys/OpenROAD.")
    parser.add_argument("--dependency_root", default=None, help="Optional external RTL dependency tree.")
    parser.add_argument("--dcpt_flow_root", type=str, default=None)
    parser.add_argument("--dcpt_config", type=str, default=None)
    parser.add_argument(
        "--dcpt_cell_library", type=str, default=None
    )
    parser.add_argument(
        "--dcpt_clocks",
        type=float,
        nargs="+",
        default=None,
        help="Evaluate each DC/PT candidate at these benchmark clock points. "
        "The benchmark optimization clock is always included.",
    )
    parser.add_argument(
        "--dcpt_constraint_tolerance_ns",
        type=float,
        default=1e-6,
        help="Timing feasibility tolerance in ns for effective_period <= clock.",
    )
    parser.add_argument(
        "--best_record_require_constraints",
        action="store_true",
        help="Only timing-feasible candidates may populate best-record and Pareto summary fields.",
    )
    parser.add_argument(
        "--proxy_checkpoint",
        type=str,
        default=None,
        help="Path to the T5Gemma PPA proxy checkpoint directory.",
    )
    parser.add_argument(
        "--proxy_encoder",
        type=str,
        default="google/t5gemma-s-s-prefixlm",
        help="Base encoder model name for the PPA proxy.",
    )
    parser.add_argument(
        "--proxy_device",
        type=str,
        default=None,
        help="PyTorch device for the PPA proxy model, e.g. 'cpu', 'cuda', or 'cuda:0'. Default: auto-detect.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["area", "delay", "static_power", "switching_power"],
        help="Ordered PPA metrics to predict (space-separated).",
    )
    parser.add_argument(
        "--proxy_input_preprocess",
        type=str,
        default="none",
        choices=["none", "yosys"],
        help="Preprocess RTL before proxy inference. 'yosys' sends a Yosys-normalized netlist to the proxy.",
    )
    parser.add_argument(
        "--proxy_yosys_bin",
        type=str,
        default="yosys",
        help="Yosys executable used when --proxy_input_preprocess yosys.",
    )
    parser.add_argument(
        "--proxy_yosys_timeout_seconds",
        type=int,
        default=120,
        help="Timeout in seconds for proxy Yosys preprocessing.",
    )
    parser.add_argument(
        "--proxy_yosys_pdk",
        type=str,
        default="nangate45",
        choices=["nangate45", "sky130"],
        help="PDK/template target for proxy Yosys preprocessing.",
    )
    # Debug Options
    parser.add_argument(
        "--debug_bypass_path",
        type=str,
        default=None,
        help="Path to pre-generated RTL samples to bypass LLM calls (e.g., src/exp/n_samples/gpt-5.4-mini).",
    )

    # problems_file
    parser.add_argument(
        "--problems_file",
        type=str,
        default="problems.txt",
        help="Path to a file containing a list of problems to run.",
    )

    args = parser.parse_args()
    if (
        args.dcpt_clocks is not None or args.best_record_require_constraints
    ) and args.synthesis_evaluator != "dcpt":
        parser.error(
            "--dcpt_clocks and --best_record_require_constraints require "
            "--synthesis_evaluator dcpt"
        )
    if args.synthesis_evaluator == "dcpt" and not all([args.dcpt_flow_root, args.dcpt_config, args.dcpt_cell_library]):
        parser.error("DC/PT requires --dcpt_flow_root, --dcpt_config and --dcpt_cell_library.")
    if (args.synthesis_evaluator == "synthesis" or args.proxy_input_preprocess == "yosys") and not args.pdk_path:
        parser.error("Provide --pdk_path for Yosys/OpenROAD or Yosys proxy preprocessing.")
    args.benchmark_root = os.path.abspath(args.benchmark_root)
    if not os.path.isdir(args.benchmark_root):
        parser.error(f"Benchmark root does not exist: {args.benchmark_root}")
    available_benchmarks = sorted(
        name
        for name in os.listdir(args.benchmark_root)
        if os.path.isdir(os.path.join(args.benchmark_root, name))
    )
    if args.benchmarks is None:
        args.benchmarks = available_benchmarks
    missing_benchmarks = sorted(set(args.benchmarks) - set(available_benchmarks))
    if missing_benchmarks:
        parser.error(f"Unknown benchmarks: {missing_benchmarks}")
    if args.synthesis_evaluator == "dcpt":
        DCPTFlow.load(args.dcpt_flow_root).preflight(
            Path(args.save_path).resolve() / ".dcpt_preflight",
            args.num_workers,
            args.dcpt_config,
        )

    api_key = None
    if args.api_backend not in [
        "vllm",
        "unsloth",
        "codex",
    ]:  # Local/CLI backends do not require an API key
        # Map backends to their required environment variables
        api_key_env_vars = {
            "openai": "RTL_OAI_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        # Check for the correct key based on the selected backend
        required_key_var = api_key_env_vars.get(args.api_backend, None)
        if required_key_var:
            api_key = os.getenv(required_key_var)
            if not api_key:
                print(
                    f"LLM Initialization Error: The environment variable '{required_key_var}' must be set for the '{args.api_backend}' backend."
                )
                exit(1)

    # Main execution block now handles comprehensive, aggregated logging
    # --- Task Preparation ---
    run_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()
    model_name_cleaned = args.model_name.replace("/", "_")
    # Define path for the new comprehensive log file for the entire run
    master_log_dir = os.path.join(args.save_path, model_name_cleaned)
    comprehensive_log_path = os.path.join(master_log_dir, f"{run_datetime}_run_log.txt")

    # Define path for the summary results file (similar to the original script's master log)
    summary_results_path = os.path.join(
        master_log_dir, f"{run_datetime}_summary_results.txt"
    )

    # Use a list to store results before writing to files
    results_data = []
    tasks_to_run = []  # Define this before the try block

    # Save a reference to the real stdout before it gets redirected
    # This allows us to print to the console even when redirecting output
    # This is to allow the progress bar to print to the console
    original_stdout = sys.stdout

    # The `finally` block will handle aggregation.
    try:
        # Redirect all output from this main script to the comprehensive log file
        with StreamRedirector(filepath=comprehensive_log_path):
            print(f"--- EoH Framework Run Started: {run_datetime} ---")
            print(f"Arguments: {vars(args)}")
            print("-" * 50)

            # --- Task Preparation ---
            # Task preparation loop to populate tasks_to_run
            for benchmark in args.benchmarks:
                benchmark_dir = os.path.join(args.benchmark_root, benchmark)
                problems_file = os.path.join(benchmark_dir, args.problems_file)
                if not os.path.exists(problems_file):
                    print(
                        f"Warning: '{problems_file}' not found in {benchmark_dir}. Skipping."
                    )
                    continue
                with open(problems_file, "r") as f:
                    all_problems = [line.strip() for line in f if line.strip()]

                problems_to_process = args.problems if args.problems else all_problems
                for problem in problems_to_process:
                    if problem in all_problems:
                        tasks_to_run.append((benchmark, problem, args))

            if not tasks_to_run:
                print("No valid problems found to run. Exiting.")
            else:
                print(
                    f"\nStarting parallel execution with {args.num_workers} workers for {len(tasks_to_run)} problems."
                )

                # Index the tasks before running, as imap_unordered does not preserve order
                indexed_tasks = list(enumerate(tasks_to_run))

                with multiprocessing.Pool(processes=args.num_workers) as pool:
                    results_iterator = pool.imap_unordered(
                        run_indexed_problem_worker, indexed_tasks
                    )

                    unordered_results = list(
                        tqdm(
                            results_iterator,
                            total=len(indexed_tasks),
                            desc="Running problems",
                            file=original_stdout,  # Use the original stdout for progress bar
                        )
                    )
                # Sort the results by the original index to maintain order
                unordered_results.sort(key=lambda x: x[0])
                # Strip the index from the results
                results_data = [result[1] for result in unordered_results]
                print("\n--- All parallel tasks completed successfully.---")

    finally:
        end_time = time.time()
        # --- This block will ALWAYS run, even if the pool crashes ---
        print("\n--- Aggregation & Finalization Step ---")

        # Re-open the comprehensive log in append mode to add aggregation results
        with open(comprehensive_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(
                "\n\n" + "=" * 20 + " AGGREGATED INDIVIDUAL LOGS " + "=" * 20 + "\n"
            )

            # Check if any results were produced before a potential crash
            if not results_data:
                log_file.write(
                    "\nNo results were returned from worker processes. This may be due to an early crash.\nCheck individual problem directories for logs.\n"
                )
            else:
                for result_str, individual_log_path in results_data:
                    try:
                        path_parts = individual_log_path.split(os.sep)
                        problem_identifier = os.path.join(
                            path_parts[-3], path_parts[-2]
                        )

                        with open(
                            individual_log_path, "r", encoding="utf-8"
                        ) as f_individual:
                            log_contents = f_individual.read()

                        log_file.write(f"\n{problem_identifier}:\n")
                        log_file.write(f"{{\n{log_contents}\n}}\n")
                        log_file.write("-" * 50 + "\n")

                    except Exception as e:
                        log_file.write(
                            f"\n--- Error processing log {individual_log_path}: {e} ---\n"
                        )
            log_file.write("\n--- EoH Framework Run Completed ---\n")
            log_file.write(f"Total run time: {end_time - start_time:.2f} seconds\n")
        # --- Write the summary results file ---
        if results_data:
            try:
                with open(summary_results_path, "w") as summary_file:
                    for result_str, _ in results_data:
                        summary_file.write(f"{result_str}\n")
                print(f"\nSummary results saved to: {summary_results_path}")
            except Exception as e:
                print(f"Error writing summary results file: {e}")

        print(
            f"Comprehensive run log with aggregated details saved to: {comprehensive_log_path}"
        )
        print(f"Total run time: {end_time - start_time:.2f} seconds")
        print("--- EoH Framework Run Completed ---")


if __name__ == "__main__":
    main()
