import argparse
import datetime
import multiprocessing
import subprocess
import os
import sys
import time
import json

from tqdm import tqdm

from revolution.evaluation import SynthesisEvaluator, VerilogEvaluatorICRTL, VerilogEvaluator

if __name__ == "__main__":
    synthesis_evaluator = SynthesisEvaluator(pdk_path="../pdk")
    # verilog_evaluator_icrtl = VerilogEvaluatorICRTL(iverilog_executable_path="iverilog", vvp_executable_path="vvp")
    # verilog_evaluator = VerilogEvaluator()

    benchmarks = [
        "../bench/ICRTL/Q1_LBP",
        "../bench/ICRTL/Q2_GEMM",
        "../bench/ICRTL/Q3_CONV",
        "../bench/ICRTL/Q4_HC",
        "../bench/ICRTL/Q5_JAM",
        "../bench/ICRTL/Q6_DT",
    ]

    output_dir = "exp/ICRTL_syn"
    formatted_benchmark_dir = "../data/bench/ICRTL"
    os.makedirs(formatted_benchmark_dir, exist_ok=True)

    # Format the benchmark directory
    # the benchmark directory requires {problem_name}_ppa.txt, {problem_name}_prompt.txt, {problem_name}_ref.sv, {problem_name}_test.sv


    os.makedirs(output_dir, exist_ok=True)
    synthesis_top_module_name = {
        "Q1_LBP": "TOP",
        "Q2_GEMM": "TOP",
        "Q3_CONV": "TOP",
        "Q4_HC": "TOP",
        "Q5_JAM": "TOP",
        "Q6_DT": "TOP",
    }
    with open(os.path.join(formatted_benchmark_dir, "synthesis_top_module_name.json"), "w") as f:
        json.dump(synthesis_top_module_name, f, indent=4)

    results = []
    for benchmark in tqdm(benchmarks):
        output_path = os.path.join(output_dir, benchmark.split("/")[-1])
        report_path = os.path.join(output_path, "report")
        syn_path = os.path.join(output_path, "syn")
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(report_path, exist_ok=True)
        os.makedirs(syn_path, exist_ok=True)
        ref_file = os.path.join(benchmark, "ref_solution/initial.sv")
        test_file = os.path.join(benchmark, "00_TB/test.sv")
        top_module_name = "TOP"
        problem_name = benchmark.split("/")[-1]

        # copy ref_file to formatted_benchmark_dir
        formatted_ref_file = os.path.join(formatted_benchmark_dir, f"{problem_name}_ref.sv")
        subprocess.run(["cp", ref_file, formatted_ref_file])
        # copy test_file to formatted_benchmark_dir
        formatted_test_file = os.path.join(formatted_benchmark_dir, f"{problem_name}_test.sv")
        subprocess.run(["cp", test_file, formatted_test_file])
        # copy spec file to formatted_benchmark_dir
        spec_file = os.path.join(benchmark, "referenced_spec/human.md")
        formatted_spec_file = os.path.join(formatted_benchmark_dir, f"{problem_name}_prompt.txt")
        subprocess.run(["cp", spec_file, formatted_spec_file])

        synthesized_netlist_path = os.path.join(syn_path, "initial_syn.v")
        synthesis_success, synthesis_log = synthesis_evaluator._run_synthesis(
            verilog_file=ref_file,
            problem_name=problem_name,
            synth_top_module_name=top_module_name,
            output_directory=output_path,
            report_base_path=report_path,
            synthesized_netlist_path=synthesized_netlist_path,
        )
        if not synthesis_success:
            print(f"Synthesis failed for {problem_name}")
            continue
        
        ppa_metrics = synthesis_evaluator._parse_ppa_log(synthesis_log)
        results.append({
            "problem_name": problem_name,
            "ppa_metrics": ppa_metrics,
        })
        with open(os.path.join(formatted_benchmark_dir, f"{problem_name}_ppa.txt"), "w") as f:
            # write ppa_metrics to f in csv format
            f.write("tns,wns,eff_clk_period,power,area\n")
            f.write(f"{ppa_metrics['tns']},{ppa_metrics['wns']},{ppa_metrics['eff_clk_period']},{ppa_metrics['power']},{ppa_metrics['area']}")


    print(results)