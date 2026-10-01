"""
This script is used to split the existing dataset by gate count.
The problems counts larger than 50 are copied to the 'opt' directory.
The gate counts are read from the 'scripts/RTLLM.csv' and 'scripts/VerilogEval-Spec-to-RTL.csv' files.
The datasets are located at 'data/bench/RTLLM' and 'data/bench/VerilogEval-Spec-to-RTL'.
"""

import os
import shutil
import json


def filter_metadata(source_dir, dest_dir, valid_problems):
    # Filter problems.txt
    problems_txt_path = os.path.join(source_dir, "problems.txt")
    if os.path.exists(problems_txt_path):
        with open(problems_txt_path, "r") as f:
            problems = [line.strip() for line in f if line.strip() in valid_problems]
        with open(os.path.join(dest_dir, "problems.txt"), "w") as f:
            for p in problems:
                f.write(p + "\n")

    # Filter synthesis_top_module_names.json
    json_path = os.path.join(source_dir, "synthesis_top_module_names.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        filtered_data = {k: v for k, v in data.items() if k in valid_problems}
        with open(os.path.join(dest_dir, "synthesis_top_module_names.json"), "w") as f:
            json.dump(filtered_data, f, indent=4)


def main():
    # Define the paths
    rtllm_dataset_path = "data/bench/RTLLM"
    verilog_eval_dataset_path = "data/bench/VerilogEval-Spec-to-RTL"
    verilog_eval_dc_dataset_path = "data/bench/VerilogEval-Spec-to-RTL-DC"
    rtllm_opt_dataset_path = "data/bench/RTLLM_opt"
    verilog_eval_opt_dataset_path = "data/bench/VerilogEval-Spec-to-RTL_opt"
    verilog_eval_dc_opt_dataset_path = "data/bench/VerilogEval-Spec-to-RTL-DC_opt"

    # Create the output directory
    os.makedirs(rtllm_opt_dataset_path, exist_ok=True)
    os.makedirs(verilog_eval_opt_dataset_path, exist_ok=True)
    os.makedirs(verilog_eval_dc_opt_dataset_path, exist_ok=True)

    # File suffixes to copy
    suffixes = ["_ppa.txt", "_prompt.txt", "_ref.sv", "_test.sv"]

    # Read the gate counts from RTLLM.csv in scripts directory
    rtllm_gate_counts = {}
    rtllm_csv_path = "scripts/RTLLM.csv"
    if os.path.exists(rtllm_csv_path):
        with open(rtllm_csv_path, "r") as f:
            next(f)  # Skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                problem_name = parts[0]
                try:
                    gate_count = int(parts[1])
                    rtllm_gate_counts[problem_name] = gate_count
                except ValueError:
                    continue

    # Read the gate counts from VerilogEval-Spec-to-RTL.csv in scripts directory
    verilog_eval_gate_counts = {}
    verilog_eval_csv_path = "scripts/VerilogEval-Spec-to-RTL.csv"
    if os.path.exists(verilog_eval_csv_path):
        with open(verilog_eval_csv_path, "r") as f:
            next(f)  # Skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                problem_name = parts[0]
                try:
                    gate_count = int(parts[1])
                    verilog_eval_gate_counts[problem_name] = gate_count
                except ValueError:
                    continue

    # Copy problems with gate count > 50 to the opt directory
    rtllm_opt_problems = []
    for problem_name, gate_count in rtllm_gate_counts.items():
        if gate_count > 50:
            rtllm_opt_problems.append(problem_name)
            for suffix in suffixes:
                source_file = os.path.join(rtllm_dataset_path, problem_name + suffix)
                destination_file = os.path.join(rtllm_opt_dataset_path, problem_name + suffix)
                if os.path.exists(source_file):
                    shutil.copy2(source_file, destination_file)

    verilog_eval_opt_problems = []
    for problem_name, gate_count in verilog_eval_gate_counts.items():
        if gate_count > 50:
            verilog_eval_opt_problems.append(problem_name)
            for suffix in suffixes:
                source_file = os.path.join(verilog_eval_dataset_path, problem_name + suffix)
                destination_file = os.path.join(verilog_eval_opt_dataset_path, problem_name + suffix)
                if os.path.exists(source_file):
                    shutil.copy2(source_file, destination_file)

    verilog_eval_dc_opt_problems = []
    for problem_name, gate_count in verilog_eval_gate_counts.items():
        if gate_count > 50:
            verilog_eval_dc_opt_problems.append(problem_name)
            for suffix in suffixes:
                source_file = os.path.join(verilog_eval_dc_dataset_path, problem_name + suffix)
                destination_file = os.path.join(verilog_eval_dc_opt_dataset_path, problem_name + suffix)
                if os.path.exists(source_file):
                    shutil.copy2(source_file, destination_file)

    # Filter and write metadata files
    filter_metadata(rtllm_dataset_path, rtllm_opt_dataset_path, rtllm_opt_problems)
    filter_metadata(verilog_eval_dataset_path, verilog_eval_opt_dataset_path, verilog_eval_opt_problems)
    filter_metadata(verilog_eval_dc_dataset_path, verilog_eval_dc_opt_dataset_path, verilog_eval_dc_opt_problems)


if __name__ == "__main__":
    main()


