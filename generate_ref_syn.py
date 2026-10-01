import argparse
import os
import glob
import subprocess
import re

def get_first_module_name(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Find the very first module definition
    # Module could be declared as `module module_name (` or `module module_name#(` or `module module_name;`
    match = re.search(r'\bmodule\s+([a-zA-Z_0-9]+)\b', content)
    if match:
        return match.group(1)
    return None

def main():
    parser = argparse.ArgumentParser(description="Synthesize caller-supplied reference RTL.")
    parser.add_argument("--benchmark", nargs="+", required=True)
    parser.add_argument("--pdk-root", required=True)
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bench_dirs = args.benchmark
    
    yosys_ref = os.path.join(base_dir, "scripts", "ref", "ref.yosys.tcl")
    pdk_path = os.path.abspath(args.pdk_root)
    ref_dir_path = os.path.join(base_dir, "scripts", "ref")
    
    count = 0
    fail_count = 0
    
    for bdir in bench_dirs:
        if not os.path.exists(bdir):
            continue
            
        print(f"Scanning {bdir}...")
        
        # Read all files carefully, walking the directory
        for root, dirs, files in os.walk(bdir):
            for file in files:
                if file.endswith("ref.sv"):
                    ref_sv = os.path.join(root, file)
                    ref_syn_sv = ref_sv.replace(".sv", ".syn.sv")
                    
                    # For reference files, the top module is typically the first one defined in the file
                    module_name = get_first_module_name(ref_sv)
                        
                    if not module_name:
                        print(f"Could not find module in {ref_sv}")
                        continue
                        
                    # write yosys script to temp
                    yosys_script = os.path.join(root, "run_yosys_tmp.tcl")
                    
                    with open(yosys_ref, "r") as infile, open(yosys_script, "w") as outfile:
                        text = infile.read()
                        text = text.replace("__VERILOG_FILE__", os.path.abspath(ref_sv))
                        text = text.replace("__MODULE_NAME__", module_name)
                        text = text.replace("__OUTPUT_DIR__", os.path.abspath(root))
                        text = text.replace("__OUTPUT_FILE__", os.path.abspath(ref_syn_sv))
                        text = text.replace("__REF_DIR__", ref_dir_path)
                        text = text.replace("__PDK_DIR__", os.path.abspath(pdk_path))
                        # 0.01ns -> 10ps default in synthesis script setup
                        text = text.replace("__CLK_PERIOD__", str(10.0))
                        outfile.write(text)
                        
                    # print(f"Running Yosys for {ref_sv} (module: {module_name})")
                    
                    cmd = f"yosys {yosys_script}"
                    process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    if process.returncode != 0:
                        print(f"Error running yosys for {ref_sv} with module={module_name}")
                        print(process.stderr.decode("utf-8")[:500])
                        fail_count += 1
                    else:
                        print(f"Success -> {ref_syn_sv}")
                        count += 1
                    
                    if os.path.exists(yosys_script):
                        os.remove(yosys_script)

    print(f"Successfully synthesized {count} files. Failed: {fail_count}")

if __name__ == "__main__":
    main()
