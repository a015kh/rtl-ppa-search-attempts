import json
import os
import re
import shutil
import subprocess
from typing import Any, Literal


class VerilogEvaluator:
    """
    Handles the compilation and simulation of Verilog files using Icarus Verilog.

    This class provides a standardized way to run Verilog simulations,
    capturing outputs, handling errors, and managing timeouts.
    """

    def __init__(self, iverilog_executable_path: str, vvp_executable_path: str) -> None:
        if not shutil.which(iverilog_executable_path):
            raise FileNotFoundError(
                f"Icarus Verilog executable (iverilog) not found or not executable at: {iverilog_executable_path}. "
                f"Please provide a valid absolute path or ensure it's in your system PATH."
            )
        self.iverilog_executable: str = iverilog_executable_path

        if not shutil.which(vvp_executable_path):
            raise FileNotFoundError(
                f"Icarus Verilog runtime (vvp) not found or not executable at: {vvp_executable_path}. "
                f"Please provide a valid absolute path or ensure it's in your system PATH."
            )
        self.vvp_executable: str = vvp_executable_path

        # Base flags for iverilog compilation
        self.base_iverilog_flags: list[str] = [
            "-Wall",
            "-Winfloop",
            "-Wno-timescale",
            "-g2012",
        ]

    def evaluate(
        self,
        generated_sv_file: str | list[str],
        test_sv_file: str,
        ref_sv_file: str | None,
        top_module_name: str = "tb",
        output_directory: str | None = None,
        simulation_timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        """
        Compiles and simulates the given Verilog files.
        The generated Verilog file(s) is compiled to generate a `.vvp` file,
        then runs the simulation using the provided testbench file.

        :param generated_sv_file: Path to the generated Verilog file, or a list of files
            (e.g., DUT + PDK cells). If a list is provided, the first file is assumed to
            be the primary DUT used for naming outputs.
        :type generated_sv_file: str | list[str]

        :param test_sv_file: Path to the Verilog testbench file.
        :type test_sv_file: str

        :param ref_sv_file: Optional path to a reference Verilog file for simulation comparison.
        :type ref_sv_file: str | None

        :param top_module_name: Name of the top-level module in the testbench. Defaults to "tb".
        :type top_module_name: str

        :param output_directory: Directory to store compiled outputs and logs.
            Defaults to the directory of ``generated_sv_file``.
        :type output_directory: str | None

        :param simulation_timeout_seconds: Timeout for simulation execution in seconds. Defaults to 60.
        :type simulation_timeout_seconds: int

        :return: Dictionary containing the following keys:

            - ``status`` (str): One of "success", "compilation_error", "simulation_error", "simulation_timeout", or "file_error".
            - ``log_file_path`` (str | None): Path to the full simulation log.
            - ``compiled_file_path`` (str | None): Path to the compiled `.vvp` file, or None if compilation failed.
            - ``compilation_stdout`` (str): Standard output from the compiler.
            - ``compilation_stderr`` (str): Standard error from the compiler.
            - ``simulation_stdout`` (str): Standard output from the simulator.
            - ``simulation_stderr`` (str): Standard error from the simulator.
        :rtype: dict[str, Any]
        """

        # --- 1. Determine paths and prepare ---
        # Handle single file or list of files for test_sv_file
        if isinstance(generated_sv_file, str):
            dut_files = [generated_sv_file]
            output_basename = os.path.splitext(os.path.basename(generated_sv_file))[0]
            if not os.path.isfile(generated_sv_file):
                return self._format_result(
                    "file_error",
                    log_file_path=None,
                    compiled_file_path=None,
                    comp_stderr=f"Generated Verilog file not found: {generated_sv_file}",
                )
            # Determine paths using the string
            if output_directory is None:
                actual_output_dir = os.path.dirname(generated_sv_file)
            else:
                actual_output_dir = output_directory
            output_basename = os.path.splitext(os.path.basename(generated_sv_file))[0]

        elif isinstance(generated_sv_file, list):
            dut_files = generated_sv_file
            # The first file in the list is considered the generated Verilog file and used to determine the output basename
            main_dut_file = dut_files[0]
            # Check for duplicates within the list while ensuring the first file is always included first
            dut_files = list(
                dict.fromkeys(dut_files)
            )  # Removes duplicates while preserving order
            for f in dut_files:
                if not os.path.isfile(f):
                    return self._format_result(
                        "file_error",
                        log_file_path=None,
                        compiled_file_path=None,
                        comp_stderr=f"Additional Verilog file not found: {f}",
                    )

            # Determine paths using the FIRST element of the list
            if output_directory is None:
                actual_output_dir = os.path.dirname(main_dut_file)
            else:
                actual_output_dir = output_directory
            output_basename = os.path.splitext(os.path.basename(main_dut_file))[0]
        else:
            return self._format_result(
                "file_error",
                log_file_path=None,
                compiled_file_path=None,
                comp_stderr="Invalid type for generated_sv_file. Expected str or list of str.",
            )

        if not os.path.isfile(test_sv_file):
            return self._format_result(
                "file_error",
                log_file_path=None,
                compiled_file_path=None,
                comp_stderr=f"Test Verilog file not found: {test_sv_file}",
            )
        # Check if the reference file is provided and exists
        if ref_sv_file is None:
            pass
        elif not os.path.isfile(ref_sv_file):
            # If ref_sv_file is not None, it should be a valid file path
            # If it is None, we skip this check
            return self._format_result(
                "file_error",
                log_file_path=None,
                compiled_file_path=None,
                comp_stderr=f"Reference Verilog file not found: {ref_sv_file}",
            )

        os.makedirs(actual_output_dir, exist_ok=True)
        compiled_vvp_file = os.path.join(
            actual_output_dir, output_basename + "_compiled.vvp"
        )
        log_file = os.path.join(actual_output_dir, output_basename + "_simulation.log")

        # Initialize result components
        comp_stdout, comp_stderr = "", ""
        sim_stdout, sim_stderr = "", ""

        # --- 2. Compile Verilog files ---
        compile_cmd_list = [self.iverilog_executable]
        compile_cmd_list.extend(self.base_iverilog_flags)
        compile_cmd_list.extend(["-s", top_module_name])  # Specify top module
        compile_cmd_list.extend(["-o", compiled_vvp_file])
        compile_cmd_list.extend(dut_files)  # Add the generated Verilog file(s)
        # Add logic for testing for duplicate files, if the same file is given multiple times, it should be added only once
        # This guard is necessary as when applying this function for reference files ref_sv_file might be the same as generated_sv_file,
        # So we need to ensure we don't add it multiple times. To avoid simulation errors.
        if test_sv_file not in dut_files:
            compile_cmd_list.append(test_sv_file)
        if ref_sv_file not in dut_files:
            # Skip adding ref_sv_file if it is None
            # This is to avoid errors when the reference file is not provided
            if ref_sv_file is not None:
                compile_cmd_list.append(ref_sv_file)

        print(f"INFO: Compile command: {' '.join(compile_cmd_list)}")
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write("--- Compilation Phase ---\n")
            lf.write(f"Command: {' '.join(compile_cmd_list)}\n\n")
            try:
                compile_process = subprocess.run(
                    compile_cmd_list,
                    capture_output=True,
                    text=True,
                    check=False,  # Do not raise exception on non-zero exit
                )
                comp_stdout = compile_process.stdout or ""
                comp_stderr = compile_process.stderr or ""

                lf.write(f"Return Code: {compile_process.returncode}\n")
                lf.write("Stdout:\n")
                lf.write(comp_stdout + "\n")
                lf.write("Stderr:\n")
                lf.write(comp_stderr + "\n")

                if compile_process.returncode != 0:
                    print(f"ERROR: Compilation failed. See {log_file} for details.")
                    return self._format_result(
                        "compilation_error", log_file, None, comp_stdout, comp_stderr
                    )
                print(f"INFO: Compilation successful. Output: {compiled_vvp_file}")

            except FileNotFoundError:
                # This case should ideally be caught by __init__, but as a safeguard:
                error_msg = f"Icarus Verilog executable (iverilog) not found during compilation. Path: {self.iverilog_executable}"
                print(f"ERROR: {error_msg}")
                lf.write(f"CRITICAL ERROR: {error_msg}\n")
                return self._format_result(
                    "file_error", log_file, None, comp_stderr=error_msg
                )
            except Exception as e:
                error_msg = f"An unexpected error occurred during compilation: {e}"
                print(f"ERROR: {error_msg}")
                lf.write(f"CRITICAL ERROR: {error_msg}\n")
                return self._format_result(
                    "compilation_error", log_file, None, comp_stderr=error_msg
                )

            lf.write("\n--- Simulation Phase ---\n")
            # --- 3. Simulate the compiled VVP file ---
            # The compiled .vvp file is typically made executable by iverilog using a shebang
            # pointing to the vvp runtime.
            if os.name != "nt":  # On non-Windows systems, ensure it's executable
                try:
                    os.chmod(compiled_vvp_file, 0o755)  # rwxr-xr-x
                except OSError as e:
                    print(
                        f"WARNING: Could not set execute permission on {compiled_vvp_file}: {e}"
                    )
                    lf.write(
                        f"WARNING: Could not set execute permission on {compiled_vvp_file}: {e}\n"
                    )

            # Execute using the vvp runtime directly
            # Use an absolute path here because the simulator runs from
            # ``simulation_working_dir`` and the compiled file path may be
            # relative to the project root.
            run_cmd_list = [self.vvp_executable, os.path.abspath(compiled_vvp_file)]
            print(f"INFO: Simulation command: {' '.join(run_cmd_list)}")
            lf.write(f"Command: {' '.join(run_cmd_list)}\n\n")

            # Set the working directory to the location of the testbench file (this is to include the miscellaneous files sometimes required by the testbench)
            # Some modules in RTLLM have files that supply the input and output files for the testbench
            # Examples: Prob013_test_data.dat, Prob026_asyn_fifo_tdata.txt, Prob026_asyn_fifo_rempty.txt,
            # Prob026_asyn_fifo_wfull.txt, Prob035_calendar_reference.txt, Prob045_alu_reference.dat,
            # Prob049_signal_generator_tri_gen.txt
            simulation_working_dir = os.path.dirname(dut_files[0])
            print(f"INFO: Running simulation in directory: {simulation_working_dir}")
            lf.write(f"Working Directory: {simulation_working_dir}\n\n")

            try:
                run_process = subprocess.run(
                    run_cmd_list,
                    capture_output=True,
                    text=True,
                    timeout=simulation_timeout_seconds,
                    check=False,  # Do not raise exception on non-zero exit
                    cwd=simulation_working_dir,  # Set the working directory for the simulation
                )
                sim_stdout = run_process.stdout or ""
                sim_stderr = run_process.stderr or ""

                lf.write(f"Return Code: {run_process.returncode}\n")
                lf.write("Stdout:\n")
                lf.write(sim_stdout + "\n")
                lf.write("Stderr:\n")
                lf.write(sim_stderr + "\n")

                if run_process.returncode == 0:
                    print(f"INFO: Simulation successful. See {log_file} for details.")
                    return self._format_result(
                        "success",
                        log_file,
                        compiled_vvp_file,
                        comp_stdout,
                        comp_stderr,
                        sim_stdout,
                        sim_stderr,
                    )
                else:
                    # Non-zero return code could be due to $finish(X) with X!=0, or runtime errors.
                    print(
                        f"ERROR: Simulation finished with non-zero status ({run_process.returncode}). See {log_file} for details."
                    )
                    return self._format_result(
                        "simulation_error",
                        log_file,
                        compiled_vvp_file,
                        comp_stdout,
                        comp_stderr,
                        sim_stdout,
                        sim_stderr,
                    )

            except subprocess.TimeoutExpired:
                timeout_msg = (
                    f"Simulation timed out after {simulation_timeout_seconds} seconds."
                )
                print(f"ERROR: {timeout_msg}")
                lf.write(f"TIMEOUT ERROR: {timeout_msg}\n")
                # Capture any partial output before timeout
                # (Note: subprocess.run with timeout might not populate stdout/stderr for timed-out process easily)
                return self._format_result(
                    "simulation_timeout",
                    log_file,
                    compiled_vvp_file,
                    comp_stdout,
                    comp_stderr,
                    sim_stderr=timeout_msg,
                )
            except FileNotFoundError:
                # This case should ideally be caught by __init__, but as a safeguard:
                error_msg = f"vvp executable not found during simulation. Path: {self.vvp_executable}"
                print(f"ERROR: {error_msg}")
                lf.write(f"CRITICAL ERROR: {error_msg}\n")
                return self._format_result(
                    "file_error",
                    log_file,
                    compiled_vvp_file,
                    comp_stdout,
                    comp_stderr,
                    sim_stderr=error_msg,
                )
            except Exception as e:
                error_msg = f"An unexpected error occurred during simulation: {e}"
                print(f"ERROR: {error_msg}")
                lf.write(f"CRITICAL ERROR: {error_msg}\n")
                return self._format_result(
                    "simulation_error",
                    log_file,
                    compiled_vvp_file,
                    comp_stdout,
                    comp_stderr,
                    sim_stderr=error_msg,
                )

    def _format_result(
        self,
        status: Literal[
            "success",
            "compilation_error",
            "simulation_error",
            "simulation_timeout",
            "file_error",
        ],
        log_file_path: str | None,
        compiled_file_path: str | None,
        comp_stdout: str = "",
        comp_stderr: str = "",
        sim_stdout: str = "",
        sim_stderr: str = "",
    ) -> dict[str, str | None]:
        """Helper method to format the return dictionary.

        :param status: Status of the evaluation.
        :type status: Literal["success", "compilation_error", "simulation_error", "simulation_timeout", "file_error"]
        :param log_file_path: Path to the log file.
        :type log_file_path: str | None
        :param compiled_file_path: Path to the compiled VVP file.
        :type compiled_file_path: str | None
        :param comp_stdout: Standard output from the compilation.
        :type comp_stdout: str
        :param comp_stderr: Standard error from the compilation.
        :type comp_stderr: str
        :param sim_stdout: Standard output from the simulation.
        :type sim_stdout: str
        :param sim_stderr: Standard error from the simulation.
        :type sim_stderr: str
        :return: A dictionary containing the evaluation results.
        :rtype: dict[str, str | None]
        """
        return {
            "status": status,
            "log_file_path": log_file_path,
            "compiled_file_path": compiled_file_path,
            "compilation_stdout": comp_stdout,
            "compilation_stderr": comp_stderr,
            "simulation_stdout": sim_stdout,
            "simulation_stderr": sim_stderr,
        }


class VerilogEvaluatorICRTL(VerilogEvaluator):
    """
    Verilog evaluator specialized for the ICRTL benchmark.

    ICRTL uses problem-local directory layouts such as ``00_TB/test.sv`` and
    testbenches that report success through printed summaries like
    ``All tests PASS!`` rather than the RTLLM/VerilogEval markers.
    """

    def _get_icrtl_working_directory(self, test_sv_file: str) -> str:
        """
        Resolve the simulation working directory expected by ICRTL testbenches.

        The open-source ICRTL ``01_run.sh`` scripts execute from the problem
        root, not from the DUT output directory. This is required because the
        testbenches reference assets using paths like ``./00_TB/...``.
        """

        test_dir = os.path.dirname(os.path.abspath(test_sv_file))
        if os.path.basename(test_dir) == "00_TB":
            return os.path.dirname(test_dir)
        return test_dir

    def _parse_icrtl_simulation_output(self, simulation_stdout: str) -> dict[str, Any]:
        """
        Parse pass/fail and latency information from ICRTL simulation output.
        """

        passed = "All tests PASS!" in simulation_stdout

        total_error = None
        total_error_match = re.search(r"Total Error:\s*(\d+)", simulation_stdout)
        if total_error_match:
            total_error = int(total_error_match.group(1))

        latency_cycles = None
        latency_match = re.search(r"total time:\s*(\d+)\s*cycles", simulation_stdout)
        if latency_match:
            latency_cycles = int(latency_match.group(1))

        return {
            "passed": passed,
            "total_error": total_error,
            "latency_cycles": latency_cycles,
        }

    def evaluate(
        self,
        generated_sv_file: str | list[str],
        test_sv_file: str,
        ref_sv_file: str | None = None,
        top_module_name: str = "test",
        output_directory: str | None = None,
        simulation_timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        """
        Compile and simulate an ICRTL design.

        Differences from ``VerilogEvaluator``:
        - Defaults the testbench top module to ``test``.
        - Runs the simulation from the benchmark problem directory so relative
          paths in ``00_TB`` resolve correctly.
        - Interprets ICRTL's printed summary as the functional pass/fail signal.
        """

        if isinstance(generated_sv_file, str):
            dut_files = [generated_sv_file]
            if not os.path.isfile(generated_sv_file):
                return self._format_result(
                    "file_error",
                    log_file_path=None,
                    compiled_file_path=None,
                    comp_stderr=f"Generated Verilog file not found: {generated_sv_file}",
                )
            main_dut_file = generated_sv_file
        elif isinstance(generated_sv_file, list):
            dut_files = list(dict.fromkeys(generated_sv_file))
            if not dut_files:
                return self._format_result(
                    "file_error",
                    log_file_path=None,
                    compiled_file_path=None,
                    comp_stderr="No generated Verilog files were provided.",
                )
            for file_path in dut_files:
                if not os.path.isfile(file_path):
                    return self._format_result(
                        "file_error",
                        log_file_path=None,
                        compiled_file_path=None,
                        comp_stderr=f"Additional Verilog file not found: {file_path}",
                    )
            main_dut_file = dut_files[0]
        else:
            return self._format_result(
                "file_error",
                log_file_path=None,
                compiled_file_path=None,
                comp_stderr="Invalid type for generated_sv_file. Expected str or list of str.",
            )

        if not os.path.isfile(test_sv_file):
            return self._format_result(
                "file_error",
                log_file_path=None,
                compiled_file_path=None,
                comp_stderr=f"Test Verilog file not found: {test_sv_file}",
            )
        if ref_sv_file is not None and not os.path.isfile(ref_sv_file):
            return self._format_result(
                "file_error",
                log_file_path=None,
                compiled_file_path=None,
                comp_stderr=f"Reference Verilog file not found: {ref_sv_file}",
            )

        actual_output_dir = (
            output_directory
            if output_directory is not None
            else os.path.dirname(os.path.abspath(main_dut_file))
        )
        os.makedirs(actual_output_dir, exist_ok=True)

        output_basename = os.path.splitext(os.path.basename(main_dut_file))[0]
        compiled_vvp_file = os.path.join(
            actual_output_dir, f"{output_basename}_compiled.vvp"
        )
        log_file = os.path.join(actual_output_dir, f"{output_basename}_simulation.log")

        comp_stdout, comp_stderr = "", ""
        sim_stdout, sim_stderr = "", ""

        compile_cmd_list = [self.iverilog_executable]
        compile_cmd_list.extend(self.base_iverilog_flags)
        compile_cmd_list.extend(["-s", top_module_name])
        compile_cmd_list.extend(["-o", compiled_vvp_file])
        compile_cmd_list.extend(dut_files)
        if test_sv_file not in dut_files:
            compile_cmd_list.append(test_sv_file)
        # if ref_sv_file is not None and ref_sv_file not in dut_files:
        #     compile_cmd_list.append(ref_sv_file)

        simulation_working_dir = self._get_icrtl_working_directory(test_sv_file)

        print(f"INFO: ICRTL compile command: {' '.join(compile_cmd_list)}")
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write("--- Compilation Phase ---\n")
            lf.write(f"Command: {' '.join(compile_cmd_list)}\n\n")
            try:
                compile_process = subprocess.run(
                    compile_cmd_list,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                comp_stdout = compile_process.stdout or ""
                comp_stderr = compile_process.stderr or ""

                lf.write(f"Return Code: {compile_process.returncode}\n")
                lf.write("Stdout:\n")
                lf.write(comp_stdout + "\n")
                lf.write("Stderr:\n")
                lf.write(comp_stderr + "\n")
                print(compile_process.returncode)

                if compile_process.returncode != 0:
                    return self._format_result(
                        "compilation_error", log_file, None, comp_stdout, comp_stderr
                    )
            except FileNotFoundError:
                error_msg = f"Icarus Verilog executable (iverilog) not found during compilation. Path: {self.iverilog_executable}"
                lf.write(f"CRITICAL ERROR: {error_msg}\n")
                return self._format_result(
                    "file_error", log_file, None, comp_stderr=error_msg
                )
            except Exception as e:
                error_msg = f"An unexpected error occurred during compilation: {e}"
                lf.write(f"CRITICAL ERROR: {error_msg}\n")
                return self._format_result(
                    "compilation_error", log_file, None, comp_stderr=error_msg
                )

            lf.write("\n--- Simulation Phase ---\n")
            if os.name != "nt":
                try:
                    os.chmod(compiled_vvp_file, 0o755)
                except OSError as e:
                    lf.write(
                        f"WARNING: Could not set execute permission on {compiled_vvp_file}: {e}\n"
                    )

            # Use an absolute path here because the simulator runs from
            # ``simulation_working_dir`` and the compiled file path may be
            # relative to the project root.
            run_cmd_list = [self.vvp_executable, os.path.abspath(compiled_vvp_file)]
            lf.write(f"Command: {' '.join(run_cmd_list)}\n")
            lf.write(f"Working Directory: {simulation_working_dir}\n\n")

            try:
                run_process = subprocess.run(
                    run_cmd_list,
                    capture_output=True,
                    text=True,
                    timeout=simulation_timeout_seconds,
                    check=False,
                    cwd=simulation_working_dir,
                )
                sim_stdout = run_process.stdout or ""
                sim_stderr = run_process.stderr or ""

                lf.write(f"Return Code: {run_process.returncode}\n")
                lf.write("Stdout:\n")
                lf.write(sim_stdout + "\n")
                lf.write("Stderr:\n")
                lf.write(sim_stderr + "\n")

                parsed_output = self._parse_icrtl_simulation_output(sim_stdout)
                lf.write(f"Parsed output: {parsed_output}\n")

                result = self._format_result(
                    "success"
                    if run_process.returncode == 0 and parsed_output["passed"]
                    else "simulation_error",
                    log_file,
                    compiled_vvp_file,
                    comp_stdout,
                    comp_stderr,
                    sim_stdout,
                    sim_stderr,
                )
                result.update(parsed_output)
                lf.write(f"Result: {result}\n")
                return result
            except subprocess.TimeoutExpired:
                timeout_msg = (
                    f"Simulation timed out after {simulation_timeout_seconds} seconds."
                )
                lf.write(f"TIMEOUT ERROR: {timeout_msg}\n")
                result = self._format_result(
                    "simulation_timeout",
                    log_file,
                    compiled_vvp_file,
                    comp_stdout,
                    comp_stderr,
                    sim_stderr=timeout_msg,
                )
                result.update(
                    {"passed": False, "total_error": None, "latency_cycles": None}
                )
                return result
            except FileNotFoundError:
                error_msg = f"vvp executable not found during simulation. Path: {self.vvp_executable}"
                lf.write(f"CRITICAL ERROR: {error_msg}\n")
                result = self._format_result(
                    "file_error",
                    log_file,
                    compiled_vvp_file,
                    comp_stdout,
                    comp_stderr,
                    sim_stderr=error_msg,
                )
                result.update(
                    {"passed": False, "total_error": None, "latency_cycles": None}
                )
                return result
            except Exception as e:
                error_msg = f"An unexpected error occurred during simulation: {e}"
                lf.write(f"CRITICAL ERROR: {error_msg}\n")
                result = self._format_result(
                    "simulation_error",
                    log_file,
                    compiled_vvp_file,
                    comp_stdout,
                    comp_stderr,
                    sim_stderr=error_msg,
                )
                result.update(
                    {"passed": False, "total_error": None, "latency_cycles": None}
                )
                return result


class SynthesisEvaluator:
    """
    Manages the hardware synthesis and PPA evaluation flow using Yosys and OpenROAD.
    """

    def __init__(
        self,
        yosys_path: str = "yosys",
        openroad_path: str = "openroad",
        pdk_path: str | None = None,
    ) -> None:
        """
        Initialize the synthesis evaluator with toolchain paths and environment configuration.

        :param yosys_path: Path to the Yosys binary.
        :param openroad_path: Path to the OpenROAD binary.
        :param pdk_path: Path to the standard cell PDK directory.
        """
        self.yosys_path: str = yosys_path
        self.openroad_path: str = openroad_path
        selected_pdk = pdk_path or os.environ.get("REVOLUTION_PDK_ROOT")
        if not selected_pdk:
            raise ValueError("Provide pdk_path or REVOLUTION_PDK_ROOT; cell libraries are not bundled.")
        self.pdk_path: str = os.path.abspath(os.path.expanduser(selected_pdk))
        if not os.path.isdir(self.pdk_path):
            raise ValueError("The supplied PDK directory does not exist: " + self.pdk_path)

        # Synthesis clk period in nanoseconds
        self.clk_period: float = 0.01  # ns

        # Get the directory where this script is located (./src/revolution/evaluation.py)
        script_main_dir = os.path.dirname(os.path.abspath(__file__))
        # Generic flow scripts are packaged; cell libraries remain external.
        self.script_root_dir: str = os.path.abspath(
            os.path.join(script_main_dir, "resources")
        )
        # The ref directory is inside the script root
        self.ref_dir_path: str = os.path.join(self.script_root_dir, "scripts", "ref")
        # Cell libraries are supplied by the caller; never select a bundled PDK.

        # Print directories for debugging
        print(f"Script Main Directory: {script_main_dir}")
        print(f"Script Root Directory: {self.script_root_dir}")
        print(f"Reference Directory: {self.ref_dir_path}")
        print(f"PDK Directory: {self.pdk_path}")

    def evaluate(
        self,
        verilog_file: str,
        problem_name: str,
        synth_top_module_name: str,
        output_directory: str,
        report_base_path: str,
        verilog_evaluator: VerilogEvaluator,
        test_sv_file: str,
        ref_sv_file: str | None,
    ) -> dict[str, bool | str | None | dict[str, Any]]:
        """
        Performs synthesis, PPA analysis, and post-synthesis verification.
        Run Yosys synthesis script, to convert the Verilog file to a synthesized netlist.
        Then uses VerilogEvaluator to run a functional simulation on the synthesized netlist
        using the provided testbench and reference design (if available).
        (Depending on benchmark testbench requires a reference design to compare against).
        Finally, uses the netlist to run OpenROAD to perform PPA analysis.

        :param verilog_file: Path to the input Verilog file.
        :param problem_name: Name of the problem/design.
        :param synth_top_module_name: Name of the top-level module for synthesis.
        :param output_directory: Directory for all generated files.
        :param report_base_path: Base path for naming report files.
        :param verilog_evaluator: An instance of VerilogEvaluator for simulation.
        :param test_sv_file: Path to the testbench for functional verification.
        :param ref_sv_file: Path to the reference design for the testbench (optional).
        :return: A dictionary containing the synthesis and PPA results.
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        synthesized_netlist_path = verilog_file.replace(".sv", ".syn.v")

        synthesis_success, synthesis_log = self._run_synthesis(
            verilog_file,
            problem_name,
            synth_top_module_name,
            output_directory,
            report_base_path,
            synthesized_netlist_path,
        )

        if not synthesis_success:
            return {
                "synthesis_success": False,
                "synthesis_functionality_success": False,  # No functionality test if synthesis failed
                "ppa_success": False,
                "synthesis_log": synthesis_log,
                "ppa_metrics": None,
            }

        # After synthesis add post-synthesis functionality test
        synthesis_functionality_success, func_check_log = (
            self._check_synthesis_functionality(
                synthesized_netlist_path,
                test_sv_file,
                ref_sv_file,
                "tb",  # Assuming the top module name for the testbench is "tb"
                output_directory,
                verilog_evaluator,
            )
        )

        if not synthesis_functionality_success:
            return {
                "synthesis_success": True,
                "synthesis_functionality_success": False,
                "ppa_success": False,
                "synthesis_log": f"{synthesis_log}\n\n--- Post-Synthesis Functional Verification Log ---\n{func_check_log}",
                "ppa_metrics": None,
            }

        ppa_metrics = self._parse_ppa_log(synthesis_log)

        return {
            "synthesis_success": True,
            "synthesis_functionality_success": True,
            "ppa_success": True,
            "synthesis_log": synthesis_log,
            "ppa_metrics": ppa_metrics,
        }

    def _run_synthesis(
        self,
        verilog_file: str,
        problem_name: str,
        synth_top_module_name: str,
        output_directory: str,
        report_base_path: str,
        synthesized_netlist_path: str,
    ) -> tuple[bool, str]:
        """
        Runs the Yosys and OpenROAD synthesis script.

        :param verilog_file: Path to the input Verilog file.
        :param problem_name: Name of the problem/design.
        :param synth_top_module_name: Name of the top-level module for synthesis.
        :param output_directory: Directory for all generated files.
        :param report_base_path: Base path for naming report files.
        :param synthesized_netlist_path: Path to save the synthesized netlist.
        :return: A tuple containing a success boolean and the path to the report log.
        """

        clk_period = self.clk_period  # ns

        output_file = synthesized_netlist_path

        sdc_file_path = self._create_sdc_file(
            verilog_file, synth_top_module_name, output_directory, clk_period=clk_period
        )
        yosys_script_path = self._create_yosys_script(
            verilog_file,
            synth_top_module_name,
            output_directory,
            clk_period,
            output_file,
        )
        openroad_script_path = self._create_openroad_script(
            sdc_file_path, synth_top_module_name, output_directory, output_file
        )

        report_path = (
            report_base_path + "_synthesis_report.rpt"
        )  # os.path.join(output_directory, f"{problem_name}_synthesis_report.rpt")

        command = f"yosys {yosys_script_path} && openroad {openroad_script_path} | tee {report_path}"

        # log_path = os.path.join(output_directory, "yosys.log")

        print(f"INFO: Running synthesis command: {command}")

        process = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if process.returncode == 0:
            print(f"Synthesis completed successfully. Report saved to {report_path}")
            return True, report_path
        else:
            print(f"Synthesis failed. Error: {process.stderr.decode()}")
            # Return the path to the report even on failure to aid debugging
            with open(report_path, "a") as f:
                f.write("\n\n--- SYNTHESIS FAILED ---\n")
                f.write(process.stderr.decode())
            return False, report_path

    # Method for post-synthesis functionality check/verification
    def _check_synthesis_functionality(
        self,
        synthesized_netlist: str,
        test_sv: str,
        ref_sv: str | None,
        tb_top_module: str,
        output_dir: str,
        verilog_evaluator: VerilogEvaluator,
    ) -> tuple[bool, str]:
        """
        Runs a functional simulation on the synthesized netlist using the provided testbench.

        :param synthesized_netlist: Path to the synthesized netlist file.
        :param test_sv: Path to the testbench Verilog file.
        :param ref_sv: Path to the reference design Verilog file (optional).
        :param tb_top_module: Name of the top-level module in the testbench.
        :param output_dir: Directory for output files.
        :param verilog_evaluator: An instance of VerilogEvaluator for simulation.
        :return: A tuple containing a success boolean and the log of the simulation.
        """
        # need to include the verilog files from pdk for simulation of synthesized netlist
        pdk_verilog_lib = os.path.join(
            self.pdk_path, "Nangate45", "work_around_yosys", "cells.v"
        )
        if not os.path.exists(pdk_verilog_lib):
            error_msg = f"PDK Verilog library not found at: {pdk_verilog_lib}"
            print(f"ERROR: {error_msg}")
            return False, error_msg

        # The evaluator expects a list of files. The synthesized netlist replaces the original DUT.
        # The VerilogEvaluator's evaluate method has been slightly adapted to accept a list of files
        # Example: iverilog -Wall -Winfloop -Wno-timescale -g2012 -o compiled.vvp -s tb testbench.sv synthesized_netlist.syn.v pdk_verilog_lib.v
        sim_results = verilog_evaluator.evaluate(
            generated_sv_file=[
                synthesized_netlist,
                pdk_verilog_lib,
            ],  # Pass synthesized netlist and PDK lib
            test_sv_file=test_sv,
            ref_sv_file=ref_sv,
            top_module_name=tb_top_module,
            # Don't use output_directory here, if we pass the synthesized_netlist its .syn suffix will differentiate it from the rtl simulation
        )

        if sim_results["status"] == "compilation_error":
            print(
                f"Synthesis functionality check failed during compilation: {sim_results.get('compilation_stderr', 'Compilation log not available')}"
            )
        log = f"Compilation Log:\n{sim_results.get('compilation_stderr')}\n\nSimulation Log:\n{sim_results.get('simulation_stdout')}\n{sim_results.get('simulation_stderr')}"

        if sim_results["status"] == "success":
            output = sim_results.get("simulation_stdout", "")
            # Check for simulation output in two ways:
            # Looks for "Mismatches: 0" in the output (VerilogEvalv2)
            # or checks for "===========Your Design Passed===========" in the output (RTLLMv2)
            m_match = re.search(r"^Mismatches: (\d+)", output, re.M)
            if (
                m_match and int(m_match.group(1)) == 0
            ) or "===========Your Design Passed===========" in output:
                return True, log

        return False, log

    def _create_sdc_file(
        self,
        verilog_file: str,
        module_name: str,
        output_directory: str,
        clk_period: float,
    ) -> str:
        """
        Creates a simple SDC file for timing constraints.

        :param verilog_file: Path to the Verilog file to extract clock ports from.
        :param module_name: Name of the top-level module in the Verilog file.
        :param output_directory: Directory to save the SDC file.
        :param clk_period: Clock period in nanoseconds.
        :return: Path to the generated SDC file.
        """
        clk_ports = []
        clk_pattern = r"\b(clk|Clock|clock|Clk|CLK|CK|ck)\w*"

        with open(verilog_file, "r") as inFile:
            lines = inFile.read().split(";")
            for line in lines:
                if f"module {module_name}" in line:
                    ob = line.find("(")
                    cb = line.rfind(")")
                    matches = re.findall(clk_pattern, line[ob : cb - 1])
                    if matches:
                        clk_ports.extend(matches)
                    else:
                        clk_ports.append("f_clk")
                break

        sdc_lines = []
        sdc_lines.append(f"current_design {module_name}\n")
        sdc_lines.append("set clk_name clk\n")
        sdc_lines.append(f"set clk_period {clk_period}\n")
        for clk_port in clk_ports:
            sdc_lines.append(
                f"create_clock -name $clk_name -period $clk_period [get_ports {clk_port}]\n"
            )

        sdc_gen = f"{output_directory}/{module_name}.sdc"
        with open(sdc_gen, "w") as outfile:
            for sdc_line in sdc_lines:
                outfile.write(sdc_line)

        return sdc_gen

    def _create_yosys_script(
        self,
        verilog_file: str,
        module_name: str,
        output_directory: str,
        clk_period: float,
        output_file: str,
    ) -> str:
        """Creates a Yosys synthesis script from a template.

        :param verilog_file: Path to the Verilog file to be synthesized.
        :param module_name: Name of the top-level module in the Verilog file.
        :param output_directory: Directory to save the generated Yosys script.
        :param clk_period: Clock period in nanoseconds.
        :param output_file: Name of the output file for the synthesized netlist.
        :return: Path to the generated Yosys script.
        """
        yosys_ref = os.path.join(self.ref_dir_path, "ref.yosys.tcl")
        yosys_gen = f"{output_directory}/{module_name}.yosys.tcl"

        with open(yosys_ref, "r") as infile:
            with open(yosys_gen, "w") as outfile:
                text = infile.read()
                text = text.replace("__VERILOG_FILE__", os.path.abspath(verilog_file))
                text = text.replace(
                    "__MODULE_NAME__", module_name
                )  # Reverted to using module_name directly as we now extract it from the Verilog file
                text = text.replace("__OUTPUT_DIR__", os.path.abspath(output_directory))
                text = text.replace("__OUTPUT_FILE__", output_file)
                text = text.replace("__REF_DIR__", self.ref_dir_path)
                text = text.replace("__PDK_DIR__", os.path.abspath(self.pdk_path))
                text = text.replace("__CLK_PERIOD__", str(clk_period * 1000))
                outfile.write(text)

        return yosys_gen

    def _create_openroad_script(
        self,
        sdc_file_path: str,
        module_name: str,
        output_directory: str,
        output_file: str,
    ) -> str:
        """Creates an OpenROAD script from a template.

        :param sdc_file_path: Path to the SDC file for timing constraints.
        :param module_name: Name of the top-level module in the Verilog file.
        :param output_directory: Directory to save the generated OpenROAD script.
        :param output_file: Name of the output file for the synthesized netlist.
        :return: Path to the generated OpenROAD script.
        """
        # A simplified OpenROAD script. This may need to be adapted for your specific PDK and design.
        or_ref = os.path.join(self.ref_dir_path, "ref.openroad.tcl")
        or_gen = f"{output_directory}/{module_name}.openroad.tcl"

        with open(or_ref, "r") as infile:
            with open(or_gen, "w") as outfile:
                text = infile.read()
                text = text.replace(
                    "__UTIL_DIR__",
                    os.path.join(self.script_root_dir, "scripts", "util"),
                )
                text = text.replace("__PDK_DIR__", os.path.abspath(self.pdk_path))
                text = text.replace("__DESIGN_NAME__", module_name)
                text = text.replace(
                    "__MODULE_NAME__", module_name
                )  # Reverted to using module_name directly as we now extract it from the Verilog file
                text = text.replace("__NETLIST__", os.path.abspath(f"{output_file}"))
                text = text.replace("__SDC__", os.path.abspath(sdc_file_path))
                text = text.replace("__UTILIZATION__", str(0.5))
                outfile.write(text)

        return or_gen

    def _parse_ppa_log(self, report_path: str) -> dict[str, float | str | None]:
        """
        Parses an OpenROAD log to extract Power, Performance, and Area (PPA) metrics.

        :param report_path: Path to the synthesis report file.
        :return: A dictionary with PPA metrics, or None if parsing fails.
        """
        tns, wns, power, area = None, None, None, None

        try:
            with open(report_path, "r") as file:
                for line in file:
                    parts = line.split()
                    if not parts:  # Skip empty lines
                        continue

                    try:
                        # Handle TNS and WNS
                        if parts[0] == "tns":
                            tns = float(parts[2] if parts[1] == "max" else parts[1])
                        elif parts[0] == "wns":
                            wns = float(parts[2] if parts[1] == "max" else parts[1])
                        # Handle Power and Area
                        elif line.startswith("Total"):
                            power = float(parts[4])
                        elif line.startswith("Design area"):
                            area = float(parts[2])
                    except (ValueError, IndexError):
                        # Safely ignore lines that don't parse correctly
                        continue
        except FileNotFoundError:
            print(f"Error: PPA report file not found at {report_path}")
            return {
                "tns": None,
                "wns": None,
                "eff_clk_period": None,
                "power": None,
                "area": None,
                "report_path": None,
            }

        # Calculate effective clock period only if wns was found
        eff_clk_period = None
        if wns is not None:
            if wns < 0.0:  # Indicates that it is a sequential design
                eff_clk_period = self.clk_period - wns
            else:  # Indicates that it is a combinational design
                eff_clk_period = 0.0

        ppa_path = report_path.replace(".rpt", ".ppa")
        with open(ppa_path, "w") as f:
            f.write("tns,wns,eff_clk_period,power,area\n")
            f.write(f"{tns},{wns},{eff_clk_period},{power},{area}")

        return {
            "tns": tns,
            "wns": wns,
            "eff_clk_period": eff_clk_period,
            "power": power,
            "area": area,
            "report_path": ppa_path,
        }


class PPAProxyEvaluator:
    """
    Fast, zero-synthesis PPA estimator backed by a trained T5Gemma model.

    The model was produced by ``src/text2text_regression/finetune_ppa.py`` and
    predicts ``area``, ``delay``, ``static_power``, and ``switching_power``
    simultaneously for a single (RTL, clock) sample.

    Example usage::

        proxy = PPAProxyEvaluator(
            checkpoint_path="/path/to/checkpoint-xxxx",
            encoder_model_name="google/t5gemma-s-s-prefixlm",
        )
        result = proxy.evaluate(rtl_code="module foo ...", clock_period_ns=1.0)
        # result: {"success": True, "area": 1234.5, "delay": 0.48, ...}
    """

    DEFAULT_METRICS: list[str] = [
        "area",
        "delay",
        "static_power",
        "switching_power",
    ]

    def __init__(
        self,
        checkpoint_path: str,
        encoder_model_name: str = "google/t5gemma-s-s-prefixlm",
        metrics: list[str] | None = None,
        max_input_len: int = 2048,
        max_target_len: int = 128,
        device: str | None = None,
    ) -> None:
        """
        Load the trained T5Gemma checkpoint.

        :param checkpoint_path: Path to a HuggingFace-compatible checkpoint
            directory saved by ``T5GemmaModel.save_pretrained``.
        :param encoder_model_name: Legacy compatibility argument. The supplied
            local checkpoint supplies model configuration and tokenizer.
        :param metrics: Ordered list of metrics the model predicts. Defaults to
            ``["area", "delay", "static_power", "switching_power"]``.
        :param max_input_len: Maximum tokeniser length for the input prompt.
        :param max_target_len: Maximum tokeniser length for the decoded output.
        :param device: PyTorch device string (``"cpu"``, ``"cuda"``, etc.).
            Defaults to CUDA when available, otherwise CPU.
        """
        self.metrics: list[str] = metrics if metrics is not None else list(self.DEFAULT_METRICS)
        self.max_input_len: int = max_input_len
        self.max_target_len: int = max_target_len

        # ------------------------------------------------------------------
        # Lazy / deferred import so that torch is not required at module-load
        # time for users who only use VerilogEvaluator / SynthesisEvaluator.
        # ------------------------------------------------------------------
        import sys

        # Add the text2text_regression source tree to sys.path so that
        # `omnipred` and `utils` packages can be found.
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _t2t_dir = os.path.abspath(os.path.join(_this_dir, "..", "text2text_regression"))
        if _t2t_dir not in sys.path:
            sys.path.insert(0, _t2t_dir)

        import torch
        from omnipred import T5GemmaModel, T5GemmaModelConfig
        from transformers import AutoTokenizer

        # Determine device. ``device=None`` keeps the historical auto-detect
        # behavior, while an explicit string lets callers force CPU or GPU.
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use the caller's checkpoint itself, never a discovered child or base-model fallback.
        checkpoint = os.path.abspath(os.path.expanduser(checkpoint_path))
        if not os.path.isdir(checkpoint) or not os.path.isfile(os.path.join(checkpoint, "config.json")):
            raise ValueError("Provide an explicit checkpoint directory containing config.json.")
        config = T5GemmaModelConfig(
            model_name=checkpoint,
            max_input_len=max_input_len,
            max_target_len=max_target_len,
            model_kwargs={"local_files_only": True},
            tokenizer_kwargs={"local_files_only": True},
        )
        self._model = T5GemmaModel(config)
        self._model.to(self.device)
        self._model.eval()

        # The tokeniser is bundled inside T5GemmaModel; expose it directly.
        self._tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, model_max_length=max_input_len, local_files_only=True
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._torch = torch
        print(
            f"PPAProxyEvaluator: loaded checkpoint '{checkpoint_path}' "
            f"on device '{self.device}'. Metrics: {self.metrics}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_prompt(self, rtl_code: str, clock_period_ns: float = 1.0) -> str:
        """Build the inference prompt matching the training format.

        This mirrors ``MultiMetricSeq2SeqDataset.__getitem__`` exactly:
        ``"Predict the {metrics_str} for clock period {clock:.1f}ns.
        The RTL code is as follows:\\n{rtl}\\n"``
        """
        metrics_str = ", ".join(self.metrics)
        return (
            f"Predict the {metrics_str} for clock period {clock_period_ns:.1f}ns. "
            f"The RTL code is as follows:\n{rtl_code}\n"
        )

    def evaluate(
        self,
        rtl_code: str,
        clock_period_ns: float,
    ) -> dict[str, Any]:
        """
        Run model inference on a single (RTL, clock) sample.

        The model decodes a newline-separated list of floating-point values in
        scientific notation, one per metric, in the order defined by
        ``self.metrics``.

        :param rtl_code: RTL source code string (Verilog / SystemVerilog).
        :param clock_period_ns: Target clock period in nanoseconds.
        :return: A dictionary with the following keys:

            - ``success`` (bool) – ``True`` if all metrics were decoded.
            - ``raw_output`` (str) – Raw decoded string from the model.
            - One key per metric in ``self.metrics`` (float or ``None``).
        """
        prompt = self._build_prompt(rtl_code, clock_period_ns)

        # Tokenise
        enc = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_len,
        )
        batch = {k: v.to(self.device) for k, v in enc.items()}

        # Decode (greedy, single sample)
        with self._torch.no_grad():
            output_ids = self._model.decode(batch, num_samples=1)

        # output_ids shape: (batch=1, num_samples=1, seq_len)
        # Squeeze to a flat ID list.
        if output_ids.dim() == 3:
            # (1, 1, L) → (L,)
            ids = output_ids[0, 0].tolist()
        elif output_ids.dim() == 2:
            # (1, L) → (L,)
            ids = output_ids[0].tolist()
        else:
            ids = output_ids.tolist()

        raw_output: str = self._tokenizer.decode(ids, skip_special_tokens=True)

        # Parse: expect N floats separated by "\n", one per metric.
        result: dict[str, Any] = {"raw_output": raw_output}
        try:
            parsed = [
                float(tok.strip())
                for tok in raw_output.strip().split("\n")
                if tok.strip()
            ]
        except ValueError:
            parsed = []

        if len(parsed) == len(self.metrics):
            result["success"] = True
            for name, value in zip(self.metrics, parsed):
                result[name] = value
        else:
            result["success"] = False
            for name in self.metrics:
                result[name] = None
            print(
                f"PPAProxyEvaluator: decode produced {len(parsed)} values, "
                f"expected {len(self.metrics)}. raw_output={raw_output!r}"
            )

        return result


class PPAProxyEvaluatorWrapper:
    """
    Adapter that wraps PPAProxyEvaluator to match the SynthesisEvaluator interface.

    This allows EoHEngine to use the fast T5Gemma proxy seamlessly instead of
    running full Yosys/OpenROAD synthesis.
    """

    def __init__(
        self,
        proxy: PPAProxyEvaluator,
        clk_period: float = 0.01,
        input_preprocess: Literal["none", "yosys"] = "none",
        yosys_bin: str = "yosys",
        yosys_timeout_seconds: int = 120,
        yosys_pdk: str = "nangate45",
        yosys_pdk_path: str | None = None,
        dependency_root: str | None = None,
    ) -> None:
        """
        :param proxy: The underlying PPAProxyEvaluator instance.
        :param clk_period: Target clock period in ns (passed to the proxy model).
        :param input_preprocess: Optional proxy input preprocessing mode.
            ``"none"`` sends raw RTL to the proxy. ``"yosys"`` sends a Yosys
            synthesized/normalized netlist to match checkpoints trained on
            synthesized RTL.
        :param yosys_bin: Yosys executable used for ``input_preprocess="yosys"``.
        :param yosys_timeout_seconds: Timeout for the Yosys preprocessing step.
        :param yosys_pdk: PDK/template target for Yosys preprocessing.
        """
        if input_preprocess not in {"none", "yosys"}:
            raise ValueError(f"Unsupported proxy input_preprocess: {input_preprocess}")
        self.proxy: PPAProxyEvaluator = proxy
        self.clk_period: float = clk_period
        self.input_preprocess: Literal["none", "yosys"] = input_preprocess
        self.yosys_bin: str = yosys_bin
        self.yosys_timeout_seconds: int = yosys_timeout_seconds
        self.yosys_pdk: str = yosys_pdk
        self.yosys_pdk_path = yosys_pdk_path or os.environ.get("REVOLUTION_PDK_ROOT")
        self.dependency_root = dependency_root
        if input_preprocess == "yosys" and not self.yosys_pdk_path:
            raise ValueError("Provide yosys_pdk_path or REVOLUTION_PDK_ROOT for Yosys preprocessing.")

    def evaluate(
        self,
        verilog_file: str,
        problem_name: str,
        synth_top_module_name: str,
        output_directory: str,
        report_base_path: str,
        verilog_evaluator: VerilogEvaluator,
        test_sv_file: str,
        ref_sv_file: str | None,
    ) -> dict[str, Any]:
        """
        Wrapped evaluate method matching SynthesisEvaluator.evaluate signature.
        """
        # 1. Read the RTL code from the file
        try:
            with open(verilog_file, "r", encoding="utf-8") as f:
                rtl_code = f.read()
        except Exception as e:
            return {
                "synthesis_success": False,
                "synthesis_functionality_success": False,
                "ppa_success": False,
                "synthesis_log": f"Proxy Error: Could not read Verilog file {verilog_file}: {e}",
                "ppa_metrics": None,
            }

        os.makedirs(output_directory, exist_ok=True)
        if report_base_path:
            proxy_report_path = f"{report_base_path}.proxy.json"
            proxy_input_path = f"{report_base_path}.proxy_input.v"
        else:
            verilog_base = os.path.splitext(os.path.basename(verilog_file))[0]
            proxy_report_path = os.path.join(
                output_directory, f"{verilog_base}.proxy.json"
            )
            proxy_input_path = os.path.join(
                output_directory, f"{verilog_base}.proxy_input.v"
            )

        proxy_input_code = rtl_code
        preprocessing_metadata: dict[str, Any] = {
            "mode": self.input_preprocess,
            "proxy_input_path": None,
        }
        if self.input_preprocess == "yosys":
            from .yosys_proxy_normalize import normalize_verilog_yosys

            normalize_result = normalize_verilog_yosys(
                rtl_code,
                design_name=problem_name,
                top_module=synth_top_module_name,
                clock=self.clk_period,
                pdk=self.yosys_pdk,
                pdk_path=self.yosys_pdk_path,
                dependency_root=self.dependency_root,
                yosys_bin=self.yosys_bin,
                timeout_seconds=self.yosys_timeout_seconds,
            )
            proxy_input_code = normalize_result.code
            preprocessing_metadata.update(
                {
                    "pdk": normalize_result.pdk,
                    "top_module": normalize_result.top_module,
                    "used_yosys": normalize_result.used_yosys,
                    "fallback_reason": normalize_result.fallback_reason,
                    "dependency_files": normalize_result.dependency_files,
                    "yosys_bin": self.yosys_bin,
                    "yosys_timeout_seconds": self.yosys_timeout_seconds,
                    "proxy_input_path": proxy_input_path,
                }
            )
            try:
                with open(proxy_input_path, "w", encoding="utf-8") as f:
                    f.write(proxy_input_code)
            except Exception as e:
                return {
                    "synthesis_success": False,
                    "synthesis_functionality_success": False,
                    "ppa_success": False,
                    "synthesis_log": (
                        f"Proxy preprocessing succeeded, but failed to write "
                        f"proxy input to {proxy_input_path}: {e}"
                    ),
                    "ppa_metrics": None,
                }

        # 2. Run proxy inference
        # Note: we use self.clk_period which is usually 0.01ns (10ps) in this framework.
        proxy_res = self.proxy.evaluate(proxy_input_code, self.clk_period)

        if not proxy_res["success"]:
            return {
                "synthesis_success": False,
                "synthesis_functionality_success": False,
                "ppa_success": False,
                "synthesis_log": (
                    f"Proxy Inference Failed.\n"
                    f"Preprocessing: {preprocessing_metadata}\n"
                    f"Raw model output:\n{proxy_res.get('raw_output')}"
                ),
                "ppa_metrics": None,
            }

        # 3. Map results to EoHEngine format
        # PPAProxyEvaluator predicts: area, delay, static_power, switching_power
        # EoHEngine expects: area, power, eff_clk_period, tns, wns
        if "total_power" not in proxy_res:
            static_power = proxy_res.get("static_power")
            switching_power = proxy_res.get("switching_power")
            if static_power is not None and switching_power is not None:
                total_power = static_power + switching_power
            else:
                total_power = None
        else:
            total_power = proxy_res.get("total_power")

        proxy_report_dir = os.path.dirname(proxy_report_path)
        if proxy_report_dir:
            os.makedirs(proxy_report_dir, exist_ok=True)
        proxy_report = {
            "problem_name": problem_name,
            "synth_top_module_name": synth_top_module_name,
            "clk_period_ns": self.clk_period,
            "preprocessing": preprocessing_metadata,
            "proxy_metrics": proxy_res,
        }
        try:
            with open(proxy_report_path, "w", encoding="utf-8") as f:
                json.dump(proxy_report, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception as e:
            return {
                "synthesis_success": False,
                "synthesis_functionality_success": False,
                "ppa_success": False,
                "synthesis_log": (
                    f"Proxy Inference Successful, but failed to write report "
                    f"to {proxy_report_path}: {e}"
                ),
                "ppa_metrics": None,
            }

        ppa_metrics = {
            "area": proxy_res.get("area"),
            "power": total_power,
            "eff_clk_period": proxy_res.get("delay"),
            "tns": proxy_res.get("tns", 0.0),
            "wns": proxy_res.get("wns", 0.0),
            "report_path": proxy_report_path,
        }

        return {
            "synthesis_success": True,
            "synthesis_functionality_success": True,  # Assume functional correct if RTL sim passed
            "ppa_success": True,
            "synthesis_log": (
                f"Proxy Inference Successful.\n"
                f"Preprocessing: {preprocessing_metadata}\n"
                f"Raw output: {proxy_res.get('raw_output')}\n"
                f"Saved PPA report: {proxy_report_path}"
            ),
            "ppa_metrics": ppa_metrics,
        }
