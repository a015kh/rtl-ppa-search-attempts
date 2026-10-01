import contextlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class YosysNormalizeResult:
    code: str
    used_yosys: bool
    fallback_reason: str | None
    top_module: str
    pdk: str
    dependency_files: list[str]


REALBENCH_DIR = ""  # Optional dependency tree is explicitly supplied by the caller.
_MODULE_TO_FILE: dict[str, str] | None = None
_INCLUDE_DIRS: set[str] | None = None


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "resources"))


def _strip_comments(code: str) -> str:
    code = re.sub(r"//.*", "", code)
    return re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)


def _extract_first_module_name(code: str) -> str | None:
    module_pattern = re.compile(r"^\s*module\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.M)
    match = module_pattern.search(code)
    return match.group(1) if match else None


def get_instantiated_modules(code: str) -> set[str]:
    clean_code = _strip_comments(code)
    stripped_code = []
    i = 0
    n = len(clean_code)
    while i < n:
        if clean_code[i] == "#" and i + 1 < n:
            j = i + 1
            while j < n and clean_code[j].isspace():
                j += 1
            if j < n and clean_code[j] == "(":
                paren_depth = 1
                k = j + 1
                while k < n and paren_depth > 0:
                    if clean_code[k] == "(":
                        paren_depth += 1
                    elif clean_code[k] == ")":
                        paren_depth -= 1
                    k += 1
                stripped_code.append(" ")
                i = k
                continue
        stripped_code.append(clean_code[i])
        i += 1

    clean_code = "".join(stripped_code)
    pattern = re.compile(
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+"
        r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    )
    keywords = {
        "always",
        "assign",
        "begin",
        "case",
        "else",
        "end",
        "endcase",
        "generate",
        "if",
        "initial",
        "inout",
        "input",
        "localparam",
        "module",
        "output",
        "parameter",
        "reg",
        "wire",
    }
    return {
        match.group(1)
        for match in pattern.finditer(clean_code)
        if match.group(1) not in keywords
    }


def _build_realbench_index() -> tuple[dict[str, str], set[str]]:
    module_to_file: dict[str, str] = {}
    include_dirs: set[str] = set()
    if not os.path.exists(REALBENCH_DIR):
        return module_to_file, include_dirs

    include_dirs.update(
        {
            os.path.join(REALBENCH_DIR, "sdc/sd_defines"),
            os.path.join(REALBENCH_DIR, "e203_hbirdv2/e203_defines"),
            os.path.join(REALBENCH_DIR, "e203_hbirdv2/config"),
            os.path.join(REALBENCH_DIR, "system/e203_cpu_top"),
        }
    )
    for root, _dirs, files in os.walk(REALBENCH_DIR):
        if "verification" in root.lower():
            continue
        if any(
            f.endswith(".vh") or (f.endswith(".v") and ("defines" in f or "config" in f))
            for f in files
        ):
            include_dirs.add(root)
        for filename in files:
            if not (filename.endswith(".v") or filename.endswith(".sv")):
                continue
            lower_filename = filename.lower()
            if (
                "tb_" in lower_filename
                or "_tb" in lower_filename
                or "testbench" in lower_filename
                or "stimulus" in lower_filename
                or "ref.sv" in lower_filename
                or "template" in lower_filename
            ):
                continue
            if filename in {
                "e203_cpu_top_top.sv",
                "sdc_controller_top.sv",
                "aes_cipher_top_top.sv",
                "aes_inv_cipher_top_top.sv",
            }:
                continue

            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except OSError:
                continue

            for module_name in re.findall(
                r"\bmodule\s+([a-zA-Z_][a-zA-Z0-9_]*)", content
            ):
                if module_name not in module_to_file:
                    module_to_file[module_name] = file_path
                elif "system/" in module_to_file[module_name] and "system/" not in file_path:
                    module_to_file[module_name] = file_path

    return module_to_file, include_dirs


def _realbench_index() -> tuple[dict[str, str], set[str]]:
    global _MODULE_TO_FILE, _INCLUDE_DIRS
    if _MODULE_TO_FILE is None or _INCLUDE_DIRS is None:
        _MODULE_TO_FILE, _INCLUDE_DIRS = _build_realbench_index()
    return _MODULE_TO_FILE, _INCLUDE_DIRS


def _resolve_dependency_files(code: str, top_module: str) -> list[str]:
    module_to_file, _include_dirs = _realbench_index()
    needed_deps = {
        module_to_file[module_name]
        for module_name in get_instantiated_modules(code)
        if module_name in module_to_file and module_name != top_module
    }

    visited = set(needed_deps)
    queue = list(needed_deps)
    while queue:
        current_file = queue.pop(0)
        try:
            with open(current_file, "r", encoding="utf-8", errors="ignore") as f:
                current_code = f.read()
        except OSError:
            continue
        for module_name in get_instantiated_modules(current_code):
            if module_name in module_to_file and module_name != top_module:
                dep_path = module_to_file[module_name]
                if dep_path not in visited:
                    visited.add(dep_path)
                    queue.append(dep_path)

    return sorted(visited)


def _fallback_normalize_verilog(code: str) -> str:
    try:
        from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
        from pyverilog.vparser.parser import VerilogParser
        from pyverilog.vparser.parser import parse
    except Exception:
        return code

    if "`" in code:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_file = os.path.join(tmpdir, "tmp.v")
                with open(tmp_file, "w", encoding="utf-8") as f:
                    f.write(code)
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    with contextlib.redirect_stderr(devnull):
                        ast, _directives = parse(filelist=[tmp_file], outputdir=tmpdir)
                return ASTCodeGenerator().visit(ast)
        except Exception:
            return code

    try:
        parser = VerilogParser()
        ast = parser.parse(code)
        return ASTCodeGenerator().visit(ast)
    except Exception:
        return code


def _template_name_for_pdk(pdk: str) -> str:
    normalized = pdk.lower()
    if normalized in {"nangate", "nangate45"}:
        return "ref.yosys.tcl"
    if normalized in {"sky130", "sky130hd"}:
        return "ref.yosys.sky130.tcl"
    raise ValueError(f"Unsupported proxy Yosys PDK: {pdk}")


def normalize_verilog_yosys(
    code: str,
    design_name: str,
    top_module: str | None = None,
    clock: float = 1.0,
    pdk: str = "nangate45",
    yosys_bin: str = "yosys",
    timeout_seconds: int = 120,
    pdk_path: str | None = None,
    dependency_root: str | None = None,
) -> YosysNormalizeResult:
    selected_top_module = top_module or _extract_first_module_name(code) or design_name or "top_module"
    global REALBENCH_DIR, _MODULE_TO_FILE, _INCLUDE_DIRS
    selected_dependency_root = dependency_root or ""
    if REALBENCH_DIR != selected_dependency_root:
        REALBENCH_DIR = selected_dependency_root
        _MODULE_TO_FILE = _INCLUDE_DIRS = None
    dependency_files = _resolve_dependency_files(code, selected_top_module)
    project_root = _project_root()
    ref_dir = os.path.join(project_root, "scripts", "ref")
    selected_pdk = pdk_path or os.environ.get("REVOLUTION_PDK_ROOT")
    if not selected_pdk:
        raise ValueError("Provide pdk_path or REVOLUTION_PDK_ROOT for Yosys preprocessing.")
    pdk_dir = os.path.abspath(os.path.expanduser(selected_pdk))
    if not os.path.isdir(pdk_dir):
        raise ValueError("The supplied PDK directory does not exist: " + pdk_dir)
    if dependency_root and not os.path.isdir(REALBENCH_DIR):
        raise ValueError("The supplied dependency directory does not exist: " + REALBENCH_DIR)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            verilog_file = os.path.join(tmpdir, "design.sv")
            output_file = os.path.join(tmpdir, "design.syn.v")
            yosys_script_path = os.path.join(tmpdir, "synth.tcl")
            with open(verilog_file, "w", encoding="utf-8") as f:
                f.write(code)

            template_path = os.path.join(ref_dir, _template_name_for_pdk(pdk))
            with open(template_path, "r", encoding="utf-8") as f:
                script_text = f.read()

            _module_to_file, include_dirs = _realbench_index()
            include_str = " ".join(f"-I {path}" for path in sorted(include_dirs))
            read_commands = [
                f"read_verilog -defer {include_str} -sv {dep}"
                for dep in dependency_files
            ]
            read_commands.append(f"read_verilog -defer {include_str} -sv $VERILOG_FILE")
            script_text = script_text.replace(
                "read_verilog -defer -sv $VERILOG_FILE", "\n".join(read_commands)
            )
            script_text = script_text.replace("__VERILOG_FILE__", os.path.abspath(verilog_file))
            script_text = script_text.replace("__MODULE_NAME__", selected_top_module)
            script_text = script_text.replace("__OUTPUT_DIR__", os.path.abspath(tmpdir))
            script_text = script_text.replace("__OUTPUT_FILE__", os.path.abspath(output_file))
            script_text = script_text.replace("__REF_DIR__", os.path.abspath(ref_dir))
            script_text = script_text.replace("__PDK_DIR__", os.path.abspath(pdk_dir))
            script_text = script_text.replace("__CLK_PERIOD__", str(float(clock) * 1000))

            with open(yosys_script_path, "w", encoding="utf-8") as f:
                f.write(script_text)

            process = subprocess.run(
                [yosys_bin, yosys_script_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"Yosys returned {process.returncode}: {process.stderr.strip()}"
                )
            with open(output_file, "r", encoding="utf-8") as f:
                return YosysNormalizeResult(
                    code=f.read(),
                    used_yosys=True,
                    fallback_reason=None,
                    top_module=selected_top_module,
                    pdk=pdk,
                    dependency_files=dependency_files,
                )
    except Exception as exc:
        return YosysNormalizeResult(
            code=_fallback_normalize_verilog(code),
            used_yosys=False,
            fallback_reason=str(exc),
            top_module=selected_top_module,
            pdk=pdk,
            dependency_files=dependency_files,
        )
