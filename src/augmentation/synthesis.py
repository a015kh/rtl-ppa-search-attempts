from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from revolution.evaluation import SynthesisEvaluator

from .dataset_io import sanitize_id
from .models import PpaMetrics


@dataclass
class SynthesisResult:
    success: bool
    metrics: PpaMetrics
    report_path: str | None
    ppa_path: str | None
    netlist_path: str | None
    yosys_log_path: str | None
    openroad_log_path: str | None
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "metrics": self.metrics.to_dict(),
            "report_path": self.report_path,
            "ppa_path": self.ppa_path,
            "netlist_path": self.netlist_path,
            "yosys_log_path": self.yosys_log_path,
            "openroad_log_path": self.openroad_log_path,
            "error": self.error,
        }


class PpaSynthesizer:
    """Run the REvolution Yosys/OpenROAD PPA flow for standalone dataset rows."""

    def __init__(
        self,
        *,
        work_root: str | Path,
        yosys_bin: str = "yosys",
        openroad_bin: str = "openroad",
        clock_period_ns: float = 0.01,
        timeout_seconds: int = 300,
    ) -> None:
        self.work_root = Path(work_root)
        self.yosys_bin = yosys_bin
        self.openroad_bin = openroad_bin
        self.timeout_seconds = timeout_seconds
        self.evaluator = SynthesisEvaluator(
            yosys_path=yosys_bin, openroad_path=openroad_bin
        )
        self.evaluator.clk_period = clock_period_ns

    def synthesize_text(
        self,
        *,
        verilog: str,
        sample_id: str,
        top_module: str,
        variant: str,
    ) -> SynthesisResult:
        sample_dir = self.work_root / sanitize_id(sample_id) / sanitize_id(variant)
        sample_dir.mkdir(parents=True, exist_ok=True)
        verilog_path = sample_dir / f"{sanitize_id(sample_id)}_{sanitize_id(variant)}.sv"
        verilog_path.write_text(verilog, encoding="utf-8")
        return self.synthesize_file(
            verilog_file=verilog_path,
            sample_id=sample_id,
            top_module=top_module,
            variant=variant,
        )

    def synthesize_file(
        self,
        *,
        verilog_file: str | Path,
        sample_id: str,
        top_module: str,
        variant: str,
    ) -> SynthesisResult:
        verilog_path = Path(verilog_file)
        output_dir = verilog_path.parent
        netlist_path = output_dir / f"{sanitize_id(sample_id)}_{sanitize_id(variant)}.syn.v"
        report_path = output_dir / f"{sanitize_id(sample_id)}_{sanitize_id(variant)}_synthesis_report.rpt"
        yosys_log_path = output_dir / "yosys.log"
        openroad_log_path = output_dir / "openroad.log"

        try:
            sdc_path = self.evaluator._create_sdc_file(
                str(verilog_path),
                top_module,
                str(output_dir),
                self.evaluator.clk_period,
            )
            yosys_script = self.evaluator._create_yosys_script(
                str(verilog_path),
                top_module,
                str(output_dir),
                self.evaluator.clk_period,
                str(netlist_path),
            )
            openroad_script = self.evaluator._create_openroad_script(
                sdc_path,
                top_module,
                str(output_dir),
                str(netlist_path),
            )

            yosys_proc = subprocess.run(
                [self.yosys_bin, yosys_script],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            yosys_log_path.write_text(
                (yosys_proc.stdout or "") + "\n" + (yosys_proc.stderr or ""),
                encoding="utf-8",
            )
            if yosys_proc.returncode != 0:
                return self._failure(
                    report_path,
                    yosys_log_path,
                    openroad_log_path,
                    netlist_path,
                    f"Yosys failed with return code {yosys_proc.returncode}.",
                )

            openroad_proc = subprocess.run(
                [self.openroad_bin, openroad_script],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            openroad_log = (openroad_proc.stdout or "") + "\n" + (
                openroad_proc.stderr or ""
            )
            openroad_log_path.write_text(openroad_log, encoding="utf-8")
            report_path.write_text(openroad_log, encoding="utf-8")
            if openroad_proc.returncode != 0:
                return self._failure(
                    report_path,
                    yosys_log_path,
                    openroad_log_path,
                    netlist_path,
                    f"OpenROAD failed with return code {openroad_proc.returncode}.",
                )

            metrics = PpaMetrics.from_mapping(
                self.evaluator._parse_ppa_log(str(report_path))
            )
            return SynthesisResult(
                success=True,
                metrics=metrics,
                report_path=str(report_path),
                ppa_path=metrics.report_path,
                netlist_path=str(netlist_path),
                yosys_log_path=str(yosys_log_path),
                openroad_log_path=str(openroad_log_path),
            )
        except subprocess.TimeoutExpired as exc:
            return self._failure(
                report_path,
                yosys_log_path,
                openroad_log_path,
                netlist_path,
                f"Synthesis timed out after {exc.timeout} seconds.",
            )
        except Exception as exc:
            return self._failure(
                report_path,
                yosys_log_path,
                openroad_log_path,
                netlist_path,
                f"Synthesis raised {type(exc).__name__}: {exc}",
            )

    def _failure(
        self,
        report_path: Path,
        yosys_log_path: Path,
        openroad_log_path: Path,
        netlist_path: Path,
        error: str,
    ) -> SynthesisResult:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("a", encoding="utf-8") as f:
            f.write(f"\n\n--- SYNTHESIS FAILED ---\n{error}\n")
        return SynthesisResult(
            success=False,
            metrics=PpaMetrics(report_path=str(report_path)),
            report_path=str(report_path),
            ppa_path=None,
            netlist_path=str(netlist_path) if netlist_path.exists() else None,
            yosys_log_path=str(yosys_log_path) if yosys_log_path.exists() else None,
            openroad_log_path=str(openroad_log_path)
            if openroad_log_path.exists()
            else None,
            error=error,
        )
