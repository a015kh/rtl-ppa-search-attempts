import sys
from pathlib import Path
import tempfile
import unittest

EVOLVE = Path(__file__).resolve().parents[1]
if str(EVOLVE) not in sys.path:
    sys.path.insert(0, str(EVOLVE))

from revolution.evolve import four_pipeline as fp


class FourPipelineTests(unittest.TestCase):
    def test_ppa_score_is_unscaled_negative_product(self):
        point = fp.Point(
            "trained_model",
            5.0,
            status="success",
            area_um2=100.0,
            cycles=20,
            latency_ns=100.0,
            power_mw=2.0,
        )
        fp.score_point(point)
        self.assertEqual(point.score, -20000.0)

    def test_invalid_point_has_no_numeric_penalty(self):
        point = fp.Point(
            "dc_pt", 3.0, status="timing_failed",
            area_um2=100.0, latency_ns=10.0, power_mw=1.0,
        )
        fp.score_point(point)
        self.assertIsNone(point.score)

    def test_odd_period_is_written_as_real_literal(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.sv"
            target = Path(directory) / "target.sv"
            source.write_text("`define CYCLE 10\n#(`CYCLE/2) clk=~clk;\n")
            fp.patch_testbench(source, target, 3.0)
            self.assertIn("`define CYCLE 3.0", target.read_text())

    def test_vcs_cycle_count_drives_latency(self):
        passed, cycles, latency = fp.parse_vcs(
            "------------total time: 138 cycles-----------------\nAll tests PASS!\n",
            3.0,
        )
        self.assertTrue(passed)
        self.assertEqual(cycles, 138)
        self.assertEqual(latency, 414.0)

    def test_pt_power_defaults_to_watts_and_exports_mw(self):
        self.assertAlmostEqual(fp.parse_power_mw("Total Power = 3.5e-04"), 0.35)

    def test_openroad_power_table_exports_mw(self):
        report = """
Group Internal Switching Leakage Total
Total 1.57e-04 1.92e-05 5.82e-05 2.34e-04 100.0%
"""
        self.assertAlmostEqual(fp.parse_power_mw(report), 0.234)


if __name__ == "__main__":
    unittest.main()
