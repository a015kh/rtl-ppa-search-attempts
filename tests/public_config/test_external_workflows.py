"""Synthetic configuration checks; no commercial tools or research inputs."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from revolution.cvdp import CVDPTask
from revolution.dcpt import DCPTError
from revolution.evaluation import SynthesisEvaluator
from revolution import yosys_proxy_normalize as normalize
from revolution.evolve import cvdp, evolve, settings


ROOT = Path(__file__).resolve().parents[2]


class ExternalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def synthetic_evaluator(self):
        native = CVDPTask(
            id='synthetic', prompt='Synthetic test circuit', context={},
            seed={'synthetic.sv': 'module synthetic; endmodule'},
            targets=('synthetic.sv',), harness={}, heavy=False,
            repository=None, catalog={},
        )
        task = evolve.Task('synthetic', self.directory / 'SPEC.md',
                           self.directory / 'baseline.sv', self.directory / 'testbench',
                           '', top='synthetic', native=native)
        candidate = self.directory / 'candidate.sv'
        candidate.write_text('module synthetic; endmodule', encoding='utf-8')
        evaluator = cvdp.CVDPCommercialEvaluator(task, {}, self.directory / 'cache')
        return evaluator, candidate

    def test_cvdp_forwards_explicit_flow_config_and_library(self):
        evaluator, candidate = self.synthetic_evaluator()
        locations = {name: self.directory / name
                     for name in ('synthesis_flow', 'synthesis_config', 'cell_verilog')}
        calls = {}
        native_result = {'rtl_passed': True, 'clock_results': [
            {'clock_ns': 2, 'success': True, 'gate_passed': True,
             'constraint_met': True, 'metrics': {'area': 3, 'wns': 0.5, 'power': 0.25}}
        ]}

        def factory(cache, **kwargs):
            calls.update(cache=cache, **kwargs)
            return SimpleNamespace(evaluate=lambda *args, **values:
                                   SimpleNamespace(to_dict=lambda: native_result))

        evaluator.flow = factory
        with patch.object(settings, 'path', side_effect=lambda name: locations[name]):
            result = evaluator.evaluate(candidate, 2.0, True)
        self.assertEqual(calls['flow_root'], locations['synthesis_flow'])
        self.assertEqual(calls['config_path'], locations['synthesis_config'])
        self.assertEqual(calls['cell_library'], locations['cell_verilog'])
        self.assertEqual(calls['clocks'], [2.0])
        self.assertTrue(result.rtl_pass)
        self.assertTrue(result.gate_pass)
        self.assertTrue(result.timing_clean)

    def test_cvdp_records_dcpt_error_as_failed_evaluation(self):
        evaluator, candidate = self.synthetic_evaluator()
        evaluator.flow = lambda *args, **kwargs: (_ for _ in ()).throw(DCPTError('synthetic configuration error'))
        result = evaluator.evaluate(candidate, 2.0, False)
        self.assertEqual(result.failure_stage, 'evaluation')
        self.assertIn('synthetic configuration error', result.feedback)
        self.assertFalse(result.rtl_pass)
        record = evaluator._work_dir(candidate, 2.0, 'proxy') / 'result.json'
        self.assertEqual(json.loads(record.read_text())['failure_stage'], 'evaluation')

    def test_shared_evolve_requires_output(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evolve.build_parser().parse_args([])

    def test_cvdp_adapter_requires_output(self):
        tools = self.directory / 'tools'
        (tools / 'bin').mkdir(parents=True)
        (tools / 'bin/iverilog').write_text('synthetic')
        with patch.dict(os.environ), self.assertRaisesRegex(evolve.ExperimentError, 'explicit --run-dir'):
            cvdp.main(['--cvdp-task', 'synthetic', '--cvdp-tools-root', str(tools)])
        self.assertFalse((self.directory / 'runs').exists())

    def test_population_cvdp_requires_output(self):
        path = ROOT / 'src/run_cvdp_evolution.py'
        spec = importlib.util.spec_from_file_location('release_cvdp_runner', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'release_cvdp_runner': module}):
            spec.loader.exec_module(module)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                module.build_parser().parse_args(['prepare', '--data-root', str(self.directory),
                                                 '--benchmark-dir', str(self.directory / 'benchmark'),
                                                 '--heavy-root', str(self.directory / 'heavy')])

    def dispatcher(self, *extra):
        config = self.directory / 'configuration.json'
        config.write_text('{}', encoding='utf-8')
        return subprocess.run([sys.executable, str(ROOT / 'scripts/run_evolve.py'), 'cvdp',
                               '--config', str(config), '--dry-run', '--', '--cvdp-task', 'synthetic',
                               '--run-dir', str(self.directory / 'output'), *extra],
                              cwd=self.directory, text=True, capture_output=True)

    def test_dispatcher_recognizes_cvdp_benchmark_override(self):
        result = self.dispatcher('--cvdp-benchmark', str(self.directory / 'external-benchmark'))
        self.assertEqual(result.returncode, 0, result.stderr)
        command = json.loads(result.stdout)['command']
        self.assertIn('--cvdp-benchmark', command)

    def test_dispatcher_does_not_accept_unrelated_benchmark_flag(self):
        result = self.dispatcher('--benchmark-dir', str(self.directory / 'wrong-flag'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('paths.cvdp_benchmark', result.stderr)

    def test_synthesis_uses_caller_pdk_and_packaged_scripts(self):
        supplied = self.directory / 'external-pdk'
        supplied.mkdir()
        unused = self.directory / 'environment-pdk'
        unused.mkdir()
        with patch.dict(os.environ, {'REVOLUTION_PDK_ROOT': str(unused)}):
            evaluator = SynthesisEvaluator(pdk_path=str(supplied))
        self.assertEqual(Path(evaluator.pdk_path), supplied)
        expected = ROOT / 'src/revolution/resources/scripts/ref'
        self.assertEqual(Path(evaluator.ref_dir_path), expected)
        self.assertTrue((expected / 'ref.yosys.tcl').is_file())

    def test_synthesis_missing_pdk_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'PDK directory does not exist'):
            SynthesisEvaluator(pdk_path=str(self.directory / 'missing'))

    def test_synthesis_requires_external_pdk(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'cell libraries are not bundled'):
            SynthesisEvaluator()

    def test_normalizer_uses_packaged_resources_without_bundled_pdk(self):
        resource_root = Path(normalize._project_root())
        self.assertEqual(resource_root, ROOT / 'src/revolution/resources')
        self.assertTrue((resource_root / 'scripts/ref/ref.yosys.tcl').is_file())
        self.assertFalse((resource_root / 'data/pdk').exists())

    def test_normalizer_missing_pdk_fails_before_tool_execution(self):
        with patch.object(normalize.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'PDK directory does not exist'):
                normalize.normalize_verilog_yosys('module synthetic; endmodule', 'synthetic',
                                                  pdk_path=str(self.directory / 'missing'))
        run.assert_not_called()

    def test_runtime_yosys_failure_records_historical_fallback(self):
        supplied = self.directory / 'external-pdk'
        supplied.mkdir()
        rtl = 'module synthetic; endmodule'
        failed = SimpleNamespace(returncode=1, stderr='synthetic Yosys failure')
        with patch.object(normalize.subprocess, 'run', return_value=failed), \
             patch.object(normalize, '_fallback_normalize_verilog', return_value=rtl):
            result = normalize.normalize_verilog_yosys(rtl, 'synthetic', pdk_path=str(supplied))
        self.assertFalse(result.used_yosys)
        self.assertIn('synthetic Yosys failure', result.fallback_reason)
        self.assertEqual(result.code, rtl)


if __name__ == '__main__':
    unittest.main()
