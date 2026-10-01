"""Release resource and external dependency checks without live EDA tools."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from revolution.evolve import four_pipeline as fp
from revolution.evolve import run_verilogeval_four_pipeline as ve
from revolution.evolve import settings
from revolution import dcpt


def test_reference_scripts_are_package_resources():
    root = Path(fp.__file__).resolve().parents[1] / 'resources'
    assert settings.ROOT == root
    for relative in ('scripts/ref/ref.yosys.tcl', 'scripts/ref/ref.openroad.tcl',
                     'scripts/ref/ref.abc.script', 'scripts/util/flow.tcl',
                     'scripts/util/helpers.tcl', 'scripts/util/flow_helpers.tcl'):
        assert (root / relative).is_file()
    # A wheel supplies code/templates, never the actual cell library.
    assert not (root / 'data/pdk').exists()


def test_yosys_render_uses_external_mapping_cells(tmp_path, monkeypatch):
    adders = tmp_path / 'external_adders.v'
    latches = tmp_path / 'external_latches.v'
    monkeypatch.setattr(fp, 'MAP_ADDERS', adders)
    monkeypatch.setattr(fp, 'MAP_LATCHES', latches)
    text = fp.render_yosys(tmp_path / 'synthetic.sv', tmp_path / 'synthetic.syn.v',
                           tmp_path / 'external.lib', 2.0)
    assert str(adders) in text
    assert str(latches) in text
    assert str(settings.ROOT / 'scripts/ref/ref.abc.script') in text
    assert 'data/pdk/cells_' not in text


@pytest.fixture
def external_dependencies(tmp_path, monkeypatch):
    def file(name):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('synthetic fixture')
        return target
    monkeypatch.setattr(fp, 'DC', file('tools/dc'))
    monkeypatch.setattr(fp, 'PT', file('tools/pt'))
    monkeypatch.setattr(fp, 'YOSYS', file('tools/yosys'))
    monkeypatch.setattr(fp, 'OPENROAD', file('tools/openroad'))
    monkeypatch.setattr(fp, 'CELL_MODELS', file('cell-models.v'))
    monkeypatch.setattr(fp, 'MAP_ADDERS', tmp_path / 'external-adders.v')
    monkeypatch.setattr(fp, 'MAP_LATCHES', tmp_path / 'external-latches.v')
    monkeypatch.setattr(fp, 'REV_PDK', tmp_path / 'external-pdk')
    pdk = {key: tmp_path / f'pdk-{key}' for key in ('lib', 'db', 'tech_lef', 'cell_lef')}
    for key in ('lib', 'db'):
        file('pdk-' + key)
    monkeypatch.setattr(fp, 'pdk_files', lambda: pdk)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    args = SimpleNamespace(pipelines=['trained_model'], raw_checkpoint=checkpoint,
                           yosys_checkpoint=tmp_path / 'unused-checkpoint')
    return args, file, pdk


def test_raw_pipeline_does_not_require_yosys_assets(external_dependencies):
    args, _file, _pdk = external_dependencies
    fp.validate_files(args)


def test_all_pipelines_require_golden_cell_model(external_dependencies):
    args, _file, _pdk = external_dependencies
    fp.CELL_MODELS.unlink()
    with pytest.raises(fp.ExperimentError, match='cell-models.v'):
        fp.validate_files(args)


def test_yosys_pipeline_checks_mapping_files(external_dependencies):
    args, file, _pdk = external_dependencies
    args.pipelines = ['yosys_trained_model']
    args.yosys_checkpoint = args.raw_checkpoint
    file('external-pdk/Nangate45_typ.lib')
    with pytest.raises(fp.ExperimentError, match='external-adders.v'):
        fp.validate_files(args)
    file('external-adders.v')
    file('external-latches.v')
    fp.validate_files(args)


def test_openroad_checks_lef_files(external_dependencies):
    args, file, _pdk = external_dependencies
    args.pipelines = ['yosys_openroad']
    file('external-pdk/Nangate45_typ.lib')
    file('external-adders.v')
    file('external-latches.v')
    with pytest.raises(fp.ExperimentError, match='pdk-tech_lef'):
        fp.validate_files(args)
    file('pdk-tech_lef')
    file('pdk-cell_lef')
    file('external-pdk/Nangate45.tracks')
    fp.validate_files(args)


def test_openroad_checks_external_tracks(external_dependencies):
    args, file, _pdk = external_dependencies
    args.pipelines = ['yosys_openroad']
    for name in ('external-pdk/Nangate45_typ.lib', 'external-adders.v',
                 'external-latches.v', 'pdk-tech_lef', 'pdk-cell_lef'):
        file(name)
    with pytest.raises(fp.ExperimentError, match='Nangate45.tracks'):
        fp.validate_files(args)


def test_verilogeval_requires_explicit_output():
    with pytest.raises(SystemExit):
        ve.build_parser().parse_args([])


def test_verilogeval_passes_external_tool_configuration(tmp_path, monkeypatch):
    config = tmp_path / 'tool-config.json'
    config.write_text('{}')
    flow_root = tmp_path / 'external-flow'
    flow_root.mkdir()
    paths = {'synthesis_flow': flow_root, 'synthesis_config': config}
    monkeypatch.setattr(settings, 'path', lambda name: paths[name])
    calls = {}
    flow = object()

    def load(root):
        calls['flow_root'] = root
        return flow

    def runner(source, cache, config_path):
        calls.update(flow=source, cache=cache, config=config_path)
        return object()

    monkeypatch.setattr(dcpt.DCPTFlow, 'load', load)
    monkeypatch.setattr(dcpt, 'DCPTJobRunner', runner)
    evaluator = ve.VerilogEvalPipelineEvaluator.__new__(ve.VerilogEvalPipelineEvaluator)
    evaluator._dcpt_flow = evaluator._dcpt_runner = None
    evaluator.root = tmp_path / 'run/pipeline'
    result = evaluator._ensure_dcpt()
    assert result[0] is flow
    assert calls['flow_root'] == flow_root
    assert calls['config'] == config
    assert calls['cache'] == tmp_path / 'run/dcpt_cache'


def test_verilogeval_missing_tool_configuration_fails_before_default(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'path', lambda name: tmp_path / name)
    evaluator = ve.VerilogEvalPipelineEvaluator.__new__(ve.VerilogEvalPipelineEvaluator)
    evaluator._dcpt_flow = evaluator._dcpt_runner = None
    with pytest.raises(fp.ExperimentError, match='synthesis_config'):
        evaluator._ensure_dcpt()
