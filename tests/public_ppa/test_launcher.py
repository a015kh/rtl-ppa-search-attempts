"""Synthetic public-interface checks; these fixtures are not research data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('public_ppa_launcher', ROOT / 'scripts/ppa.py')
ppa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ppa)


@pytest.fixture
def configured(tmp_path):
    config = json.loads((ROOT / 'configs/ppa.example.json').read_text(encoding='utf-8'))
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    # Entirely fabricated tiny example and PPA labels.
    row = {'rtl': 'module synthetic(input a, output y); assign y = a; endmodule',
           'clock': 2, 'area_value': 1.5, 'delay_value': 0.5, 'total_power_value': 0.25}
    for split in ('train', 'valid', 'heldout'):
        (dataset / (split + '.jsonl')).write_text(json.dumps(row) + '\n', encoding='utf-8')
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({'model_type': 't5gemma', 'is_encoder_decoder': True}))
    (model / 'tokenizer_config.json').write_text('{}')
    (model / 'tokenizer.json').write_text('{}')
    # Intentionally non-weight bytes: structural dry-run must not load this file.
    (model / 'model.safetensors').write_bytes(b'synthetic-not-real-weights')
    args = SimpleNamespace(operation='evaluate', repo=ROOT, dataset=dataset, model=str(model),
                           output=tmp_path / 'new-output', split='heldout', device='cpu', limit=3)
    return args, config


def test_no_private_manifest_required(configured):
    args, config = configured
    assert ppa.validate(args, config) == ROOT / 'src/text2text_regression'
    assert not (args.dataset / 'dataset_manifest.json').exists()


@pytest.mark.parametrize('operation', ['evaluate', 'predict'])
def test_split_is_explicit(configured, operation):
    args, config = configured
    args.operation = operation
    args.split = None
    with pytest.raises(ValueError, match='--split is required'):
        ppa.validate(args, config)


def test_training_uses_only_configured_splits(configured):
    args, config = configured
    args.operation, args.split = 'train', None
    assert ppa.validate(args, config)
    cmd = ppa.command(args, config, ROOT / 'src/text2text_regression')
    assert '--do_train' in cmd
    assert cmd[cmd.index('--encoder_model_name') + 1] == str(Path(args.model).resolve())
    (args.dataset / 'valid.jsonl').unlink()
    with pytest.raises(ValueError, match='Missing dataset split'):
        ppa.validate(args, config)


def test_evaluation_dispatch_is_standalone(configured):
    args, config = configured
    cmd = ppa.command(args, config, ppa.validate(args, config))
    assert Path(cmd[1]).name == 'evaluate_ppa.py'
    assert '--do_train' not in cmd
    assert '--train_splits' not in cmd
    assert cmd[cmd.index('--num-samples') + 1] == '16'
    assert cmd[cmd.index('--split') + 1] == 'heldout'


def test_optional_checksums_succeed(configured):
    args, config = configured
    config['checksums'] = {'splits': {'heldout': ppa.digest(args.dataset / 'heldout.jsonl')},
                           'model': {'model.safetensors': ppa.digest(Path(args.model) / 'model.safetensors')}}
    assert ppa.validate(args, config)


@pytest.mark.parametrize('group,name', [('splits', 'heldout'), ('model', 'model.safetensors')])
def test_supplied_checksum_is_enforced(configured, group, name):
    args, config = configured
    config['checksums'] = {group: {name: '0' * 64}}
    with pytest.raises(ValueError, match='checksum mismatch'):
        ppa.validate(args, config)


def test_unused_split_checksum_does_not_load_training_data(configured):
    args, config = configured
    (args.dataset / 'train.jsonl').unlink()
    config['checksums'] = {'splits': {'train': '0' * 64}}
    assert ppa.validate(args, config)


@pytest.mark.parametrize('field', ['dataset', 'model', 'repo'])
def test_output_protects_inputs(configured, field):
    args, config = configured
    args.output = Path(getattr(args, field)) / 'nested-output'
    with pytest.raises(ValueError, match='Output must be outside'):
        ppa.validate(args, config)


def test_existing_output_is_rejected(configured):
    args, config = configured
    args.output.mkdir()
    with pytest.raises(ValueError, match='Output already exists'):
        ppa.validate(args, config)


@pytest.mark.parametrize('split', ['../train', '/tmp/test', '.hidden', 'train\\test', '', 'train.extra'])
def test_invalid_split(configured, split):
    args, config = configured
    args.split = split
    with pytest.raises(ValueError):
        ppa.validate(args, config)


@pytest.mark.parametrize('field,value', [('rtl', ''), ('clock', None), ('area_value', 'bad'),
                                       ('delay_value', True), ('total_power_value', float('inf'))])
def test_formatter_fields_are_validated(configured, field, value):
    args, config = configured
    path = args.dataset / 'heldout.jsonl'
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match=field):
        ppa.validate(args, config)


def test_simple_metric_aliases_match_formatter(configured):
    args, config = configured
    path = args.dataset / 'heldout.jsonl'
    row = json.loads(path.read_text())
    for metric in config['metrics']:
        row[metric] = row.pop(metric + '_value')
    path.write_text(json.dumps(row) + '\n')
    assert ppa.validate(args, config)


def test_metric_alias_precedence_matches_formatter(configured):
    args, config = configured
    path = args.dataset / 'heldout.jsonl'
    row = json.loads(path.read_text())
    row['area'], row['area_value'] = 3, 'invalid'
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='area_value'):
        ppa.validate(args, config)


def test_missing_metric_is_actionable(configured):
    args, config = configured
    path = args.dataset / 'heldout.jsonl'
    row = json.loads(path.read_text())
    del row['delay_value']
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='missing metric delay'):
        ppa.validate(args, config)


def test_empty_split_is_rejected(configured):
    args, config = configured
    (args.dataset / 'heldout.jsonl').write_text('\n')
    with pytest.raises(ValueError, match='empty'):
        ppa.validate(args, config)


def test_sharded_weights_layout(configured):
    args, config = configured
    model = Path(args.model)
    (model / 'model.safetensors').unlink()
    (model / 'model-00001-of-00001.safetensors').write_bytes(b'synthetic')
    (model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'model-00001-of-00001.safetensors'}}))
    assert ppa.validate(args, config)
    (model / 'model-00001-of-00001.safetensors').unlink()
    with pytest.raises(ValueError, match='weight shard'):
        ppa.validate(args, config)


def test_missing_tokenizer_is_actionable(configured):
    args, config = configured
    (Path(args.model) / 'tokenizer.json').unlink()
    with pytest.raises(ValueError, match='tokenizer vocabulary'):
        ppa.validate(args, config)


def test_causal_model_is_rejected(configured):
    args, config = configured
    (Path(args.model) / 'config.json').write_text(json.dumps({'model_type': 'gpt2'}))
    with pytest.raises(ValueError, match='encoder-decoder'):
        ppa.validate(args, config)


def test_bad_optional_checksum_shape(configured):
    args, config = configured
    config['checksums'] = {'model': {'../private': '0' * 64}}
    with pytest.raises(ValueError, match='plain filenames'):
        ppa.validate(args, config)


@pytest.mark.parametrize('change', [{'num_samples': 0}, {'metrics': ['area', 'area']},
                                  {'training': {'eval_batch_size': 2.5}}, {'training': {'other': 2, 'eval_batch_size': 4}}])
def test_invalid_configuration(configured, change):
    args, config = configured
    config.update(change)
    with pytest.raises(ValueError):
        ppa.validate(args, config)


def test_dry_run_from_another_directory_without_loading_weights(configured, tmp_path):
    args, config = configured
    path = tmp_path / 'experiment.json'
    path.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/ppa.py'), 'evaluate',
                             '--config', str(path), '--dataset', str(args.dataset),
                             '--model', args.model, '--output', str(args.output),
                             '--split', args.split, '--dry-run'], cwd=tmp_path,
                            text=True, capture_output=True, check=True)
    dispatch = json.loads(result.stdout)
    assert Path(dispatch['command'][1]).name == 'evaluate_ppa.py'
    assert not args.output.exists()


def test_cli_config_is_required():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/ppa.py'), 'train',
                             '--dataset', '/path/to/dataset', '--model', '/path/to/checkpoint',
                             '--output', '/path/to/output'], text=True, capture_output=True)
    assert result.returncode == 2
    assert '--config' in result.stderr
