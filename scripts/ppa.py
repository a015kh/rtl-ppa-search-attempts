"""Train, evaluate or predict PPA using explicit local artifacts and configuration."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = {
    name: [name + '_value', name]
    for name in ('area', 'delay', 'static_power', 'switching_power', 'total_power', 'tns', 'wns')
}
TRAINING_OPTIONS = {
    'num_train_epochs', 'train_batch_size', 'eval_batch_size',
    'gradient_accumulation_steps', 'learning_rate', 'weight_decay',
    'eval_steps', 'save_steps', 'dataloader_num_workers',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def split_name(value):
    """Splits name JSONL files, never paths or dataset identifiers."""
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError('Use a plain split name containing letters, numbers, _ or - (without .jsonl).')
    return value


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Cannot read JSON file {path.name}: {error}') from error
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} must contain a JSON object.')
    return value


def validate_config(config):
    for key in ('rtl_key', 'clock_key'):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f'Configuration requires a nonempty {key}.')
    metrics = config.get('metrics')
    if not isinstance(metrics, list) or not metrics or any(not isinstance(m, str) or not m for m in metrics):
        raise ValueError('Configuration metrics must be a nonempty list of names.')
    if len(set(metrics)) != len(metrics):
        raise ValueError('Configuration metrics must be unique.')
    for key in ('max_input_len', 'max_target_len', 'num_digits', 'num_samples'):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'Configuration {key} must be a positive integer.')
    for key in ('train_splits', 'validation_splits'):
        values = config.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f'Configuration {key} must be a nonempty list.')
        for value in values:
            split_name(value)
    training = config.get('training')
    if not isinstance(training, dict) or 'eval_batch_size' not in training:
        raise ValueError('Configuration training requires eval_batch_size.')
    for key, value in training.items():
        if key not in TRAINING_OPTIONS:
            raise ValueError(f'Unsupported training setting: {key}')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Training setting {key} must be a finite number.')
        if value < 0 or (value == 0 and key not in ('weight_decay', 'dataloader_num_workers')):
            raise ValueError(f'Training setting {key} is outside its valid range.')
        if key not in ('learning_rate', 'weight_decay', 'num_train_epochs') and not isinstance(value, int):
            raise ValueError(f'Training setting {key} must be an integer.')
    checksums = config.get('checksums', {})
    if not isinstance(checksums, dict) or set(checksums) - {'splits', 'model'}:
        raise ValueError('Optional checksums accepts only splits and model mappings.')
    for group in ('splits', 'model'):
        values = checksums.get(group, {})
        if not isinstance(values, dict):
            raise ValueError(f'checksums.{group} must map names to SHA-256 values.')
        for name, checksum in values.items():
            if group == 'splits':
                split_name(name)
            elif not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name or name.startswith('.'):
                raise ValueError('Model checksum keys must be plain filenames.')
            if not isinstance(checksum, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', checksum):
                raise ValueError(f'Invalid SHA-256 for {group}.{name}')


def validate_split(path, config):
    """Check formatter inputs without loading tokenizer, tensors or model weights.

    The formatter filters non-numeric rows. Rejecting those rows at this public
    interface exposes incompatible input instead of silently changing its size.
    Identifiers are optional, matching the existing formatter.
    """
    if not path.is_file():
        raise ValueError(f'Missing dataset split: {path.name}')
    columns = None
    count = 0
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path.name} line {number}: invalid JSON.') from error
            if not isinstance(row, dict):
                raise ValueError(f'{path.name} line {number}: expected a JSON object.')
            if columns is None:
                # The dataset resolves one column per metric before reading rows.
                # Use the same precedence (metric_value before metric).
                columns = {}
                for metric in config['metrics']:
                    candidates = METRIC_COLUMNS.get(metric, [metric])
                    column = next((name for name in candidates if name in row), None)
                    if column is None:
                        raise ValueError(f'{path.name}: missing metric {metric}; expected one of {candidates}.')
                    columns[metric] = column
            rtl = row.get(config['rtl_key'])
            if not isinstance(rtl, str) or not rtl.strip():
                raise ValueError(f'{path.name} line {number}: {config["rtl_key"]} must contain source text.')
            for key in [config['clock_key'], *columns.values()]:
                try:
                    value = row[key]
                    parsed = float(value)
                    if isinstance(value, bool) or not math.isfinite(parsed):
                        raise ValueError()
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f'{path.name} line {number}: {key} must be a finite number.') from error
            count += 1
    if not count:
        raise ValueError(f'Dataset split {path.name} is empty.')
    return count


def validate_model(model):
    """Validate a local Hugging Face seq2seq layout without deserializing weights."""
    if not model.is_dir():
        raise ValueError('Model must be a local Hugging Face checkpoint directory.')
    metadata = read_json(model / 'config.json')
    if not isinstance(metadata.get('model_type'), str) or not metadata['model_type']:
        raise ValueError('Model config.json requires model_type.')
    seq2seq_types = {'t5gemma', 't5', 'mt5', 'longt5', 'bart', 'mbart', 'pegasus', 'encoder-decoder'}
    if metadata.get('is_encoder_decoder') is False or (metadata.get('is_encoder_decoder') is not True and metadata['model_type'] not in seq2seq_types):
        raise ValueError('PPA backend requires encoder-decoder metadata compatible with AutoModelForSeq2SeqLM.')
    read_json(model / 'tokenizer_config.json')
    if not any((model / name).is_file() for name in ('tokenizer.json', 'tokenizer.model', 'spiece.model', 'sentencepiece.bpe.model', 'vocab.json')):
        raise ValueError('Checkpoint must include local tokenizer vocabulary (for example tokenizer.json or spiece.model).')
    if (model / 'model.safetensors').is_file() or (model / 'pytorch_model.bin').is_file():
        return
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index = model / name
        if not index.is_file():
            continue
        mapping = read_json(index).get('weight_map')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f'{name} must contain a nonempty weight_map.')
        for shard in mapping.values():
            if not isinstance(shard, str) or Path(shard).name != shard or '/' in shard or '\\' in shard or not (model / shard).is_file():
                raise ValueError(f'{name} references a missing or invalid weight shard.')
        return
    raise ValueError('Checkpoint requires model.safetensors, pytorch_model.bin, or an index and all weight shards.')


def validate(args, config):
    validate_config(config)
    repo = args.repo.resolve()
    source = repo / 'src/text2text_regression'
    if not (source / 'finetune_ppa.py').is_file() or not (source / 'evaluate_ppa.py').is_file():
        raise ValueError('Provide a release checkout containing training and evaluation code with --repo.')
    if args.operation != 'train' and not args.split:
        raise ValueError('--split is required for evaluation and prediction.')
    if args.split:
        split_name(args.split)
    model = Path(args.model).resolve()
    validate_model(model)
    splits = config['train_splits'] + config['validation_splits'] if args.operation == 'train' else [args.split]
    for split in set(splits):
        path = args.dataset / (split + '.jsonl')
        validate_split(path, config)
        expected = config.get('checksums', {}).get('splits', {}).get(split)
        if expected and digest(path).lower() != expected.lower():
            raise ValueError(f'Dataset split checksum mismatch: {path.name}')
    for name, expected in config.get('checksums', {}).get('model', {}).items():
        path = model / name
        if not path.is_file() or digest(path).lower() != expected.lower():
            raise ValueError(f'Model file checksum mismatch: {name}')
    output = args.output.resolve()
    protected = [args.dataset.resolve(), repo, model]
    if any(output == path or output.is_relative_to(path) for path in protected):
        raise ValueError('Output must be outside the code, dataset and model directories.')
    if output.exists():
        raise ValueError('Output already exists; choose a fresh output directory.')
    return source


def command(args, config, source):
    if args.operation == 'evaluate':
        cmd = [sys.executable, str(source / 'evaluate_ppa.py'), '--dataset-dir', str(args.dataset.resolve()),
               '--checkpoint', str(Path(args.model).resolve()), '--output-dir', str(args.output.resolve()),
               '--split', args.split, '--device', args.device, '--metrics', *config['metrics']]
        for key in ('rtl_key', 'clock_key', 'max_input_len', 'max_target_len', 'num_digits', 'num_samples'):
            cmd += ['--' + key.replace('_', '-'), str(config[key])]
        return cmd + ['--batch-size', str(config['training']['eval_batch_size'])]
    cmd = [sys.executable, str(source / 'finetune_ppa.py'), '--dataset_dir', str(args.dataset.resolve()),
           '--output_dir', str(args.output.resolve()), '--logging_dir', str(args.output.resolve() / 'logs')]
    for key in ('rtl_key', 'clock_key', 'max_input_len', 'max_target_len', 'num_digits', 'num_samples'):
        cmd += ['--' + key, str(config[key])]
    for key in ('metrics', 'train_splits', 'validation_splits'):
        cmd += ['--' + key, *config[key]]
    cmd += ['--callback_eval_splits', *config['validation_splits']]
    # Training constructs evaluation datasets even without --do_eval. Select
    # those explicitly too, rather than discovering additional test files.
    cmd += ['--eval_splits', *dict.fromkeys(config['train_splits'] + config['validation_splits'])]
    for key, value in config['training'].items():
        cmd += ['--' + key, str(value)]
    return cmd + ['--do_train', '--encoder_model_name', str(Path(args.model).resolve())]


def predict(args, config, source):
    # Reuse the original formatter, sampled decoding and median computation.
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq
    from omnipred import T5GemmaModel, T5GemmaModelConfig
    from utils import MultiMetricSeq2SeqDataset
    rows = []
    with (args.dataset / (args.split + '.jsonl')).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= args.limit:
                    break
    model_path = str(Path(args.model).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dataset = MultiMetricSeq2SeqDataset(tokenizer=tokenizer, dataset=Dataset.from_list(rows),
             max_input_len=config['max_input_len'], max_target_len=config['max_target_len'],
             metrics=config['metrics'], rtl_key=config['rtl_key'], clock_key=config['clock_key'],
             num_digits=config['num_digits'])
    device = args.device
    dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
    model = T5GemmaModel(T5GemmaModelConfig(model_name=model_path,
            max_input_len=config['max_input_len'], max_target_len=config['max_target_len'],
            model_kwargs={'torch_dtype': dtype, 'local_files_only': True},
            tokenizer_kwargs={'local_files_only': True})).to(device).eval()
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model.model)
    torch.manual_seed(42)
    results = []
    for i, sample in enumerate(dataset.samples):
        batch = {key: value.to(device) for key, value in collator([dataset[i]]).items()}
        with torch.no_grad():
            tokens = model.decode(batch, num_samples=config['num_samples'])[0]
        draws = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        valid = []
        for draw in draws:
            try:
                values = [float(value.strip()) for value in draw.strip().split('\n') if value.strip()]
                if len(values) == len(config['metrics']) and all(np.isfinite(values)):
                    valid.append(values)
            except ValueError:
                pass
        prediction = np.median(valid, axis=0).tolist() if valid else None
        results.append({'design_name': sample['design_name'], 'family_id': sample['family_id'],
                        'clock': sample['clock'], 'prediction': prediction, 'target': sample['y'],
                        'decoded_draws': draws})
    args.output.mkdir(parents=True)
    report = {'split': args.split, 'device': device, 'metrics': config['metrics'],
              'num_samples': config['num_samples'], 'records': results,
              'note': 'Fresh sampled inference; historical sampled predictions need not be identical.'}
    (args.output / 'predictions.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'samples': len(results), 'decoded': sum(row['prediction'] is not None for row in results),
                      'output': str(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['train', 'evaluate', 'predict'])
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--model', required=True, help='Local Hugging Face checkpoint with tokenizer')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', help='Required for evaluate/predict; names a JSONL file without its extension')
    parser.add_argument('--limit', type=int, default=3, help='Maximum prediction examples (default: 3)')
    parser.add_argument('--device', default='cpu', help='Evaluation/prediction device; training uses Trainer placement')
    parser.add_argument('--dry-run', action='store_true', help='Validate artifacts and show dispatch without loading weights')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('Use a positive limit.')
    try:
        config = read_json(args.config)
        source = validate(args, config)
        if args.dry_run:
            print(json.dumps({'operation': args.operation, 'cwd': str(source),
                              'command': command(args, config, source) if args.operation != 'predict' else None}, indent=2))
            return
        if args.operation == 'predict':
            predict(args, config, source)
        else:
            env = os.environ.copy()
            env['WANDB_MODE'] = 'offline'
            env['TOKENIZERS_PARALLELISM'] = 'false'
            subprocess.run(command(args, config, source), cwd=source, env=env, check=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
