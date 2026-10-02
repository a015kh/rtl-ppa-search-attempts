"""Temporary synthetic inputs for the measured-record dataset builder.

All RTL, reports, labels and upstream identities below are newly fabricated.
No test depends on retained research data, remote datasets or EDA tools.
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from revolution import dataset_generation as dg


ROOT = Path(__file__).resolve().parents[2]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def measured_row(index, *, design=None, rtl=None):
    return {
        'SDC': 'create_clock -period 2.0 -name clk [get_ports clk]',
        'has_clock': True, 'clock': 2.0,
        'area_value': 10.0 + index, 'delay_value': 0.5,
        'wns': 1.5, 'tns': 0.0, 'total_power_value': 0.25,
        'static_power_value': 0.10, 'switching_power_value': 0.15,
        'synth': 'synthetic cell summary', 'area': 'synthetic area calculation',
        'delay': 'synthetic timing calculation',
        'static_power': 'synthetic static power', 'switching_power': 'synthetic dynamic power',
        'synthesis_time': 0.01, 'design_name': design or f'synthetic_{index}',
        'rtl': rtl or f'module synthetic_{index}(input clk, input a, output y); assign y = a; endmodule',
    }


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.legacy = self.directory / 'measured'
        self.legacy.mkdir()
        self.seed = self.directory / 'seed.jsonl'
        self.config_file = self.directory / 'builder.toml'
        self.base = [measured_row(index) for index in range(8)]
        train = self.base[:4]
        duplicate = dict(self.base[0], area_value=999.0)
        write_rows(self.legacy / 'train.jsonl', train)
        write_rows(self.legacy / 'valid.jsonl', [*self.base[4:], duplicate])
        write_rows(self.seed, [{'rtl': row['rtl']} for row in self.base])
        catalog = []
        for index, row in enumerate(self.base):
            catalog.append({
                'rtl_hash': dg.sha256_text(dg.normalized_rtl(row['rtl'])),
                'source_dataset': 'synthetic/preferred', 'source_revision': 'fixture-r1',
                'source_split': 'train', 'source_row_index': index,
                'module_id': f'synthetic-preferred-{index}',
            })
        catalog.append({**catalog[0], 'source_dataset': 'synthetic/secondary',
                        'source_revision': 'fixture-r2', 'module_id': 'synthetic-secondary-0'})
        write_rows(self.legacy / 'metadata/source_catalog.jsonl', catalog)
        augmented = [measured_row(
            index, design=f'sample_{index:06d}_area_gen1_cand0',
            rtl=f'module augmented_{index}(input clk, input a, output y); assign y = ~a; endmodule',
        ) for index in range(8)]
        write_rows(self.legacy / 'augmentation_first.jsonl', augmented)
        write_rows(self.legacy / 'augmentation_second.jsonl', [dict(row, area_value=888.0) for row in augmented])
        write_rows(self.legacy / 'test_synthetic.jsonl', [measured_row(20), measured_row(21)])
        self.config_file.write_text('''
[dataset]
output_root = "output"
legacy_final = "measured"
seed_train = "seed.jsonl"
catalog_cache = "cache/catalog.jsonl"
split_seed = 42
valid_fraction = 0.25
split_tolerance = 0.001

[[upstream]]
name = "synthetic/preferred"
revision = "fixture-r1"
split = "train"
rtl_field = "rtl"
priority = 1

[[upstream]]
name = "synthetic/secondary"
revision = "fixture-r2"
split = "train"
rtl_field = "rtl"
priority = 2

[[upstream]]
name = "synthetic/unused"
revision = "fixture-r3"
split = "train"
rtl_field = "RTL"
priority = 3

[[augmentation]]
origin = "synthetic_first"
legacy_shard = "augmentation_first.jsonl"
model = "synthetic-generator-first"
forced_source = ""

[[augmentation]]
origin = "synthetic_second"
legacy_shard = "augmentation_second.jsonl"
model = "synthetic-generator-second"
forced_source = ""

[tests]
shards = ["test_synthetic.jsonl"]
''', encoding='utf-8')
        self.config = dg.load_config(self.config_file)

    def build(self, config=None):
        selected = config or self.config
        result = dg.build_command(selected, allow_network=False, publish_result=True)
        self.assertEqual(result['status'], 'published')
        return selected.output_root

    def split_records(self, output):
        return {name: read_rows(output / name) for name in ('train.jsonl', 'valid.jsonl',
                                                          'train_aug.jsonl', 'valid_aug.jsonl',
                                                          'test_synthetic.jsonl')}

    def test_paths_are_relative_to_configuration_not_working_directory(self):
        self.assertEqual(self.config.output_root, self.directory / 'output')
        self.assertEqual(self.config.legacy_final, self.legacy)
        self.assertEqual(self.config.seed_train, self.seed)
        self.assertEqual(self.config.catalog_cache, self.directory / 'cache/catalog.jsonl')

    def test_offline_preflight_selects_snapshot(self):
        report = dg.preflight(self.config, allow_network=False)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['catalog_mode'], 'snapshot')
        self.assertFalse(self.config.output_root.exists())

    def test_real_build_verify_schema_and_numeric_labels(self):
        output = self.build()
        manifest = json.loads((output / 'dataset_manifest.json').read_text())
        expected = {'train.jsonl': 6, 'valid.jsonl': 2, 'train_aug.jsonl': 6,
                    'valid_aug.jsonl': 2, 'test_synthetic.jsonl': 2}
        self.assertEqual(manifest['output_counts'], expected)
        records = self.split_records(output)
        for split, rows in records.items():
            self.assertEqual(len(rows), expected[split])
            for row in rows:
                self.assertEqual(tuple(row), dg.ALL_FIELDS)
                self.assertEqual(row['record_id'], 'sha256:' + dg.identity_digest(row))
                self.assertEqual(row['clock'], 2.0)
                self.assertEqual(row['delay_value'], 0.5)
                self.assertEqual(row['total_power_value'], 0.25)
        verified = dg.verify_dataset(output, self.config)
        self.assertEqual(verified['counts'], expected)
        self.assertEqual(verified['output_sha256'], manifest['output_sha256'])
        self.assertFalse((output / '.work').exists())

    def test_dedup_and_upstream_priority_preserve_first_measurement(self):
        records = self.split_records(self.build())
        base = records['train.jsonl'] + records['valid.jsonl']
        original = next(row for row in base if row['design_name'] == 'synthetic_0')
        self.assertEqual(original['area_value'], 10.0)
        self.assertEqual(original['source_dataset'], 'synthetic/preferred')
        aug = records['train_aug.jsonl'] + records['valid_aug.jsonl']
        self.assertEqual(len(aug), 8)
        self.assertTrue(all(row['data_origin'] == 'synthetic_first' for row in aug))
        self.assertTrue(all(row['generation_model'] == 'synthetic-generator-first' for row in aug))
        self.assertTrue(all(row['area_value'] != 888.0 for row in aug))

    def test_seed_and_augmentation_share_deterministic_split(self):
        first = self.build()
        first_records = self.split_records(first)
        second_config = replace(self.config, output_root=self.directory / 'output-again')
        second = self.build(second_config)
        for name in first_records:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        lineage_splits = {}
        for name, rows in first_records.items():
            if name.startswith('test'):
                continue
            split = 'valid' if name.startswith('valid') else 'train'
            for row in rows:
                self.assertEqual(lineage_splits.setdefault(row['lineage_id'], split), split)
        self.assertEqual(len(lineage_splits), 8)
        self.assertTrue(all(row['seed_present_in_base'] for row in
                            first_records['train_aug.jsonl'] + first_records['valid_aug.jsonl']))

    def test_true_tests_keep_test_provenance_without_training_assignment(self):
        rows = read_rows(self.build() / 'test_synthetic.jsonl')
        self.assertTrue(all(row['record_kind'] == 'test' for row in rows))
        self.assertTrue(all(row['source_split'] == 'test' for row in rows))
        self.assertTrue(all(row['generation_model'] is None for row in rows))
        self.assertTrue(all(row['seed_present_in_base'] is None for row in rows))

    def test_optional_empty_augmentation_preflights_but_retains_fraction_check(self):
        config = replace(self.config, augmentation=(), test_shards=(), test_sources={})
        self.assertEqual(dg.preflight(config, allow_network=False)['status'], 'ok')
        with self.assertRaisesRegex(dg.UnifiedDatasetError, 'augmentation validation fraction'):
            dg.build_command(config, allow_network=False, publish_result=True)
        self.assertFalse(config.output_root.exists())

    def test_verify_rejects_changed_identity(self):
        output = self.build()
        path = output / 'train.jsonl'
        rows = read_rows(path)
        rows[0]['design_name'] = 'changed-synthetic-identity'
        write_rows(path, rows)
        with self.assertRaises(dg.UnifiedDatasetError):
            dg.verify_dataset(output, self.config)

    def test_verify_rejects_schema_change(self):
        output = self.build()
        path = output / 'valid.jsonl'
        rows = read_rows(path)
        rows[0]['unexpected_field'] = 'synthetic'
        write_rows(path, rows)
        with self.assertRaises(dg.UnifiedDatasetError):
            dg.verify_dataset(output, self.config)

    def test_verify_enforces_recorded_output_hash_for_valid_label_change(self):
        output = self.build()
        path = output / 'train.jsonl'
        rows = read_rows(path)
        rows[0]['area_value'] += 0.1
        write_rows(path, rows)
        with self.assertRaisesRegex(dg.UnifiedDatasetError, 'SHA-256 mismatch'):
            dg.verify_dataset(output, self.config)

    def test_verify_rejects_lineage_leakage(self):
        output = self.build()
        source = output / 'train_aug.jsonl'
        destination = output / 'valid_aug.jsonl'
        train = read_rows(source)
        valid = read_rows(destination)
        valid.append(train.pop())
        write_rows(source, train)
        write_rows(destination, valid)
        # Update counts/hashes so the explicit lineage invariant is tested,
        # rather than merely relying on saved file-integrity checks.
        manifest = json.loads((output / 'dataset_manifest.json').read_text())
        for name in ('train_aug.jsonl', 'valid_aug.jsonl'):
            manifest['output_counts'][name] = len(read_rows(output / name))
            manifest['output_sha256'][name] = dg.sha256_file(output / name)
        with self.assertRaisesRegex(dg.UnifiedDatasetError, 'lineage leakage'):
            dg.verify_dataset(output, self.config, manifest=manifest)

    def test_invalid_numeric_labels_reject_build_without_publication(self):
        for value in (-1.0, float('nan'), float('inf'), 'not-a-number'):
            with self.subTest(value=value):
                write_rows(self.legacy / 'train.jsonl', [dict(self.base[0], area_value=value), *self.base[1:4]])
                with self.assertRaises(dg.UnifiedDatasetError):
                    dg.build_command(self.config, allow_network=False, publish_result=True)
                self.assertFalse(self.config.output_root.exists())
                self.assertFalse(list(self.directory.glob('.output.staging.*')))

    def test_missing_ppa_field_rejects_build(self):
        row = dict(self.base[0])
        del row['total_power_value']
        write_rows(self.legacy / 'train.jsonl', [row, *self.base[1:4]])
        with self.assertRaisesRegex(dg.UnifiedDatasetError, 'missing PPA fields'):
            dg.build_command(self.config, allow_network=False, publish_result=True)

    def test_preflight_missing_source_is_actionable(self):
        (self.legacy / 'valid.jsonl').unlink()
        with self.assertRaisesRegex(dg.UnifiedDatasetError, 'valid.jsonl'):
            dg.preflight(self.config, allow_network=False)

    def test_offline_catalog_missing_never_imports_upstream_library(self):
        (self.legacy / 'metadata/source_catalog.jsonl').unlink()
        with patch.dict(sys.modules, {'datasets': None}), self.assertRaises(dg.UnifiedDatasetError):
            dg.preflight(self.config, allow_network=False)
        self.assertFalse(self.config.catalog_cache.exists())

    def test_missing_optional_network_dependency_is_actionable(self):
        (self.legacy / 'metadata/source_catalog.jsonl').unlink()
        builder = dg.Builder(self.config, self.directory / 'temporary-build', allow_network=True)
        self.addCleanup(builder.close)
        with patch.dict(sys.modules, {'datasets': None}), self.assertRaisesRegex(dg.UnifiedDatasetError, 'datasets'):
            builder.build_catalog()
        self.assertFalse(self.config.catalog_cache.exists())

    def test_catalog_revision_mismatch_is_rejected(self):
        path = self.legacy / 'metadata/source_catalog.jsonl'
        rows = read_rows(path)
        rows[0]['source_revision'] = 'wrong-fixture-revision'
        write_rows(path, rows)
        with self.assertRaisesRegex(dg.UnifiedDatasetError, 'catalog revision mismatch'):
            dg.build_command(self.config, allow_network=False, publish_result=True)

    def test_existing_output_is_preserved(self):
        self.config.output_root.mkdir()
        marker = self.config.output_root / 'sentinel.txt'
        marker.write_text('keep this previous output')
        with self.assertRaises(dg.UnifiedDatasetError):
            dg.build_command(self.config, allow_network=False, publish_result=True)
        self.assertEqual(marker.read_text(), 'keep this previous output')

    def test_output_overlap_with_input_is_rejected(self):
        for output in (self.legacy, self.legacy / 'nested-output', self.directory):
            with self.subTest(output=output), self.assertRaises(dg.UnifiedDatasetError):
                dg.build_command(replace(self.config, output_root=output), allow_network=False, publish_result=True)
        self.assertEqual(len(read_rows(self.legacy / 'train.jsonl')), 4)

    def test_cache_overlap_with_input_is_rejected(self):
        config = replace(self.config, catalog_cache=self.seed)
        with self.assertRaises(dg.UnifiedDatasetError):
            dg.preflight(config, allow_network=False)
        self.assertEqual(len(read_rows(self.seed)), 8)

    def test_source_checkout_is_protected_as_output(self):
        output = ROOT / 'synthetic-output-that-must-not-be-created'
        with self.assertRaises(dg.UnifiedDatasetError):
            dg.build_command(replace(self.config, output_root=output), allow_network=False, publish_result=True)
        self.assertFalse(output.exists())

    def test_builder_help_works_without_configuration_or_private_paths(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_dataset.py'), '--help'],
                                cwd=self.directory, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('preflight', result.stdout)
        operation = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_dataset.py'), 'preflight', '--help'],
                                   cwd=self.directory, text=True, capture_output=True)
        self.assertEqual(operation.returncode, 0, operation.stderr)
        self.assertIn('--config', operation.stdout)
        for forbidden in ('/home/', '/tmp2/', 'checkpoint-'):
            self.assertNotIn(forbidden, result.stdout)
            self.assertNotIn(forbidden, operation.stdout)
        self.assertNotRegex(result.stdout, r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.assertNotRegex(operation.stdout, r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

    def test_cli_configuration_is_required(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_dataset.py'), 'preflight'],
                                cwd=self.directory, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--config', result.stderr)

    def test_network_is_opt_in_for_preflight_and_build(self):
        for operation in ('preflight', 'build'):
            with self.subTest(operation=operation):
                arguments = [operation, '--config', str(self.config_file)]
                self.assertFalse(dg.parser().parse_args(arguments).allow_network)
                self.assertTrue(dg.parser().parse_args([*arguments, '--allow-network']).allow_network)

    def test_cli_verifies_real_output_from_another_directory(self):
        output = self.build()
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_dataset.py'),
                                 'verify', '--config', str(self.config_file), '--dataset-root', str(output)],
                                cwd=self.directory, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(sum(report['counts'].values()), 18)
        self.assertEqual(report['output_sha256'],
                         json.loads((output / 'dataset_manifest.json').read_text())['output_sha256'])


class TransformationTests(unittest.TestCase):
    def test_clock_normalization_preserves_identity(self):
        row = measured_row(0)
        self.assertEqual(dg.identity_digest(row), dg.identity_digest(dict(row, clock='2.000')))
        self.assertEqual(dg.canonical_clock('2e0'), '2')
        with self.assertRaises(dg.UnifiedDatasetError):
            dg.canonical_clock(float('nan'))

    def test_lineage_normalizes_whitespace_without_altering_record_rtl(self):
        rtl = 'module example;\n  endmodule'
        self.assertEqual(dg.rtl_lineage(rtl), dg.rtl_lineage('module example; endmodule'))
        row = measured_row(0, rtl=rtl)
        self.assertEqual(dg.normalize_ppa_row(row, 'synthetic')['rtl'], rtl)

    def test_augmentation_identity_fields_are_preserved(self):
        parsed = dg.parse_augmentation_id('prefix_sample_000123_delay_gen4_cand2_suffix')
        self.assertEqual(parsed, {'seed_sample_id': 'sample_000123', 'seed_row_index': 123,
                                  'optimization_direction': 'delay', 'generation_index': 4,
                                  'candidate_index': 2})
        with self.assertRaises(dg.UnifiedDatasetError):
            dg.parse_augmentation_id('unrelated-synthetic-name')

    def test_report_formatting_preserves_measured_labels(self):
        row = measured_row(0)
        row['synth'] = 'Report : cell\nCell Reference Library Area\nu0 INV_X1 synthetic 1.25\nu1 INV_X1 synthetic 1.25\nTotal 2\n'
        row['delay'] = 'Report : timing\nNo paths.\n'
        normalized = dg.normalize_ppa_row(row, 'synthetic')
        self.assertIn('2 INV_X1', normalized['synth'])
        self.assertIn('total area is 2.50', normalized['area'])
        self.assertIn('total delay is 0.00', normalized['delay'])
        self.assertEqual(normalized['area_value'], row['area_value'])
        self.assertEqual(normalized['delay_value'], row['delay_value'])

    def test_lineage_assignment_deterministic_for_input_order(self):
        groups = {f'synthetic-{index}': (index % 3 + 1, index % 2 + 1) for index in range(20)}
        first = dg.Builder.split_lineages(groups, 42, 0.25)
        second = dg.Builder.split_lineages(dict(reversed(list(groups.items()))), 42, 0.25)
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
