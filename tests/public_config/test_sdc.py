"""Formatting checks for the locally authored generic SDC templates."""
import importlib.util
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('release_sdc_templates', ROOT / 'src/text2text_regression/utils/sdc.py')
sdc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sdc)


class SDCTemplateTests(unittest.TestCase):
    def test_clocked_format_and_data_collection(self):
        text = sdc.SDC_CONSTRAINTS_CLOCK.format(clock_period=2.5, clock_name='sample_clock',
                                                clock_port='clock_i', max_fanout=12)
        self.assertIn('create_clock -period 2.5 -name sample_clock [get_ports clock_i]', text)
        self.assertIn('set clk_input [get_ports clock_i]', text)
        self.assertIn('[remove_from_collection [all_inputs] $clk_input]', text)
        self.assertIn('set_input_delay 0 -clock sample_clock $all_inputs_wo_clk', text)
        self.assertIn('set_output_delay 0 -clock sample_clock [all_outputs]', text)
        self.assertIn('set_max_fanout 12 [current_design]', text)
        self.assertTrue(text.endswith('\n'))

    def test_clockless_format_uses_virtual_clock(self):
        text = sdc.SDC_CONSTRAINTS_WO_CLOCK.format(clock_period=4, clock_name='virtual_clock', max_fanout=9)
        clock = text.splitlines()[0]
        self.assertEqual(clock, 'create_clock -period 4 -name virtual_clock')
        self.assertNotIn('get_ports', clock)
        self.assertIn('set_input_delay 0 -clock virtual_clock [all_inputs]', text)
        self.assertIn('set_output_delay 0 -clock virtual_clock [all_outputs]', text)
        self.assertIn('set_max_fanout 9 [current_design]', text)

    def test_driver_cell_is_explicit_and_formattable(self):
        text = sdc.SDC_DRIVER.format(driving_cell='synthetic_driver', driving_cell_pin='Y')
        self.assertIn('set_driving_cell', text)
        self.assertIn('-pin Y -lib_cell synthetic_driver', text)
        self.assertTrue(text.endswith('\n'))

    def test_driver_variables_defined_for_both_constraint_modes(self):
        for template, mode in ((sdc.SDC_CONSTRAINTS_CLOCK, 'clocked'),
                               (sdc.SDC_CONSTRAINTS_WO_CLOCK, 'virtual')):
            with self.subTest(mode=mode):
                constraints = template.format(clock_period=2, clock_name='clk', clock_port='clock_i', max_fanout=10)
                driver = sdc.SDC_DRIVER.format(driving_cell='synthetic_driver', driving_cell_pin='Y')
                text = constraints + driver
                used = set(re.findall(r'\$([A-Za-z_][A-Za-z0-9_]*)', text))
                declared = set(re.findall(r'^set\s+([A-Za-z_][A-Za-z0-9_]*)\s', text, re.M))
                self.assertLessEqual(used, declared)


if __name__ == '__main__':
    unittest.main()
