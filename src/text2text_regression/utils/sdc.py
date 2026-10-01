"""Locally authored templates for conventional zero-I/O-delay SDC constraints.

Clocked designs exclude the clock port from data-input delays. Combinational
designs use a virtual clock. These interfaces replace a formerly copied
template module; the MetRex inspiration is cited in THIRD_PARTY_NOTICES.md.
"""

def _commands(*lines):
    return "\n".join(lines) + "\n"


SDC_CONSTRAINTS_CLOCK = _commands(
    "create_clock -period {clock_period} -name {clock_name} [get_ports {clock_port}]",
    "set clk_input [get_ports {clock_port}]",
    "set all_inputs_wo_clk [remove_from_collection [all_inputs] $clk_input]",
    "set_input_delay 0 -clock {clock_name} $all_inputs_wo_clk",
    "set_output_delay 0 -clock {clock_name} [all_outputs]",
    "set_max_fanout {max_fanout} [current_design]",
)

SDC_CONSTRAINTS_WO_CLOCK = _commands(
    "create_clock -period {clock_period} -name {clock_name}",
    "set all_inputs_wo_clk [all_inputs]",
    "set clk_input {{}}",
    "set_input_delay 0 -clock {clock_name} [all_inputs]",
    "set_output_delay 0 -clock {clock_name} [all_outputs]",
    "set_max_fanout {max_fanout} [current_design]",
)

SDC_DRIVER = _commands(
    "set_driving_cell -pin {driving_cell_pin} -lib_cell {driving_cell} $all_inputs_wo_clk",
    "if {{[sizeof_collection $clk_input] > 0}} {{",
    "  set_driving_cell -pin {driving_cell_pin} -lib_cell {driving_cell} $clk_input",
    "}}",
)
