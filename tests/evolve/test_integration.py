"""Regression checks for relocation and explicit artifact selection."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('workflow',['icrtl','four-pipeline','verilogeval','cvdp','compare'])
def test_dispatch_resolves_packaged_modules_from_another_directory(tmp_path,workflow):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_evolve.py'),workflow,'--config',str(ROOT/'configs/evolve.example.json'),'--','--help'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'usage:' in result.stdout

def test_no_implicit_output_directory(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_evolve.py'),'icrtl','--config',str(ROOT/'configs/evolve.example.json'),'--dry-run'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode!=0
    assert 'explicit --run-dir' in result.stderr

def test_evaluation_command_does_not_launch_training(tmp_path):
    spec=importlib.util.spec_from_file_location('ppa_workflow',ROOT/'scripts/ppa.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    args=SimpleNamespace(operation='evaluate',dataset=tmp_path,model=str(tmp_path/'model'),output=tmp_path/'out',split='test_pluto',device='cpu')
    config=json.loads((ROOT/'configs/ppa.example.json').read_text())
    command=m.command(args,config,ROOT/'src/text2text_regression')
    assert command[1].endswith('evaluate_ppa.py')
    assert '--do_train' not in command
    assert command[command.index('--max-input-len')+1]=='3072'
    assert command[command.index('--checkpoint')+1]==str((tmp_path/'model').resolve())

def test_three_value_formatter_backend_and_training_are_present():
    for name in ['finetune_ppa.py','evaluate_ppa.py','ppa_evaluation.py','omnipred/modeling_omnipred_ext.py','utils/ppa_dataset.py']:
        assert (ROOT/'src/text2text_regression'/name).is_file()
