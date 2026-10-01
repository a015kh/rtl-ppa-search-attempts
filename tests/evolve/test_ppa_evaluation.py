"""Preserve saved-metric semantics while separating evaluation from training."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ppa_evaluation',ROOT/'src/text2text_regression/ppa_evaluation.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)

def test_median_decoding_failures_and_zero_targets_keep_original_metrics(tmp_path):
    class Model:
        def eval(self):return self
        def decode(self,batch,num_samples):
            assert num_samples==2
            return torch.tensor([[[1],[2]],[[3],[4]],[[2],[2]]])
    class Tokenizer:
        pad_token_id=0
        def batch_decode(self,rows,skip_special_tokens=True):
            values={1:'10\n2\n4',2:'invalid',3:'0\n1\n1',4:'0\n1\n1',10:'10\n2\n4',11:'0\n0\n0',12:'1\n1\n1'}
            return [values[row[0]] for row in rows]
    class Loader(list):pass
    loader=Loader([{'input_ids':torch.ones(3,1,dtype=torch.long),'labels':torch.tensor([[10],[11],[12]])}])
    loader.dataset=SimpleNamespace(samples=[{'design_name':str(i),'family_id':'f','clock':1} for i in range(3)])
    result=evaluation.evaluate_model(Model(),loader,'fixture',tokenizer=Tokenizer(),device=torch.device('cpu'),metrics=['area','delay','total_power'],num_samples=2,output_dir=str(tmp_path))
    assert result['total_samples']==2
    assert result['decode_fail']==1
    assert result['avg_relative_mae']==pytest.approx(2e10/6)
    assert result['avg_acc_at_20_percent']==66.67
    assert result['all_metrics_acc_at_20_percent']==50.0
    assert (tmp_path/'predictions_ground_truth_fixture.csv').is_file()
