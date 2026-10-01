"""Evaluate a checkpoint without loading training data or constructing a Trainer."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--split',required=True)
    p.add_argument('--metrics',nargs='+',default=['area','delay','total_power'])
    p.add_argument('--rtl-key',default='rtl')
    p.add_argument('--clock-key',default='clock')
    p.add_argument('--max-input-len',type=int,default=3072)
    p.add_argument('--max-target-len',type=int,default=128)
    p.add_argument('--num-digits',type=int,default=4)
    p.add_argument('--num-samples',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',default='cpu')
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.output_dir.exists():p.error('Choose a new output directory.')
    if '/' in a.split or '\\' in a.split or a.split.startswith('.'):p.error('Use a plain split name.')
    if a.limit is not None and a.limit<1:p.error('limit must be positive')
    import pandas as pd
    import torch
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq
    from omnipred import T5GemmaModel, T5GemmaModelConfig
    from utils import MultiMetricSeq2SeqDataset
    from ppa_evaluation import evaluate_model
    rows=[]
    with (a.dataset_dir/(a.split+'.jsonl')).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
            if a.limit and len(rows)>=a.limit:break
    checkpoint=str(a.checkpoint.resolve())
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,model_max_length=a.max_input_len)
    dataset=MultiMetricSeq2SeqDataset(tokenizer=tokenizer,dataset=Dataset.from_list(rows),max_input_len=a.max_input_len,max_target_len=a.max_target_len,metrics=a.metrics,rtl_key=a.rtl_key,clock_key=a.clock_key,num_digits=a.num_digits)
    config=T5GemmaModelConfig(model_name=checkpoint,max_input_len=a.max_input_len,max_target_len=a.max_target_len)
    model=T5GemmaModel.from_pretrained(checkpoint,config=config).to(a.device)
    loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=0,collate_fn=DataCollatorForSeq2Seq(model=model.model,tokenizer=tokenizer,max_length=a.max_input_len,padding='longest'))
    a.output_dir.mkdir(parents=True)
    result=evaluate_model(model,loader,a.split,tokenizer=tokenizer,device=torch.device(a.device),metrics=a.metrics,num_samples=a.num_samples,output_dir=str(a.output_dir))
    pd.DataFrame({a.split:result}).T.to_csv(a.output_dir/'evaluation_results.csv')
    (a.output_dir/'evaluation_config.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
