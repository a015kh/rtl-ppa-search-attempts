import os
from datasets import load_dataset

def load_hf_dataset(path):
    files = os.listdir(path)
    files = sorted(files)
    # Map file names to their paths
    # train.jsonl -> train, valid.jsonl -> eval
    data_files = {}
    for f in files:
        if f.endswith(".jsonl"):
            file_name = f.replace(".jsonl", "")
            data_files[file_name] = f
    dataset = load_dataset(path, data_files=data_files)
    return dataset