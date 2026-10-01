import json
import os
import re
import ipdb
import sys

sys.path.append(".")

import numpy as np
import random
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from utils.sdc import SDC_CONSTRAINTS_CLOCK, SDC_DRIVER, SDC_CONSTRAINTS_WO_CLOCK

FLOAT_REGEX = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

def split_by_length(train_dataset, eval_dataset, path, num_splits=4):
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B-Instruct")
    train_ds = Dataset.from_list(train_dataset)
    eval_ds = Dataset.from_list(eval_dataset)
    train_ds = train_ds.map(
        lambda example: {
            "length": len(tokenizer(example["rtl"])["input_ids"])
        }
    )
    eval_ds = eval_ds.map(
        lambda example: {
            "length": len(tokenizer(example["rtl"])["input_ids"])
        }
    )
    length = sorted(train_ds["length"])
    percentiles = [length[int(len(length) * (i / (num_splits + 1)))] for i in range(0, num_splits + 1)]
    # do not include the max length in the last split
    percentiles[0] = 0
    print(f"Length percentiles: {percentiles}")

    length_dataset = {}
    for i in range(num_splits):
        l_prev = percentiles[i]
        l_curr = percentiles[i + 1]
        print(l_prev, l_curr)
        length_dataset[f"train_l{i}"] = train_ds.filter(lambda x: l_prev <= x["length"] and x["length"] < l_curr).shuffle(seed=42).select(range(128))
        length_dataset[f"eval_l{i}"] = eval_ds.filter(lambda x: l_prev <= x["length"] and x["length"] < l_curr).shuffle(seed=42).select(range(128))


    length_dataset = DatasetDict(length_dataset)
    length_dataset.save_to_disk(os.path.join(path, "length_split_512"))


def metrex_dataset_regression():
    path = "data/metrex_regression"
    os.makedirs(path, exist_ok=True)
    dataset = load_dataset("scale-lab/MetRex")
    all_data = []
    regex = re.compile(FLOAT_REGEX)
    for item in tqdm(dataset["train"]):
        rtl = item["RTL"]
        power = item["static_power"]
        power = float(regex.findall(power)[-1])
        area = item["area"]
        area = float(regex.findall(area)[-1])
        delay = item["delay"]
        delay = float(regex.findall(delay)[-1])
        area_sdc, delay_sdc, power_sdc = get_sdc(rtl)
        all_data.append({"rtl": rtl, "power": power, "area": area, "delay": delay, "area_sdc": area_sdc, "delay_sdc": delay_sdc, "power_sdc": power_sdc})
    train, valid = train_test_split(all_data, test_size=0.1, random_state=42)
    split_by_length(train, path)
    with open(os.path.join(path, "train.jsonl"), "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join(path, "valid.jsonl"), "w") as f:
        for item in valid:
            f.write(json.dumps(item) + "\n")
    # print(f"Dataset saved to {path}")
    # load original test files
    test_data_path = "metrex_dataset/eval/"
    all_test_data = {}
    test1_data = {}
    test2_data = {}
    test3_data = {}
    for metric in ["power", "area", "delay"]:
        with open(os.path.join(test_data_path, f"{metric}.json"), "r") as f:
            data = json.load(f)
        for split, items in data.items():
            for item in items:
                if item["x"] not in all_test_data:
                    area_sdc, delay_sdc, power_sdc = get_sdc(item["x"])
                    all_test_data[item["x"]] = {metric: item["y"], "area_sdc": area_sdc, "delay_sdc": delay_sdc, "power_sdc": power_sdc}
                else:
                    all_test_data[item["x"]][metric] = item["y"]
                if split == "test1":
                    test1_data[item["x"]] = all_test_data[item["x"]]
                elif split == "test2":
                    test2_data[item["x"]] = all_test_data[item["x"]]
                elif split == "test3":
                    test3_data[item["x"]] = all_test_data[item["x"]]
    print(f"Test data prepared: {len(test1_data)} (test1), {len(test2_data)} (test2), {len(test3_data)} (test3)")

    with open(os.path.join(path, "test.jsonl"), "w") as f:
        for key, item in all_test_data.items():
            item["rtl"] = key
            f.write(json.dumps(item) + "\n")
    with open(os.path.join(path, "test1.jsonl"), "w") as f:
        for key, item in test1_data.items():
            item["rtl"] = key
            f.write(json.dumps(item) + "\n")
    with open(os.path.join(path, "test2.jsonl"), "w") as f:
        for key, item in test2_data.items():
            item["rtl"] = key
            f.write(json.dumps(item) + "\n")
    with open(os.path.join(path, "test3.jsonl"), "w") as f:
        for key, item in test3_data.items():
            item["rtl"] = key
            f.write(json.dumps(item) + "\n")

def dc_dataset():
    base_dir = "dc_dataset"
    # original data syn_data.zip
    # extract it will have rtl0.json, rtl1.json, rtl2.json, rtl3.json
    # os.makedirs(base_dir, exist_ok=True)
    all_data = []
    for i in range(4):
        file_path = os.path.join(base_dir, f"rtl{i}.json")
        with open(file_path, "r") as f:
            data = json.load(f)
        for design, v in data.items():
            rtl = v["RTL"]
            syn = v["Syn"]
            for result in syn:
                all_data.append(
                    {
                        "design_name": design,
                        "rtl": rtl,
                        "area": result["Area"],
                        "delay": result["Delay"],
                        "power": result["StaticPower"],
                        "tns": result["TNS"],
                        "wns": result["WNS"],
                        "SDC": result["SDC"],
                        "clock": result["Clock"],
                        "has_clock": result["Has Clock"],
                    }
                )
    train, valid = train_test_split(all_data, test_size=0.1, random_state=42)
    os.makedirs(os.path.join("data", base_dir), exist_ok=True)
    with open(os.path.join("data", base_dir, "train.jsonl"), "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join("data", base_dir, "valid.jsonl"), "w") as f:
        for item in valid:
            f.write(json.dumps(item) + "\n")
    split_by_length(train, os.path.join("data", base_dir))
    test_path = ["verilog_eval"]
    test_data = []
    for i in range(1, 4):
        test_i_data = []
        for test_d in test_path:
            file_path = os.path.join(base_dir, test_d, f"test{i}.json")
            with open(file_path, "r") as f:
                data = json.load(f)
            for design, v in data.items():
                rtl = v["RTL"]
                syn = v["Syn"]
                for result in syn:
                    flattened_datum = {
                        "design_name": design,
                        "rtl": rtl,
                        "area": result["Area"],
                        "delay": result["Delay"],
                        "power": result["StaticPower"],
                        "tns": result["TNS"],
                        "wns": result["WNS"],
                        "SDC": result["SDC"],
                        "clock": result["Clock"],
                        "has_clock": result["Has Clock"]
                    }
                    test_data.append(flattened_datum)
                    test_i_data.append(flattened_datum)
        with open(os.path.join("data", base_dir, f"test{i}.jsonl"), "w") as f:
            for item in test_i_data:
                f.write(json.dumps(item) + "\n")
    with open(os.path.join("data", base_dir, "test.jsonl"), "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    print(f"Dataset saved to {os.path.join('data', base_dir)}")

def dc_dataset_v2():
    base_dir = "dc_dataset"
    sub_path = ["design_0816", "design_0816_2"]
    test_path = ["verilog_eval"]
    all_data = []
    test_data = []
    # test1_data = []
    for design_path in sub_path:
        for i in range(4):
            file_path = os.path.join(base_dir, design_path, f"rtl{i}.json")
            with open(file_path, "r") as f:
                data = json.load(f)
            for design, v in data.items():
                rtl = v["RTL"]
                syn = v["Syn"]
                for result in syn:
                    all_data.append(
                        {
                            "design_name": design,
                            "rtl": rtl,
                            "area": result["Area"],
                            "delay": result["Delay"],
                            "power": result["StaticPower"],
                            "tns": result["TNS"],
                            "wns": result["WNS"],
                            "SDC": result["SDC"],
                            "clock": result["Clock"],
                            "has_clock": result["Has Clock"]
                        }
                    )
    v2_base_dir = base_dir + "_v2"
    os.makedirs(os.path.join("data", v2_base_dir), exist_ok=True)
    for i in range(1, 4):
        test_i_data = []
        for test_d in test_path:
            file_path = os.path.join(base_dir, test_d, f"test{i}.json")
            with open(file_path, "r") as f:
                data = json.load(f)
            for design, v in data.items():
                rtl = v["RTL"]
                syn = v["Syn"]
                for result in syn:
                    flattened_datum = {
                        "design_name": design,
                        "rtl": rtl,
                        "area": result["Area"],
                        "delay": result["Delay"],
                        "power": result["StaticPower"],
                        "tns": result["TNS"],
                        "wns": result["WNS"],
                        "SDC": result["SDC"],
                        "clock": result["Clock"],
                        "has_clock": result["Has Clock"]
                    }
                    test_data.append(flattened_datum)
                    test_i_data.append(flattened_datum)
        with open(os.path.join("data", v2_base_dir, f"test{i}.jsonl"), "w") as f:
            for item in test_i_data:
                f.write(json.dumps(item) + "\n")

    train, valid = train_test_split(all_data, test_size=0.1, random_state=42)
    split_by_length(train, os.path.join("data", v2_base_dir))
    with open(os.path.join("data", v2_base_dir, "train.jsonl"), "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join("data", v2_base_dir, "valid.jsonl"), "w") as f:
        for item in valid:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join("data", v2_base_dir, "test.jsonl"), "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    print(f"Dataset saved to {os.path.join('data', v2_base_dir)}")

def length_testing_dataset():
    path = "data/metrex_regression"
    os.makedirs(path, exist_ok=True)
    dataset = load_dataset("scale-lab/MetRex", split="train")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B-Instruct")
    # split dataset according to encoded length and select 512 samples
    # all_length = []
    regex = re.compile(FLOAT_REGEX)
    dataset = dataset.map(
        lambda example: {
            "input_length": len(tokenizer(example["RTL"])["input_ids"]),
            "power_value": float(regex.findall(example["static_power"])[-1]),
            "area_value": float(regex.findall(example["area"])[-1]),
            "delay_value": float(regex.findall(example["delay"])[-1]),
        }
    )
    train_dataset, eval_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()
    # ipdb.set_trace()
    length = sorted(train_dataset["input_length"])
    
    l1 = length[int(len(length)*0.25)]
    l2 = length[int(len(length)*0.5)]
    l3 = length[int(len(length)*0.75)]
    # l4 = length[int(len(length)*0.8)]

    print(f"Length percentiles: 25%: {l1}, 50%: {l2}, 75%: {l3}")
    train_dataset: Dataset = train_dataset
    train_length_dataset = {
        "l1": train_dataset.filter(lambda x: x["input_length"] < l1).shuffle(seed=42).select(range(128)),
        "l2": train_dataset.filter(lambda x: l1 <= x["input_length"] and x["input_length"] < l2).shuffle(seed=42).select(range(128)),
        "l3": train_dataset.filter(lambda x: l2 <= x["input_length"] and x["input_length"] < l3).shuffle(seed=42).select(range(128)),
        "l4": train_dataset.filter(lambda x: l3 <= x["input_length"]).shuffle(seed=42).select(range(128)),
    }
    train_length_dataset = DatasetDict(train_length_dataset)
    train_length_dataset.save_to_disk(os.path.join(path, "length_split_train_512"))

def has_clk_signal_str(rtl: str):
    has_clk = False
    clk_name = None
    lines = rtl.split("\n")
    for line in lines:
        tokens = re.split('[ ,()_]', line.strip().strip(';').strip())
        if ('input' in tokens and ('clk' in tokens or 'CLK' in tokens or 'clock' in tokens)):
            has_clk = True
            tokens_2 = re.split('[ ,()]', line.strip().strip(';').strip())
            for token in tokens_2:
                if ('clk' in token or 'CLK' in token):
                    clk_name = token
            break

    return has_clk, clk_name

def get_sdc(design: str, target_clock_period: int = 20):
    # dataset = load_dataset("scale-lab/MetRex", split="train")
    # designs = dataset["RTL"]
    # target_clock_period = 20
    # for design in tqdm(designs):
    has_clk, clk_port = has_clk_signal_str(design)

    if has_clk: 
        synth_sdc_script = SDC_CONSTRAINTS_CLOCK.format(
            clock_name="clk", 
            clock_port=clk_port, 
            clock_period=target_clock_period, 
            max_fanout=10, 
        )
        delay_sdc = SDC_CONSTRAINTS_CLOCK.format(
            clock_name="clk",
            clock_port=clk_port,
            clock_period=0,
            max_fanout=10
        )
        power_sdc = SDC_CONSTRAINTS_CLOCK.format(
            has_clk, 
            clock_name="myClock", 
            clock_port=clk_port, 
            clock_period=1, 
            max_fanout=10, 
            driving_cell="sky130_fd_sc_hd__inv_2", 
            driving_cell_pin="Y"
        )
    else:
        synth_sdc_script = SDC_CONSTRAINTS_WO_CLOCK.format(
            clock_name="clk", 
            clock_period=target_clock_period, 
            max_fanout=10, 
        )
        delay_sdc = SDC_CONSTRAINTS_WO_CLOCK.format(
            clock_name="clk",
            clock_period=0,
            max_fanout=10
        )
        power_sdc = SDC_CONSTRAINTS_WO_CLOCK.format(
            has_clk, 
            clock_name="myClock", 
            clock_period=1, 
            max_fanout=10, 
        )

    synth_sdc_script += SDC_DRIVER.format(
        driving_cell="sky130_fd_sc_hd__inv_2",
        driving_cell_pin="Y"
    )
    
    delay_sdc += SDC_DRIVER.format(
        driving_cell="sky130_fd_sc_hd__inv_2",
        driving_cell_pin="Y"
    )
    
    power_sdc += SDC_DRIVER.format(
        driving_cell="sky130_fd_sc_hd__inv_2",
        driving_cell_pin="Y"
    )
    return synth_sdc_script, delay_sdc, power_sdc

KEY_MAP_V3 = {
    "area": "area",
    "Total Area": "area_value",
    "Delay": "delay_value",
    "delay": "delay",
    "Total Power": "total_power_value",
    "Static Power": "static_power_value",
    "static_power": "static_power",
    "switching_power": "switching_power",
    "TNS": "tns",
    "WNS": "wns",
    "SDC": "SDC",
    "Clock": "clock",
    "Has Clock": "has_clock",
    "synth": "synth",
    "synthesis_time": "synthesis_time",
    # "netlist": "netlist",
}

def extract_data_v3(verilog_data):
    rtl = verilog_data["RTL"]
    syn = verilog_data["Syn"]
    all_data = []
    for result in syn:
        static_power = float(re.findall(FLOAT_REGEX, result["static_power"])[-1])
        switching_power = float(re.findall(FLOAT_REGEX, result["switching_power"])[-1])
        new_result = {}
        for k, v in result.items():
            if k in KEY_MAP_V3:
                new_result[KEY_MAP_V3[k]] = v
        new_result["design_name"] = verilog_data["Name"]
        new_result["rtl"] = rtl
        new_result["static_power_value"] = static_power
        new_result["switching_power_value"] = switching_power
        all_data.append(new_result)
        # {
        #     "design_name": verilog_data["Name"],
        #     "rtl": rtl,
        #     "area": result["area"],
        #     "area_value": result["Total Area"],
        #     "delay_value": result["Delay"],
        #     "delay": result["delay"],
        #     "switching_power_value": result["Total Power"],
        #     "static_power_value": static_power,
        #     "static_power": result["static_power"],
        #     "switching_power": result["switching_power"],
        #     "tns": result["TNS"],
        #     "wns": result["WNS"],
        #     "SDC": result["SDC"],
        #     "clock": result["Clock"],
        #     "has_clock": result["Has Clock"],
        #     "synth": result["synth"],
        #     "synthesis_time": result["synthesis_time"],
        # }
    return all_data

def extract_data_v5(verilog_data):
    all_data = extract_data_v3(verilog_data)
    max_clock = max(all_data, key=lambda x: x["clock"] if x["has_clock"] else 0)
    return [max_clock]

def dc_dataset_v3(dataset_path: str = "dc_dataset/dc_v3", sub_path: list = ["0", "1", "2"], test_path: list = ["verilogeval"], output_path: str = "data/dc_dataset_v3"):
    np.random.seed(42)
    random.seed(42)
    all_data = []
    test_data = []
    for design_path in sub_path:
        for i in range(4):
            file_path = os.path.join(dataset_path, design_path, f"rtl{i}.json")
            if os.path.exists(file_path) is False:
                continue
            with open(file_path, "r") as f:
                data = json.load(f)
            for v in data:
                all_data.extend(extract_data_v3(v))

    os.makedirs(os.path.join("data", output_path), exist_ok=True)
    for i in range(1, 4):
        test_i_data = []
        for test_d in test_path:
            file_path = os.path.join(dataset_path, test_d, f"test{i}.json")
            with open(file_path, "r") as f:
                data = json.load(f)
            for v in data:
                extracted_data = extract_data_v3(v)
                test_data.extend(extracted_data)
                test_i_data.extend(extracted_data)
        with open(os.path.join("data", output_path, f"test{i}.jsonl"), "w") as f:
            for item in test_i_data:
                f.write(json.dumps(item) + "\n")

    train, valid = train_test_split(all_data, test_size=0.1, random_state=42)
    split_by_length(train, valid, os.path.join("data", output_path))
    with open(os.path.join("data", output_path, "train.jsonl"), "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join("data", output_path, "valid.jsonl"), "w") as f:
        for item in valid:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join("data", output_path, "test.jsonl"), "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    print(f"Dataset saved to {os.path.join('data', output_path)}")

def dc_dataset_v4(
    dataset_path: list = ["dc_dataset/dc_v4", "dc_dataset/pyranet"],
    sub_path: list = ["0", "1", "2"],
    output_path: str = "data/dc_dataset_v4",
    use_tokenized_length_filter: bool = False,
    tokenized_length_limit: int = 2048,
):
    np.random.seed(42)
    random.seed(42)
    all_data = []
    test_data = []
    for data_path in dataset_path:
        if sub_path is None:
            for i in range(4):
                file_path = os.path.join(data_path, f"rtl{i}.json")
                if os.path.exists(file_path) is False:
                    continue
                with open(file_path, "r") as f:
                    data = json.load(f)
                all_data.extend(data)
            file_path = os.path.join(data_path, "rtl.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                all_data.extend(data)
        else:
            for design_path in sub_path:
                for i in range(4):
                    file_path = os.path.join(data_path, design_path, f"rtl{i}.json")
                    if os.path.exists(file_path) is False:
                        continue
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    all_data.extend(data)

    os.makedirs(os.path.join("data", output_path), exist_ok=True)

    # Deduplicate on RTL source (whitespace-normalised) before any splitting.
    # Each element of all_data is a raw design dict with an "RTL" key.
    seen_rtl: set[str] = set()
    deduped_data = []
    for item in all_data:
        # Normalise whitespace so minor formatting differences don't survive.
        key = " ".join(item["RTL"].split())
        if key not in seen_rtl:
            seen_rtl.add(key)
            deduped_data.append(item)
    print(f"Deduplication: {len(all_data)} raw designs -> {len(deduped_data)} unique designs.")
    all_data = deduped_data

    if use_tokenized_length_filter:
        tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-s-s-prefixlm")
        filtered_data = []
        for item in tqdm(all_data, desc="Filtering by tokenized RTL length"):
            length = len(tokenizer(item["RTL"])["input_ids"])
            if length <= tokenized_length_limit:
                filtered_data.append(item)
        print(
            f"Tokenized length filter: {len(all_data)} unique designs -> "
            f"{len(filtered_data)} designs (limit={tokenized_length_limit})."
        )
        all_data = filtered_data

    train, valid = train_test_split(all_data, test_size=0.1, random_state=42)
    extracted_train = []
    extracted_valid = []
    for v in train:
        extracted_train.extend(extract_data_v3(v))
    for v in valid:
        extracted_valid.extend(extract_data_v3(v))
    train, valid = extracted_train, extracted_valid
    with open(os.path.join("data", output_path, "train.jsonl"), "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join("data", output_path, "valid.jsonl"), "w") as f:
        for item in valid:
            f.write(json.dumps(item) + "\n")
    print(f"Dataset saved to {os.path.join('data', output_path)}")


def verilogeval_dataset(dataset_path: str = "dc_dataset/verilogeval", output_path: str = "dc_dataset_v4", extract_func=extract_data_v3, dataset_name="test"):
    np.random.seed(42)
    random.seed(42)
    test_data = []
    for i in range(1, 4):
        file_path = os.path.join(dataset_path, f"test{i}.json")
        with open(file_path, "r") as f:
            data = json.load(f)
        for v in data:
            extracted_data = extract_func(v)
            test_data.extend(extracted_data)

    os.makedirs(os.path.join("data", output_path), exist_ok=True)
    with open(os.path.join("data", output_path, f"{dataset_name}.jsonl"), "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    print(f"Dataset saved to {os.path.join('data', output_path, f'{dataset_name}.jsonl')}")

def realbench_dataset(dataset_path: str = "dc_dataset/RealBench", output_path: str = "dc_dataset_v4", extract_func=extract_data_v3):
    np.random.seed(42)
    random.seed(42)
    test_data = []
    test_name = ["aes/aes", "e203_hbirdv2/e203_main", "e203_general/e203_submodule", "sdc/sdc"]
    for test in test_name:
        file_path = os.path.join(dataset_path, f"{test}.json")
        if os.path.exists(file_path) is False:
            continue
        with open(file_path, "r") as f:
            data = json.load(f)
        for v in data:
            extracted_data = extract_func(v)
            test_data.extend(extracted_data)

    os.makedirs(os.path.join("data", output_path), exist_ok=True)
    with open(os.path.join("data", output_path, "test_realbench.jsonl"), "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    print(f"Dataset saved to {os.path.join('data', output_path, 'test_realbench.jsonl')}")

def opt_dataset(dataset_path: str = "data", dataset_name: str = "pluto", output_path=None, extract_func=extract_data_v3):
    np.random.seed(42)
    random.seed(42)
    test_data = []
    test_name = [f"test_{dataset_name}"]
    # dataset_path = os.path.join(dataset_path, dataset_name)
    for test in test_name:
        file_path = os.path.join("examples", f"{test}.json")
        if os.path.exists(file_path) is False:
            continue
        with open(file_path, "r") as f: 
            data = json.load(f)
        for v in data:
            extracted_data = extract_func(v)
            test_data.extend(extracted_data)

    if output_path is None:
        output_path = dataset_name
    os.makedirs(os.path.join("data", output_path), exist_ok=True)
    with open(os.path.join("data", output_path, f"test_{dataset_name}.jsonl"), "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    print(f"Dataset saved to {os.path.join('data', output_path, f'test_{dataset_name}.jsonl')}")

def dc_dataset_v5(dataset_path: str = "dc_dataset/dc_v4", output_path: str = "data/dc_dataset_v4"):
    np.random.seed(42)
    random.seed(42)
    all_data = []
    for i in range(4):
        file_path = os.path.join(dataset_path, f"rtl{i}.json")
        if os.path.exists(file_path) is False:
            continue
        with open(file_path, "r") as f:
            data = json.load(f)
        all_data.extend(data)

    train, valid = train_test_split(all_data, test_size=0.1, random_state=42)
    extracted_train = []
    extracted_valid = []
    for v in train:
        all_data = extract_data_v3(v)
        max_clock_data = max(all_data, key=lambda x: x["clock"] if x["has_clock"] else 0)
        extracted_train.append(max_clock_data)
    for v in valid:
        all_data = extract_data_v3(v)
        max_clock_data = max(all_data, key=lambda x: x["clock"] if x["has_clock"] else 0)
        extracted_valid.append(max_clock_data)
    train, valid = extracted_train, extracted_valid
    # split_by_length(train, valid, os.path.join("data", output_path))
    os.makedirs(os.path.join("data", output_path), exist_ok=True)
    with open(os.path.join("data", output_path, "train.jsonl"), "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join("data", output_path, "valid.jsonl"), "w") as f:
        for item in valid:
            f.write(json.dumps(item) + "\n")
    print(f"Dataset saved to {os.path.join('data', output_path)}")


def icrtl_dataset(dataset_path: str = "dc_dataset/icrtl", dataset_name: str = "icrtl", output_path: str = None, extract_func=extract_data_v3):
    np.random.seed(42)
    random.seed(42)
    test_data = []
    test_name = ["dataset"]
    # dataset_path = os.path.join(dataset_path, dataset_name)
    for test in test_name:
        file_path = os.path.join(dataset_path, f"{test}.json")
        if os.path.exists(file_path) is False:
            continue
        with open(file_path, "r") as f: 
            data = json.load(f)
        for v in data:
            extracted_data = extract_func(v)
            test_data.extend(extracted_data)

    if output_path is None:
        output_path = dataset_name
    os.makedirs(os.path.join("data", output_path), exist_ok=True)
    with open(os.path.join("data", output_path, f"test_{dataset_name}.jsonl"), "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    print(f"Dataset saved to {os.path.join('data', output_path, f'test_{dataset_name}.jsonl')}")

def fix_verilogos_dataset(output_path: str = "dataset_v6.1.3", extract_func=extract_data_v3):
    np.random.seed(42)
    random.seed(42)
    
    train_rtls = set()
    train_file = os.path.join("data", output_path, "train.jsonl")
    if os.path.exists(train_file):
        with open(train_file, "r") as f:
            for line in f:
                item = json.loads(line)
                train_rtls.add(" ".join(item["rtl"].split()))

    valid_rtls = set()
    valid_file = os.path.join("data", output_path, "valid.jsonl")
    if os.path.exists(valid_file):
        with open(valid_file, "r") as f:
            for line in f:
                item = json.loads(line)
                valid_rtls.add(" ".join(item["rtl"].split()))

    new_train_data = []
    new_valid_data = []

    for test_name in ["verilogos_train", "verilogos_valid"]:
        file_path = os.path.join("examples", f"test_{test_name}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
            for v in data:
                original_code = v.get("original", "")
                if not original_code:
                    continue
                original_code_norm = " ".join(original_code.split())
                extracted = extract_func(v)
                
                if original_code_norm in train_rtls:
                    new_train_data.extend(extracted)
                elif original_code_norm in valid_rtls:
                    new_valid_data.extend(extracted)

    os.makedirs(os.path.join("data", output_path), exist_ok=True)
    with open(os.path.join("data", output_path, "test_verilogos_train.jsonl"), "w") as f:
        for item in new_train_data:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join("data", output_path, "test_verilogos_valid.jsonl"), "w") as f:
        for item in new_valid_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Dataset saved to {os.path.join('data', output_path, 'test_verilogos_train.jsonl')} ({len(new_train_data)} inputs)")
    print(f"Dataset saved to {os.path.join('data', output_path, 'test_verilogos_valid.jsonl')} ({len(new_valid_data)} inputs)")
def concat_jsonl(path1: str, path2: str, output_path:str):
    # Concatenate two jsonl files
    with open(path1, "r") as f1, open(path2, "r") as f2, open(output_path, "w") as f:
        for line in f1:
            f.write(line)
        for line in f2:
            f.write(line)

if __name__ == "__main__":
    # metrex_dataset_regression()
    # dc_dataset()
    # dc_dataset_v2()
    # dc_dataset_v3()
    # sub_path = [str(i) for i in range(9)]
    # dc_dataset_v3(dataset_path="dc_dataset/dc_v3.1", sub_path=sub_path, test_path=["verilogeval"], output_path="dc_dataset_v3.1")

    # dc_dataset_v4(dataset_path="dc_dataset/dc_v4", sub_path=sub_path, test_path=["verilogeval"], output_path="dc_dataset_v4")
    # sub_path = [str(i) for i in range(10)]
    # dc_dataset_v4(dataset_path="dc_dataset/dc_v4.1", sub_path=sub_path, test_path=["verilogeval"], output_path="dc_dataset_v4.1")

    # dc_dataset_v4(dataset_path="dc_dataset/design_ver1", sub_path=None, test_path=["verilogeval"], output_path="dc_dataset_v4.2")
    # dc_dataset_v4(dataset_path="dc_dataset/design_ver2", sub_path=None, test_path=["verilogeval"], output_path="dc_dataset_v4.3")

    # realbench_dataset(dataset_path="dc_dataset/RealBench_ver2", output_path="dc_dataset_v4.3")
    # verilogeval_dataset(dataset_path="dc_dataset/verilogeval_ver2", output_path="dc_dataset_v4.3")

    # dc_dataset_v4(dataset_path="data/dc_dataset_v4.x", sub_path=None, test_path=["verilogeval"], output_path="dc_dataset_v4.3.5")

    # dc_dataset_v4(dataset_path=["dc_dataset/design_ver3"], sub_path=None, test_path=["verilogeval"], output_path="dataset_v6.0.0")
    # realbench_dataset(dataset_path="dc_dataset/RealBench_ver4", output_path="dataset_v6.0.0")
    # verilogeval_dataset(dataset_path="dc_dataset/verilogeval_ver3", output_path="dataset_v6.0.0")

    # dc_dataset_v4(
    #     dataset_path=[
    #         # "dc_dataset/pyranet_ver2",
    #         # "dc_dataset/pyranet_ver2.1",
    #         # "dc_dataset/pyranet_ver2.2",
    #         "dc_dataset/design_ver4",
    #         "dc_dataset/design_ver4.1",
    #         # "dc_dataset/verilogos",
    #     ],
    #     sub_path=None,
    #     output_path="dataset_v6.2.0",
    #     use_tokenized_length_filter=True,
    #     tokenized_length_limit=2048,
    # )
    # realbench_dataset(dataset_path="dc_dataset/RealBench_ver6", output_path="dataset_v6.1.5")
    # verilogeval_dataset(dataset_path="dc_dataset/verilogeval_ver4/verilogeval", output_path="dataset_v6.1.8", dataset_name="test")
    # verilogeval_dataset(dataset_path="dc_dataset/verilogeval_ver4/verilogeval_2", output_path="dataset_v6.0.4", dataset_name="test_ood_clk")

    # fix_verilogos_dataset(output_path="dataset_v6.1.5")
    opt_dataset(dataset_path="data", dataset_name="aug_200", output_path="dataset_v6.2.0")
    # opt_dataset(dataset_path="data", dataset_name="rtl_opt", output_path="dataset_v6.1.5")
    # opt_dataset(dataset_path="data", dataset_name="ds_verilog_eval", output_path="dataset_v6.1.5")
    # icrtl_dataset(dataset_path="dc_dataset/icrtl", dataset_name="icrtl", output_path="dataset_v6.1.5")
    # icrtl_dataset(dataset_path="dc_dataset/icrtl_resyn", dataset_name="icrtl_resyn", output_path="dataset_v6.1.5")
    # dc_dataset_v4(dataset_path=["dc_dataset/pyranet_ver2", "dc_dataset/pyranet_ver2.1", "dc_dataset/design_ver4", "dc_dataset/design_ver4.1"], sub_path=None, output_path="dataset_v6.1.0")
    # realbench_dataset(dataset_path="dc_dataset/RealBench_ver6", output_path="dataset_v6.1.0")
    # verilogeval_dataset(dataset_path="dc_dataset/verilogeval_ver4/verilogeval", output_path="dataset_v6.1.0", dataset_name="test")
    # verilogeval_dataset(dataset_path="dc_dataset/verilogeval_ver4/verilogeval_2", output_path="dataset_v6.0.4", dataset_name="test_ood_clk")

    # opt_dataset(dataset_path="data", dataset_name="pluto", output_path="dataset_v6.1.0")
    # opt_dataset(dataset_path="data", dataset_name="rtl_opt", output_path="dataset_v6.1.0")

    # opt_dataset(dataset_path="data", dataset_name="ds_verilog_eval", output_path="dataset_v6.1.3")
    # opt_dataset(dataset_path="data", dataset_name="ppa_feedback", output_path="dataset_v6.1.6")
    # opt_dataset(dataset_path="data", dataset_name="no_feedback", output_path="dataset_v6.1.6")
    # opt_dataset(dataset_path="data", dataset_name="deepseek_ref", output_path="dataset_v6.1.6")
    # dc_dataset_v5(dataset_path="dc_dataset/design_ver1",output_path="dc_dataset_v5.0")
    # realbench_dataset(dataset_path="dc_dataset/RealBench_ver2", output_path="dc_dataset_v5.1", extract_func=extract_data_v5)
    # verilogeval_dataset(dataset_path="dc_dataset/verilogeval_ver2", output_path="dc_dataset_v5.1", extract_func=extract_data_v5)
    # dc_dataset_v5(dataset_path="dc_dataset/design_ver2",output_path="dc_dataset_v5.1")
    # length_testing_dataset()

    # concat_jsonl("data/dataset_v6.1.5/train.jsonl", "data/dataset_v6.1.5/test_ds_verilog_eval.jsonl", "data/dataset_v6.1.7/train.jsonl")
    pass
