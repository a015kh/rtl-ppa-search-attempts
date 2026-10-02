"""Shared evaluation implementation extracted from the verified v6.3 trainer."""
import logging
import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from revolution.rank_metrics import metric_rank_fields
logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate_model(model, dataloader, split_name: str, *, tokenizer, device, metrics, num_samples, output_dir):
    model.eval()
    all_relative_maes: list[float] = []
    per_target_relative_maes: list[list[float]] = []
    max_relative_maes: list[float] = []
    gt_cols: list[list[float]] = []
    pred_cols: list[list[float]] = []
    design_names: list[str] = []
    family_ids: list[str] = []
    clocks: list[float] = []
    sample_offset = 0
    total_samples = 0
    total_decode_fail = 0

    for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
        batch_size = batch["input_ids"].shape[0]
        dataset_samples = getattr(dataloader.dataset, "samples", [])
        batch_samples = dataset_samples[sample_offset:sample_offset + batch_size]
        batch_design_names = [
            sample.get("design_name", "") for sample in batch_samples
        ]
        batch_family_ids = [
            sample.get("family_id", "") for sample in batch_samples
        ]
        batch_clocks = [
            sample.get("clock", "") for sample in batch_samples
        ]
        if len(batch_design_names) < batch_size:
            batch_design_names.extend([""] * (batch_size - len(batch_design_names)))
        if len(batch_family_ids) < batch_size:
            batch_family_ids.extend([""] * (batch_size - len(batch_family_ids)))
        if len(batch_clocks) < batch_size:
            batch_clocks.extend([""] * (batch_size - len(batch_clocks)))
        sample_offset += batch_size

        batch = {k: v.to(device) for k, v in batch.items()}
        output_ids = (
            model.decode(batch, num_samples=num_samples)
            if num_samples > 1
            else model.decode(batch).squeeze()
        )
        if num_samples == 1 and output_ids.dim() == 1:
            output_ids = output_ids.unsqueeze(0)

        labels = batch["labels"].cpu()
        labels[labels == -100] = tokenizer.pad_token_id
        ground_truth_strs = tokenizer.batch_decode(
            labels.tolist(), skip_special_tokens=True
        )
        output_ids_list = output_ids.cpu().tolist()

        for sample_output_ids, gt_str, design_name, family_id, clock in zip(
            output_ids_list,
            ground_truth_strs,
            batch_design_names,
            batch_family_ids,
            batch_clocks,
        ):
            try:
                gt_floats = [float(v.strip()) for v in gt_str.strip().split("\n") if v.strip()]
            except ValueError:
                total_decode_fail += 1
                continue
            if len(gt_floats) == 0:
                total_decode_fail += 1
                continue

            try:
                if num_samples > 1:
                    out_str = tokenizer.batch_decode(sample_output_ids, skip_special_tokens=True)
                    decoded_per_draw: list[list[float]] = []
                    for draw in out_str:
                        try:
                            floats = [float(v.strip()) for v in draw.strip().split("\n") if v.strip()]
                            if len(floats) == len(gt_floats):
                                decoded_per_draw.append(floats)
                        except ValueError:
                            continue
                    if len(decoded_per_draw) == 0:
                        total_decode_fail += 1
                        continue
                    pred_floats = [
                        float(np.median([draw[i] for draw in decoded_per_draw]))
                        for i in range(len(gt_floats))
                    ]
                else:
                    out_str = tokenizer.decode(sample_output_ids, skip_special_tokens=True)
                    pred_floats = [float(v.strip()) for v in out_str.strip().split("\n") if v.strip()]
                    if len(pred_floats) != len(gt_floats):
                        total_decode_fail += 1
                        continue
            except Exception:
                total_decode_fail += 1
                continue

            # Initialise column lists on first valid sample
            if not gt_cols:
                gt_cols   = [[] for _ in gt_floats]
                pred_cols = [[] for _ in pred_floats]

            sample_rel_maes = []
            for i, (pred, gt) in enumerate(zip(pred_floats, gt_floats)):
                rel_mae = abs(pred - gt) / (abs(gt) + 1e-10)
                all_relative_maes.append(rel_mae)
                sample_rel_maes.append(rel_mae)
                if i >= len(per_target_relative_maes):
                    per_target_relative_maes.append([])
                per_target_relative_maes[i].append(rel_mae)
                gt_cols[i].append(gt)
                pred_cols[i].append(pred)

            if sample_rel_maes:
                max_relative_maes.append(max(sample_rel_maes))

            design_names.append(design_name)
            family_ids.append(family_id)
            clocks.append(clock)
            total_samples += 1

    # Save per-sample CSV with named metric columns
    if gt_cols:
        csv_data = {
            "design_name": design_names,
            "family_id": family_ids,
            "clock": clocks,
        }
        for i, (gts, preds) in enumerate(zip(gt_cols, pred_cols)):
            name = metrics[i] if i < len(metrics) else f"metric{i}"
            csv_data[f"gt_{name}"]   = gts
            csv_data[f"pred_{name}"] = preds
        csv_df = pd.DataFrame(csv_data)
        csv_path = os.path.join(
            output_dir, f"predictions_ground_truth_{split_name}.csv"
        )
        csv_df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")

    if total_samples == 0:
        return {"avg_acc_at_10_percent": 0, "avg_relative_mae": 1.0,
                "decode_fail": total_decode_fail,
                **metric_rank_fields(metrics, gt_cols, pred_cols)}

    arr_all = np.array(all_relative_maes)
    arr_max = np.array(max_relative_maes)
    result = {
        "avg_relative_mae":      float(arr_all.mean()),
        "avg_acc_at_5_percent":  round(float(np.mean(arr_all < 0.05)) * 100, 2),
        "avg_acc_at_10_percent": round(float(np.mean(arr_all < 0.1))  * 100, 2),
        "avg_acc_at_20_percent": round(float(np.mean(arr_all < 0.2))  * 100, 2),
        "all_metrics_acc_at_5_percent":  round(float(np.mean(arr_max < 0.05)) * 100, 2),
        "all_metrics_acc_at_10_percent": round(float(np.mean(arr_max < 0.1))  * 100, 2),
        "all_metrics_acc_at_20_percent": round(float(np.mean(arr_max < 0.2))  * 100, 2),
        "total_samples":         total_samples,
        "decode_fail":           total_decode_fail,
    }
    for i, target_maes in enumerate(per_target_relative_maes):
        arr  = np.array(target_maes)
        name = metrics[i] if i < len(metrics) else f"metric{i}"
        result[f"{name}_acc_at_5_percent"]  = round(float(np.mean(arr < 0.05)) * 100, 2)
        result[f"{name}_acc_at_10_percent"] = round(float(np.mean(arr < 0.1))  * 100, 2)
        result[f"{name}_acc_at_20_percent"] = round(float(np.mean(arr < 0.2))  * 100, 2)
    result.update(metric_rank_fields(metrics, gt_cols, pred_cols))
    return result

