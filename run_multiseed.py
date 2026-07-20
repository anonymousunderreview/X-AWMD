#!/usr/bin/env python3
"""
Multi-seed runner for the proposed X-AWMD model.

For each seed, this script trains the default STFT+phase+pair-ranking model,
selects a threshold on validation using Youden's J, and evaluates Test0/Test1/
Test2 with that fixed validation threshold.

Example:
    python run_multiseed.py \
      --manifest /path/to/dataset_manifest.csv \
      --out_dir experiments/multiseed_xawmd \
      --seeds 42 2024
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from train import GeneralizedXAWMD, ManifestEvalDataset, collate_eval, evaluate


METRICS = ("AUROC", "F1", "Recall")
DEFAULT_SPLITS = ("test0_in", "test1_in", "test2_in")


def compute_metrics(labels, scores, threshold=None):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if len(np.unique(labels)) < 2:
        return None
    auroc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden_thr = float(thresholds[int(np.argmax(tpr - fpr))])
    threshold = youden_thr if threshold is None else float(threshold)
    pred = (scores >= threshold).astype(int)
    return {
        "AUROC": auroc,
        "F1": float(f1_score(labels, pred, pos_label=1, zero_division=0)),
        "Recall": float(pred[labels == 1].mean()) if np.any(labels == 1) else float("nan"),
        "youden_thr": youden_thr,
        "thr_used": threshold,
    }


def eval_split(model, cfg, manifest, split, device, batch_size, threshold=None):
    dataset = ManifestEvalDataset(manifest, split, cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate_eval,
        pin_memory=device.type == "cuda",
    )
    labels, scores, _ = evaluate(model, loader, device)
    return compute_metrics(labels, scores, threshold)


def train_seed(args, seed):
    seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    ckpt_path = os.path.join(seed_dir, "model.pth")
    log_path = os.path.join(seed_dir, "train.log")

    if args.skip_train:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint for --skip_train: {ckpt_path}")
        return ckpt_path

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"),
        "--manifest",
        args.manifest,
        "--save_path",
        ckpt_path,
        "--log_file",
        log_path,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--features",
        "stft_mag_phase",
        "--pair_weight",
        str(args.pair_weight),
        "--pair_margin",
        str(args.pair_margin),
    ]
    print(f"[seed={seed}] training -> {ckpt_path}", flush=True)
    subprocess.run(cmd, check=True)
    return ckpt_path


def eval_seed(args, ckpt_path):
    device = torch.device(args.device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint["cfg"]
    cfg["manifest"] = args.manifest
    cfg["device"] = args.device
    model = GeneralizedXAWMD(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    results = {}
    val = eval_split(
        model,
        cfg,
        args.manifest,
        args.validation_split,
        device,
        args.eval_batch_size,
    )
    if val is None:
        raise RuntimeError(f"Validation split is single-class: {args.validation_split}")
    results[args.validation_split] = val
    val_thr = val["youden_thr"]
    results["_val_thr"] = val_thr

    for split in args.test_splits:
        try:
            res = eval_split(
                model,
                cfg,
                args.manifest,
                split,
                device,
                args.eval_batch_size,
                threshold=val_thr,
            )
        except RuntimeError as exc:
            print(f"[eval] skip {split}: {exc}", flush=True)
            continue
        if res is not None:
            results[split] = res
    return results


def aggregate(per_seed):
    output = {}
    splits = sorted({split for result in per_seed for split in result if not split.startswith("_")})
    for split in splits:
        output[split] = {}
        for metric in METRICS + ("youden_thr",):
            values = [result[split][metric] for result in per_seed if split in result and metric in result[split]]
            if values:
                output[split][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "per_seed": [float(value) for value in values],
                }
    return output


def print_summary(aggregated):
    print("\nX-AWMD multi-seed summary")
    print("split        AUROC              F1                 Recall")
    print("-" * 68)
    for split, metrics in aggregated.items():
        cells = []
        for metric in METRICS:
            value = metrics.get(metric)
            cells.append("N/A" if value is None else f"{value['mean']:.4f} +/- {value['std']:.4f}")
        print(f"{split:<12} {cells[0]:<18} {cells[1]:<18} {cells[2]:<18}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed X-AWMD experiments")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", default="experiments/multiseed_xawmd")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validation_split", default="validation")
    parser.add_argument("--test_splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--pair_weight", type=float, default=0.5)
    parser.add_argument("--pair_margin", type=float, default=0.25)
    parser.add_argument("--skip_train", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpts = [train_seed(args, seed) for seed in args.seeds]
    per_seed = []
    for seed, ckpt_path in zip(args.seeds, ckpts):
        print(f"[seed={seed}] evaluating {ckpt_path}", flush=True)
        result = eval_seed(args, ckpt_path)
        per_seed.append(result)
        seed_json = os.path.join(args.out_dir, f"seed_{seed}", "results.json")
        with open(seed_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    aggregated = aggregate(per_seed)
    out_json = os.path.join(args.out_dir, "aggregated.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump({"seeds": args.seeds, "aggregated": aggregated}, handle, indent=2)
    print_summary(aggregated)
    print(f"\nSaved aggregate results to {out_json}")


if __name__ == "__main__":
    main()
