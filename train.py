#!/usr/bin/env python3
"""
X-AWMD: STFT log-magnitude + differential-phase audio watermark detector.

This is the main implementation used for the paper:
"X-AWMD: A Generalizable Framework for Black-Box Audio Watermark Detection
under Method and Domain Shift".

The default configuration corresponds to the proposed model, i.e. ablation
row C in the paper:

  STFT log-magnitude + cos/sin differential phase + 2D CNN + BCE + pair ranking

Supported feature configurations:
  xlsr
  waveform
  stft_mag
  stft_mag_phase
  xlsr+waveform
  xlsr+stft_mag
  xlsr+stft_mag_phase

Training additions are independently configurable:
  pair ranking loss
  method-centroid invariance loss
  method-balanced sampling
  GroupDRO over watermark methods

The training dataset is represented as clean/watermarked pairs sharing the same
orig_path. Validation and test data are evaluated row-by-row. The detector is
black-box with respect to the embedder: it does not use the watermarking
algorithm, secret key, decoder, or clean reference at inference time.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


CFG = {
    "manifest": None,
    "train_split": "train",
    "val_split": "validation",
    "sample_rate": 16000,
    "duration": 3.0,
    "batch_size": 3,  # number of pairs; effective waveform batch is 6
    "num_workers": 2,
    "num_epochs": 15,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "features": "stft_mag_phase",
    "embedding_dim": 256,
    "fusion_hidden": 256,
    "dropout": 0.1,
    # XLS-R branch
    "xlsr_model": "facebook/wav2vec2-xls-r-300m",
    "layer_idx": 9,
    "xlsr_conv_channels": 256,
    "xlsr_conv_layers": 3,
    "xlsr_lstm_hidden": 256,
    "xlsr_lstm_layers": 2,
    "attn_dim": 128,
    # raw-waveform branch
    "wave_channels": "64,128,192,256",
    "wave_kernels": "15,11,7,5",
    "wave_strides": "5,4,4,2",
    # STFT branch
    "n_fft": 512,
    "hop_length": 160,
    "win_length": 400,
    "stft_channels": "32,64,128",
    # objectives
    "pair_weight": 0.5,
    "pair_margin": 0.25,
    "invariant_weight": 0.0,
    "method_balanced": False,
    "groupdro": False,
    "groupdro_eta": 0.05,
    # runtime/reporting
    "save_path": "./checkpoints/xawmd_stft_phase_pair.pth",
    "log_file": None,
    "log_every": 50,
    "early_stop_patience": 3,
    "thr_strategy": "tpr_fpr",
    "fixed_thr": None,
    "eval_only": False,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def canonical_method(value: str) -> str:
    value = (value or "").strip()
    return value if value else "clean"


def load_audio(path: str, cfg: dict) -> np.ndarray:
    wav, _ = librosa.load(path, sr=int(cfg["sample_rate"]), mono=True)
    target = int(float(cfg["duration"]) * int(cfg["sample_rate"]))
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))
    else:
        wav = wav[:target]
    return wav.astype(np.float32)


@dataclass
class PairItem:
    clean_path: str
    wm_path: str
    method: str
    orig_path: str


class PairedTrainDataset(Dataset):
    """One item is an aligned clean/watermarked pair."""

    def __init__(self, manifest: str, split: str, cfg: dict):
        rows_by_orig: Dict[str, dict] = defaultdict(lambda: {"clean": None, "wm": []})
        with open(manifest, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != split:
                    continue
                path = row.get("derived_path")
                if not path or not os.path.isfile(path):
                    continue
                orig = row.get("orig_path") or path
                if int(row.get("is_watermarked", 0)):
                    rows_by_orig[orig]["wm"].append(
                        (path, canonical_method(row.get("watermark_method")))
                    )
                else:
                    rows_by_orig[orig]["clean"] = path

        self.items: List[PairItem] = []
        missing_clean = 0
        for orig, group in rows_by_orig.items():
            clean = group["clean"]
            if not clean and os.path.isfile(orig):
                clean = orig
            if not clean:
                missing_clean += len(group["wm"])
                continue
            for wm_path, method in group["wm"]:
                self.items.append(PairItem(clean, wm_path, method, orig))
        if not self.items:
            raise RuntimeError(f"No valid clean/watermarked pairs found in split={split}")
        self.cfg = cfg
        self.missing_clean = missing_clean
        self.methods = sorted({item.method for item in self.items})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        return (
            load_audio(item.clean_path, self.cfg),
            load_audio(item.wm_path, self.cfg),
            item.method,
            item.orig_path,
        )


class ManifestEvalDataset(Dataset):
    """Row-wise evaluation matching the existing protocol."""

    def __init__(self, manifest: str, split: str, cfg: dict):
        self.items = []
        with open(manifest, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != split:
                    continue
                path = row.get("derived_path")
                if not path or not os.path.isfile(path):
                    continue
                label = int(row.get("is_watermarked", 0))
                method = canonical_method(row.get("watermark_method")) if label else "clean"
                self.items.append((path, label, method))
        if not self.items:
            raise RuntimeError(f"No valid evaluation rows found in split={split}")
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label, method = self.items[index]
        return load_audio(path, self.cfg), label, method


def collate_pairs(batch):
    clean, wm, methods, orig_paths = zip(*batch)
    return list(clean), list(wm), list(methods), list(orig_paths)


def collate_eval(batch):
    wavs, labels, methods = zip(*batch)
    return (
        list(wavs),
        torch.tensor(labels, dtype=torch.float32),
        list(methods),
    )


def make_method_balanced_sampler(dataset: PairedTrainDataset) -> WeightedRandomSampler:
    counts = Counter(item.method for item in dataset.items)
    weights = [1.0 / counts[item.method] for item in dataset.items]
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


class AttentionPool(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int, output_dim=None):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.output = nn.Linear(input_dim, output_dim) if output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.output(pooled)


class ConvBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class XLSRBranch(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg["xlsr_model"])
        self.backbone = Wav2Vec2Model.from_pretrained(cfg["xlsr_model"])
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        input_dim = int(self.backbone.config.hidden_size)
        channels = int(cfg["xlsr_conv_channels"])
        self.proj = nn.Conv1d(input_dim, channels, 1)
        self.conv = nn.Sequential(
            *[ConvBlock1D(channels) for _ in range(int(cfg["xlsr_conv_layers"]))]
        )
        self.lstm = nn.LSTM(
            channels,
            int(cfg["xlsr_lstm_hidden"]),
            num_layers=int(cfg["xlsr_lstm_layers"]),
            batch_first=True,
            bidirectional=True,
        )
        self.output_dim = 2 * int(cfg["xlsr_lstm_hidden"])
        self.pool = AttentionPool(self.output_dim, int(cfg["attn_dim"]))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        arrays = [wav.detach().cpu().numpy() for wav in wavs]
        inputs = self.processor(
            arrays,
            sampling_rate=int(self.cfg["sample_rate"]),
            return_tensors="pt",
            padding=True,
        )
        values = inputs.input_values.to(wavs.device)
        mask = inputs.attention_mask.to(wavs.device)
        with torch.no_grad():
            outputs = self.backbone(
                values,
                attention_mask=mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[int(self.cfg["layer_idx"])]
        x = self.conv(self.proj(hidden.transpose(1, 2))).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.pool(x)


class WaveformBranch(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        channels = parse_int_list(cfg["wave_channels"])
        kernels = parse_int_list(cfg["wave_kernels"])
        strides = parse_int_list(cfg["wave_strides"])
        if not (len(channels) == len(kernels) == len(strides)):
            raise ValueError("wave_channels, wave_kernels and wave_strides must match")
        blocks = []
        in_channels = 1
        for out_channels, kernel, stride in zip(channels, kernels, strides):
            blocks.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride=stride,
                        padding=kernel // 2,
                    ),
                    nn.GroupNorm(1, out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.pool = AttentionPool(
            channels[-1], int(cfg["attn_dim"]), int(cfg["embedding_dim"])
        )
        self.output_dim = int(cfg["embedding_dim"])

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        x = self.encoder(wavs.unsqueeze(1)).transpose(1, 2)
        return self.pool(x)


class STFTBranch(nn.Module):
    def __init__(self, cfg: dict, include_phase: bool):
        super().__init__()
        self.n_fft = int(cfg["n_fft"])
        self.hop_length = int(cfg["hop_length"])
        self.win_length = int(cfg["win_length"])
        self.include_phase = include_phase
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        channels = parse_int_list(cfg["stft_channels"])
        input_channels = 3 if include_phase else 1
        blocks = []
        for output_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                ]
            )
            input_channels = output_channels
        self.encoder = nn.Sequential(*blocks)
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[-1], int(cfg["embedding_dim"])),
        )
        self.output_dim = int(cfg["embedding_dim"])

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(
            wavs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(wavs.device),
            return_complex=True,
            center=True,
        )
        magnitude = torch.log1p(torch.abs(spectrum)).unsqueeze(1)
        if self.include_phase:
            phase = torch.angle(spectrum)
            delta = torch.diff(phase, dim=-1, prepend=phase[..., :1])
            features = torch.cat(
                [magnitude, torch.cos(delta).unsqueeze(1), torch.sin(delta).unsqueeze(1)],
                dim=1,
            )
        else:
            features = magnitude
        return self.output(self.encoder(features))


class GeneralizedXAWMD(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        requested = [name.strip() for name in str(cfg["features"]).split("+") if name.strip()]
        valid = {"xlsr", "waveform", "stft_mag", "stft_mag_phase"}
        unknown = set(requested) - valid
        if unknown:
            raise ValueError(f"Unknown feature branch(es): {sorted(unknown)}")
        if "stft_mag" in requested and "stft_mag_phase" in requested:
            raise ValueError("Choose stft_mag or stft_mag_phase, not both")

        self.branch_names = requested
        branches = {}
        for name in requested:
            if name == "xlsr":
                branches[name] = XLSRBranch(cfg)
            elif name == "waveform":
                branches[name] = WaveformBranch(cfg)
            elif name == "stft_mag":
                branches[name] = STFTBranch(cfg, include_phase=False)
            elif name == "stft_mag_phase":
                branches[name] = STFTBranch(cfg, include_phase=True)
        self.branches = nn.ModuleDict(branches)

        fused_dim = sum(branch.output_dim for branch in self.branches.values())
        if len(requested) == 1:
            # For features=xlsr this preserves the existing
            # XLS-R -> Conv -> BiLSTM -> Attention -> Linear architecture.
            self.fusion = nn.Identity()
            classifier_dim = fused_dim
        else:
            self.fusion = nn.Sequential(
                nn.Linear(fused_dim, int(cfg["fusion_hidden"])),
                nn.LayerNorm(int(cfg["fusion_hidden"])),
                nn.GELU(),
                nn.Dropout(float(cfg["dropout"])),
            )
            classifier_dim = int(cfg["fusion_hidden"])
        self.classifier = nn.Linear(classifier_dim, 1)

    def forward(self, wavs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = [self.branches[name](wavs) for name in self.branch_names]
        fused = self.fusion(torch.cat(embeddings, dim=-1))
        logits = self.classifier(fused).squeeze(-1)
        return logits, F.normalize(fused, dim=-1)


def pair_ranking_loss(
    clean_logits: torch.Tensor, wm_logits: torch.Tensor, margin: float
) -> torch.Tensor:
    clean_scores = torch.sigmoid(clean_logits)
    wm_scores = torch.sigmoid(wm_logits)
    return F.relu(float(margin) - wm_scores + clean_scores).mean()


def method_invariance_loss(
    wm_embeddings: torch.Tensor, method_ids: torch.Tensor
) -> torch.Tensor:
    """
    Penalize dispersion between normalized watermark-method centroids.
    BCE and pair ranking preserve class discrimination; this term only aligns
    positive-group centroids and is therefore kept at a small weight.
    """
    centroids = []
    for method_id in torch.unique(method_ids):
        mask = method_ids == method_id
        if torch.any(mask):
            centroid = F.normalize(wm_embeddings[mask].mean(dim=0), dim=0)
            centroids.append(centroid)
    if len(centroids) < 2:
        return wm_embeddings.new_tensor(0.0)
    stacked = torch.stack(centroids)
    global_centroid = F.normalize(stacked.mean(dim=0), dim=0)
    return ((stacked - global_centroid.unsqueeze(0)) ** 2).sum(dim=1).mean()


class GroupDRO:
    def __init__(self, methods: Sequence[str], eta: float, device: torch.device):
        self.methods = list(sorted(methods))
        self.method_to_id = {name: idx for idx, name in enumerate(self.methods)}
        self.eta = float(eta)
        self.q = torch.ones(len(self.methods), device=device) / len(self.methods)

    def ids(self, methods: Sequence[str], device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.method_to_id[method] for method in methods],
            dtype=torch.long,
            device=device,
        )

    def loss(
        self, per_pair_losses: torch.Tensor, method_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        group_losses = torch.zeros_like(self.q)
        present = torch.zeros_like(self.q, dtype=torch.bool)
        for group_id in torch.unique(method_ids):
            mask = method_ids == group_id
            group_losses[group_id] = per_pair_losses[mask].mean()
            present[group_id] = True

        with torch.no_grad():
            self.q[present] *= torch.exp(self.eta * group_losses[present].detach())
            self.q /= self.q.sum().clamp_min(1e-12)

        # Renormalize over groups represented in this minibatch.
        local_q = self.q * present.float()
        local_q = local_q / local_q.sum().clamp_min(1e-12)
        robust_loss = torch.sum(local_q * group_losses)
        stats = {name: float(self.q[idx].item()) for idx, name in enumerate(self.methods)}
        return robust_loss, stats

    def state_dict(self) -> dict:
        return {"methods": self.methods, "eta": self.eta, "q": self.q.detach().cpu()}

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        if state.get("methods") != self.methods:
            raise ValueError("GroupDRO method list in checkpoint does not match dataset")
        self.q = state["q"].to(device)


def flatten_pair_batch(
    clean_wavs: Sequence[np.ndarray], wm_wavs: Sequence[np.ndarray], device: torch.device
) -> torch.Tensor:
    arrays = list(clean_wavs) + list(wm_wavs)
    return torch.from_numpy(np.stack(arrays)).to(device=device, dtype=torch.float32)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    labels, scores, methods = [], [], []
    with torch.no_grad():
        for wavs, batch_labels, batch_methods in loader:
            tensor = torch.from_numpy(np.stack(wavs)).to(device=device, dtype=torch.float32)
            logits, _ = model(tensor)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.tolist())
            methods.extend(batch_methods)
    return np.asarray(labels), np.asarray(scores), methods


def per_method_report(labels: np.ndarray, scores: np.ndarray, methods: Sequence[str]) -> dict:
    report = {}
    clean_scores = scores[labels == 0]
    report["clean_mean_score"] = float(clean_scores.mean()) if len(clean_scores) else math.nan
    for method in sorted(set(methods) - {"clean"}):
        positive_mask = np.asarray(methods) == method
        method_scores = scores[positive_mask]
        combined_scores = np.concatenate([clean_scores, method_scores])
        combined_labels = np.concatenate(
            [np.zeros(len(clean_scores), dtype=int), np.ones(len(method_scores), dtype=int)]
        )
        report[method] = {
            "count": int(len(method_scores)),
            "mean_score": float(method_scores.mean()) if len(method_scores) else math.nan,
            "auroc_vs_all_clean": (
                float(roc_auc_score(combined_labels, combined_scores))
                if len(clean_scores) and len(method_scores)
                else math.nan
            ),
        }
    return report


def choose_threshold(labels: np.ndarray, scores: np.ndarray, cfg: dict) -> float:
    if cfg.get("fixed_thr") is not None:
        return float(cfg["fixed_thr"])
    if cfg["thr_strategy"] == "f1_watermark":
        best_threshold, best_f1 = 0.5, -1.0
        for threshold in np.linspace(0.0, 1.0, 201):
            pred = scores >= threshold
            tp = np.sum(pred & (labels == 1))
            fp = np.sum(pred & (labels == 0))
            fn = np.sum((~pred) & (labels == 1))
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            if f1 > best_f1:
                best_threshold, best_f1 = float(threshold), float(f1)
        return best_threshold
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--manifest")
    parser.add_argument("--train_split")
    parser.add_argument("--val_split")
    parser.add_argument("--features")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save_path")
    parser.add_argument("--log_file")
    parser.add_argument("--pair_weight", type=float)
    parser.add_argument("--pair_margin", type=float)
    parser.add_argument("--invariant_weight", type=float)
    parser.add_argument("--groupdro_eta", type=float)
    parser.add_argument("--method_balanced", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--groupdro", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--fixed_thr", type=float)
    parser.add_argument(
        "--thr_strategy", choices=["tpr_fpr", "f1_watermark"]
    )
    parser.add_argument("--early_stop_patience", type=int)
    parser.add_argument("--log_every", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CFG.copy()
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            cfg.update(json.load(handle))

    # In evaluation mode, restore the architecture from the checkpoint first.
    # Explicit CLI values below still override runtime settings such as
    # manifest, val_split, device and fixed_thr.
    checkpoint_for_eval = None
    if args.eval_only:
        requested_save_path = args.save_path or cfg["save_path"]
        checkpoint_for_eval = torch.load(requested_save_path, map_location="cpu")
        cfg.update(checkpoint_for_eval.get("cfg", {}))
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            cfg[key] = value
    if cfg["manifest"] is None:
        raise SystemExit("--manifest is required (or set 'manifest' in a JSON config)")

    set_seed(int(cfg["seed"]))
    device = torch.device(cfg["device"])
    log_handle = None
    if cfg.get("log_file"):
        os.makedirs(os.path.dirname(cfg["log_file"]) or ".", exist_ok=True)
        log_handle = open(cfg["log_file"], "a", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        if log_handle:
            log_handle.write(message + "\n")
            log_handle.flush()

    train_set = PairedTrainDataset(cfg["manifest"], cfg["train_split"], cfg)
    val_set = ManifestEvalDataset(cfg["manifest"], cfg["val_split"], cfg)
    sampler = make_method_balanced_sampler(train_set) if cfg["method_balanced"] else None
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_pairs,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["batch_size"]) * 2,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_eval,
        pin_memory=device.type == "cuda",
    )

    log(json.dumps(cfg, indent=2, sort_keys=True))
    log(
        f"Pairs={len(train_set)} val_rows={len(val_set)} methods={train_set.methods} "
        f"missing_clean={train_set.missing_clean}"
    )

    model = GeneralizedXAWMD(cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Parameters: trainable={trainable:,} total={total:,}")

    groupdro = (
        GroupDRO(train_set.methods, cfg["groupdro_eta"], device)
        if cfg["groupdro"]
        else None
    )

    if cfg["eval_only"]:
        checkpoint = checkpoint_for_eval
        if checkpoint is None:
            checkpoint = torch.load(cfg["save_path"], map_location=device)
        model.load_state_dict(checkpoint["model"])
        if groupdro and checkpoint.get("groupdro"):
            groupdro.load_state_dict(checkpoint["groupdro"], device)
    else:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        best_auc = -math.inf
        epochs_without_improvement = 0
        for epoch in range(int(cfg["num_epochs"])):
            model.train()
            running = defaultdict(float)
            seen_pairs = 0
            started = time.time()
            last_q = {}
            for step, (clean_wavs, wm_wavs, methods, _) in enumerate(train_loader, start=1):
                wavs = flatten_pair_batch(clean_wavs, wm_wavs, device)
                logits, embeddings = model(wavs)
                pair_count = len(methods)
                clean_logits, wm_logits = logits[:pair_count], logits[pair_count:]
                clean_embeddings = embeddings[:pair_count]
                wm_embeddings = embeddings[pair_count:]

                clean_targets = torch.zeros(pair_count, device=device)
                wm_targets = torch.ones(pair_count, device=device)
                clean_bce = F.binary_cross_entropy_with_logits(
                    clean_logits, clean_targets, reduction="none"
                )
                wm_bce = F.binary_cross_entropy_with_logits(
                    wm_logits, wm_targets, reduction="none"
                )
                per_pair_bce = 0.5 * (clean_bce + wm_bce)

                if groupdro:
                    method_ids = groupdro.ids(methods, device)
                    bce_loss, last_q = groupdro.loss(per_pair_bce, method_ids)
                else:
                    method_to_local = {
                        method: idx for idx, method in enumerate(sorted(set(methods)))
                    }
                    method_ids = torch.tensor(
                        [method_to_local[method] for method in methods],
                        device=device,
                        dtype=torch.long,
                    )
                    bce_loss = per_pair_bce.mean()

                pair_loss = pair_ranking_loss(
                    clean_logits, wm_logits, float(cfg["pair_margin"])
                )
                invariant_loss = method_invariance_loss(wm_embeddings, method_ids)
                loss = (
                    bce_loss
                    + float(cfg["pair_weight"]) * pair_loss
                    + float(cfg["invariant_weight"]) * invariant_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                seen_pairs += pair_count
                running["loss"] += float(loss.item()) * pair_count
                running["bce"] += float(bce_loss.item()) * pair_count
                running["pair"] += float(pair_loss.item()) * pair_count
                running["inv"] += float(invariant_loss.item()) * pair_count
                if step % int(cfg["log_every"]) == 0:
                    rate = seen_pairs / max(time.time() - started, 1e-6)
                    log(
                        f"[train] epoch={epoch + 1} pairs={seen_pairs}/{len(train_set)} "
                        f"loss={running['loss'] / seen_pairs:.4f} "
                        f"bce={running['bce'] / seen_pairs:.4f} "
                        f"pair={running['pair'] / seen_pairs:.4f} "
                        f"inv={running['inv'] / seen_pairs:.4f} "
                        f"pairs/s={rate:.2f}"
                    )

            labels, scores, methods = evaluate(model, val_loader, device)
            val_auc = float(roc_auc_score(labels, scores))
            log(
                f"Epoch {epoch + 1}/{cfg['num_epochs']} "
                f"train_loss={running['loss'] / max(seen_pairs, 1):.4f} "
                f"val_AUROC={val_auc:.4f}"
            )
            if last_q:
                log("GroupDRO q=" + json.dumps(last_q, sort_keys=True))

            if val_auc > best_auc:
                best_auc = val_auc
                epochs_without_improvement = 0
                os.makedirs(os.path.dirname(cfg["save_path"]) or ".", exist_ok=True)
                torch.save(
                    {
                        "cfg": cfg,
                        "model": model.state_dict(),
                        "groupdro": groupdro.state_dict() if groupdro else None,
                        "val_auroc": val_auc,
                    },
                    cfg["save_path"],
                )
                log(f"Saved best checkpoint to {cfg['save_path']}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(cfg["early_stop_patience"]):
                    log("Early stopping")
                    break

        checkpoint = torch.load(cfg["save_path"], map_location=device)
        model.load_state_dict(checkpoint["model"])

    labels, scores, methods = evaluate(model, val_loader, device)
    auc = float(roc_auc_score(labels, scores))
    threshold = choose_threshold(labels, scores, cfg)
    predictions = (scores >= threshold).astype(int)
    method_metrics = per_method_report(labels, scores, methods)
    log(f"AUROC: {auc:.4f}")
    log(f"Best thr: {threshold:.4f}")
    log(classification_report(labels, predictions, target_names=["Clean", "Watermarked"], zero_division=0))
    log("Per-method metrics:\n" + json.dumps(method_metrics, indent=2, sort_keys=True))

    result_path = os.path.splitext(cfg["save_path"])[0] + f"_{cfg['val_split']}_scores.npz"
    np.savez_compressed(
        result_path,
        labels=labels,
        scores=scores,
        methods=np.asarray(methods),
        threshold=np.asarray([threshold]),
    )
    log(f"Saved scores to {result_path}")
    if log_handle:
        log_handle.close()


if __name__ == "__main__":
    main()
