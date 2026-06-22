#!/usr/bin/env python3
"""
WavLM + Temp.Head ablation: frozen WavLM-Large + Conv1D + BiLSTM + attention pooling.

Drop-in backbone replacement for X-AWMD. Same temporal head and training loop;
only the frozen front-end changes from XLS-R-300M to WavLM-Large.
This is the "WavLM + Temp.Head" row in Table V of the X-AWMD paper.

Usage:
    python ablation/wavlm_conv_lstm.py --manifest /path/to/dataset_manifest.csv
"""
import os, json, time, argparse, random
from typing import List

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from transformers import Wav2Vec2FeatureExtractor, WavLMModel

CFG = {
    "manifest": None,
    "train_split": "train",
    "val_split": "validation",
    "sample_rate": 16000,
    "duration": 4.0,                        # audio clip length in seconds
    "batch_size": 6,
    "num_epochs": 15,
    "lr": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wavlm_model": "microsoft/wavlm-large",
    "layer_idx": 9,
    "layer_range": None,
    "layer_fuse": "single",
    "conv_channels": 256,
    "conv_layers": 3,
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "attn_dim": 128,
    "save_path": "./checkpoints/wavlm_conv_lstm.pth",
    "early_stop_patience": 3,
    "log_every": 50,
    "log_every_eval": 50,
    "log_file": None,
    "thr_strategy": "tpr_fpr",
    "fixed_thr": None,
    "seed": 42,
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_audio(path, cfg):
    y, _ = librosa.load(path, sr=cfg["sample_rate"])
    if cfg.get("duration") is None:
        return y
    tgt = int(cfg["sample_rate"] * cfg["duration"])
    return np.pad(y, (0, max(0, tgt - len(y))))[:tgt]


class WADDataset(Dataset):
    def __init__(self, manifest, split, cfg):
        import csv
        self.items, self.cfg = [], cfg
        with open(manifest) as f:
            for row in csv.DictReader(f):
                if row.get("split") != split: continue
                p = row.get("derived_path")
                if p and os.path.isfile(p):
                    self.items.append((p, int(row.get("is_watermarked", 0))))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return load_audio(p, self.cfg).astype(np.float32), y


def collate_batch(batch):
    wavs, labels = zip(*batch)
    return list(wavs), torch.tensor(labels, dtype=torch.float32)


class WavLMFeature(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg["wavlm_model"])
        self.model = WavLMModel.from_pretrained(cfg["wavlm_model"]).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.layer_range = None
        if cfg.get("layer_range"):
            lo, hi = cfg["layer_range"].split(",")
            self.layer_range = (int(lo), int(hi))

    def select_layer(self, hidden_states):
        if self.cfg["layer_fuse"] == "avg" and self.layer_range is not None:
            lo, hi = self.layer_range
            return torch.stack(hidden_states[lo:hi + 1], dim=0).mean(0)
        return hidden_states[int(self.cfg["layer_idx"])]

    @torch.no_grad()
    def forward(self, wavs: List[np.ndarray]) -> torch.Tensor:
        inputs = self.processor(wavs, sampling_rate=self.cfg["sample_rate"],
                                return_tensors="pt", padding=True)
        out = self.model(inputs.input_values.to(self.device),
                         attention_mask=inputs.attention_mask.to(self.device),
                         output_hidden_states=True)
        return self.select_layer(out.hidden_states)  # [B, T, 1024]


class ConvBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU())

    def forward(self, x): return self.net(x)


class TemporalHead(nn.Module):
    def __init__(self, cfg, in_dim):
        super().__init__()
        c = cfg["conv_channels"]
        self.proj = nn.Conv1d(in_dim, c, 1)
        self.blocks = nn.Sequential(*[ConvBlock1D(c) for _ in range(cfg["conv_layers"])])
        self.lstm = nn.LSTM(c, cfg["lstm_hidden"], cfg["lstm_layers"],
                            batch_first=True, bidirectional=True)
        h2 = cfg["lstm_hidden"] * 2
        self.attn = nn.Sequential(nn.Linear(h2, cfg["attn_dim"]), nn.Tanh(), nn.Linear(cfg["attn_dim"], 1))
        self.head = nn.Linear(h2, 1)

    def forward(self, h):
        x = self.proj(h.transpose(1, 2))
        x = self.blocks(x).transpose(1, 2)
        x, _ = self.lstm(x)
        a = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        return self.head((x * a.unsqueeze(-1)).sum(1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser(description="WavLM + Temp.Head ablation (Table V backbone section)")
    ap.add_argument("--config",              type=str,   default=None)
    ap.add_argument("--manifest",            type=str,   default=None, help="Path to dataset_manifest.csv (required)")
    ap.add_argument("--duration",            type=float, default=None, help="Audio clip length in seconds")
    for k in ["train_split", "val_split", "save_path", "wavlm_model", "layer_range", "layer_fuse", "device", "log_file", "thr_strategy"]:
        ap.add_argument(f"--{k}", type=str, default=None)
    for k in ["batch_size", "num_epochs", "log_every", "log_every_eval", "early_stop_patience", "layer_idx", "conv_layers", "conv_channels", "lstm_hidden", "lstm_layers", "seed"]:
        ap.add_argument(f"--{k}", type=int, default=None)
    for k in ["lr", "fixed_thr"]:
        ap.add_argument(f"--{k}", type=float, default=None)
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    cfg = CFG.copy()
    if args.config:
        with open(args.config) as f: cfg.update(json.load(f))
    for k, v in vars(args).items():
        if v is not None and k not in ("config", "eval_only"):
            cfg[k] = v

    if cfg["manifest"] is None:
        ap.error("--manifest is required (or set 'manifest' in a --config JSON file)")

    log_fh = None
    if cfg.get("log_file"):
        os.makedirs(os.path.dirname(cfg["log_file"]) or ".", exist_ok=True)
        log_fh = open(cfg["log_file"], "a")

    def log(msg):
        print(msg, flush=True)
        if log_fh: log_fh.write(msg + "\n"); log_fh.flush()

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    log(f"WavLM-Large (frozen) + TemporalHead | layer={cfg['layer_idx']} | lr={cfg['lr']}")
    log(f"Audio duration: {cfg['duration']}s")

    train_set = WADDataset(cfg["manifest"], cfg["train_split"], cfg)
    val_set   = WADDataset(cfg["manifest"], cfg["val_split"],   cfg)
    train_loader = DataLoader(train_set, cfg["batch_size"], shuffle=True,  num_workers=2, collate_fn=collate_batch)
    val_loader   = DataLoader(val_set,   cfg["batch_size"], shuffle=False, num_workers=2, collate_fn=collate_batch)

    feat = WavLMFeature(cfg)
    with torch.no_grad():
        h = feat([load_audio(train_set.items[0][0], cfg)])
    model = TemporalHead(cfg, in_dim=h.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    bce = nn.BCEWithLogitsLoss()

    best_auc, no_improve = 0.0, 0
    if not args.eval_only:
        for ep in range(cfg["num_epochs"]):
            model.train()
            total_loss, n = 0.0, 0
            for wavs, y in train_loader:
                h = feat(wavs).to(device); y = y.to(device)
                loss = bce(model(h), y)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item() * y.size(0); n += y.size(0)
                if n > 0 and (n // cfg["batch_size"]) % cfg["log_every"] == 0:
                    log(f"[train] ep{ep+1} | {n}/{len(train_set)} | loss {loss.item():.4f}")

            model.eval()
            all_y, all_s = [], []
            with torch.no_grad():
                for wavs, y in val_loader:
                    all_s.extend(torch.sigmoid(model(feat(wavs).to(device))).cpu().tolist())
                    all_y.extend(y.tolist())
            auc = roc_auc_score(all_y, all_s)
            log(f"Epoch {ep+1} | loss {total_loss/max(1,n):.4f} | val_AUROC {auc:.4f}")
            if auc > best_auc:
                best_auc = auc; no_improve = 0
                os.makedirs(os.path.dirname(cfg["save_path"]) or ".", exist_ok=True)
                torch.save({"cfg": cfg, "model": model.state_dict()}, cfg["save_path"])
                log(f"  Saved -> {cfg['save_path']}")
            else:
                no_improve += 1
                if no_improve >= cfg["early_stop_patience"]:
                    log(f"Early stopping at epoch {ep+1}."); break
    else:
        model.load_state_dict(torch.load(cfg["save_path"], map_location=device)["model"])

    model.eval()
    all_y, all_s = [], []
    with torch.no_grad():
        for wavs, y in val_loader:
            all_s.extend(torch.sigmoid(model(feat(wavs).to(device))).cpu().tolist())
            all_y.extend(y.tolist())
    auroc = roc_auc_score(all_y, all_s)
    log(f"\nFinal Val AUROC: {auroc:.4f}")
    fpr, tpr, thr = roc_curve(all_y, all_s)
    best_thr = float(thr[np.argmax(tpr - fpr)])
    pred = (np.array(all_s) >= best_thr).astype(int)
    log(f"Threshold (Youden-J): {best_thr:.4f}")
    log(classification_report(all_y, pred, target_names=["Clean", "Watermarked"], zero_division=0))
    if log_fh: log_fh.close()


if __name__ == "__main__":
    main()
