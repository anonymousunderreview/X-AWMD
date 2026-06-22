#!/usr/bin/env python3
"""
XLS-R + Conv1D + Conformer + attention pooling ablation.

Replaces the BiLSTM in X-AWMD with a Conformer encoder.
This is the "Conformer" row in the Temporal Head Variant section of Table V.

Usage:
    python ablation/xlsr_conv_conformer.py --manifest /path/to/dataset_manifest.csv
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

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

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
    "xlsr_model": "facebook/wav2vec2-xls-r-300m",
    "layer_idx": 9,
    "layer_range": None,
    "layer_fuse": "single",
    "conv_channels": 256,
    "conv_layers": 3,
    "n_heads": 8,
    "ff_dim": 512,
    "conf_layers": 2,
    "conv_kernel": 15,
    "attn_dim": 128,
    "save_path": "./checkpoints/xlsr_conv_conformer.pth",
    "early_stop_patience": 3,
    "log_every": 50,
    "log_every_eval": 50,
    "log_file": None,
    "thr_strategy": "tpr_fpr",
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


class XLSRFeature(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg["xlsr_model"])
        self.model = Wav2Vec2Model.from_pretrained(cfg["xlsr_model"]).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.layer_range = None
        if cfg.get("layer_range"):
            lo, hi = cfg["layer_range"].split(",")
            self.layer_range = (int(lo), int(hi))

    def select_layer(self, hs):
        if self.cfg["layer_fuse"] == "avg" and self.layer_range is not None:
            lo, hi = self.layer_range
            return torch.stack(hs[lo:hi + 1], dim=0).mean(0)
        return hs[int(self.cfg["layer_idx"])]

    @torch.no_grad()
    def forward(self, wavs):
        inputs = self.processor(wavs, sampling_rate=self.cfg["sample_rate"],
                                return_tensors="pt", padding=True)
        out = self.model(inputs.input_values.to(self.device),
                         attention_mask=inputs.attention_mask.to(self.device),
                         output_hidden_states=True)
        return self.select_layer(out.hidden_states)


class ConvBlock1D(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(ch, ch, k, padding=k // 2), nn.BatchNorm1d(ch), nn.ReLU())

    def forward(self, x): return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim, conv_kernel):
        super().__init__()
        self.ffn1 = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim),
            nn.BatchNorm1d(dim), nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.ffn2 = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.ln1, self.ln2, self.ln3, self.ln4 = [nn.LayerNorm(dim) for _ in range(4)]

    def forward(self, x):
        x = x + 0.5 * self.ffn1(self.ln1(x))
        attn_out, _ = self.self_attn(self.ln2(x), self.ln2(x), self.ln2(x))
        x = x + attn_out
        x = x + self.conv(self.ln3(x).transpose(1, 2)).transpose(1, 2)
        x = x + 0.5 * self.ffn2(self.ln4(x))
        return x


class XLSRConvConformer(nn.Module):
    def __init__(self, cfg, in_dim):
        super().__init__()
        c = cfg["conv_channels"]
        self.proj = nn.Conv1d(in_dim, c, 1)
        self.blocks = nn.Sequential(*[ConvBlock1D(c) for _ in range(cfg["conv_layers"])])
        self.conformer = nn.ModuleList([
            ConformerBlock(c, cfg["n_heads"], cfg["ff_dim"], cfg["conv_kernel"])
            for _ in range(cfg["conf_layers"])
        ])
        self.attn = nn.Sequential(nn.Linear(c, cfg["attn_dim"]), nn.Tanh(), nn.Linear(cfg["attn_dim"], 1))
        self.head = nn.Linear(c, 1)

    def forward(self, h):
        x = self.proj(h.transpose(1, 2))
        x = self.blocks(x).transpose(1, 2)
        for blk in self.conformer:
            x = blk(x)
        a = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        return self.head((x * a.unsqueeze(-1)).sum(1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser(description="XLS-R + Conformer ablation (Table V temporal head section)")
    ap.add_argument("--config",              type=str,   default=None)
    ap.add_argument("--manifest",            type=str,   default=None, help="Path to dataset_manifest.csv (required)")
    ap.add_argument("--duration",            type=float, default=None)
    ap.add_argument("--train_split",         type=str,   default=None)
    ap.add_argument("--val_split",           type=str,   default=None)
    ap.add_argument("--batch_size",          type=int,   default=None)
    ap.add_argument("--num_epochs",          type=int,   default=None)
    ap.add_argument("--lr",                  type=float, default=None)
    ap.add_argument("--xlsr_model",          type=str,   default=None)
    ap.add_argument("--layer_idx",           type=int,   default=None)
    ap.add_argument("--layer_range",         type=str,   default=None)
    ap.add_argument("--layer_fuse",          type=str,   default=None)
    ap.add_argument("--device",              type=str,   default=None)
    ap.add_argument("--save_path",           type=str,   default=None)
    ap.add_argument("--log_every",           type=int,   default=None)
    ap.add_argument("--log_every_eval",      type=int,   default=None)
    ap.add_argument("--early_stop_patience", type=int,   default=None)
    ap.add_argument("--log_file",            type=str,   default=None)
    ap.add_argument("--thr_strategy",        type=str,   default=None)
    ap.add_argument("--seed",                type=int,   default=None)
    ap.add_argument("--eval_only",           action="store_true")
    args = ap.parse_args()

    cfg = CFG.copy()
    if args.config:
        with open(args.config) as f: cfg.update(json.load(f))
    for k in ["manifest", "duration", "train_split", "val_split", "batch_size", "num_epochs", "lr",
              "xlsr_model", "layer_idx", "layer_range", "layer_fuse", "device", "save_path",
              "log_every", "log_every_eval", "early_stop_patience", "log_file", "thr_strategy", "seed"]:
        v = getattr(args, k, None)
        if v is not None:
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
    log(f"XLS-R (frozen) + Conformer | layer={cfg['layer_idx']} | duration={cfg['duration']}s")

    train_set = WADDataset(cfg["manifest"], cfg["train_split"], cfg)
    val_set   = WADDataset(cfg["manifest"], cfg["val_split"],   cfg)
    train_loader = DataLoader(train_set, cfg["batch_size"], shuffle=True,  num_workers=2, collate_fn=collate_batch)
    val_loader   = DataLoader(val_set,   cfg["batch_size"], shuffle=False, num_workers=2, collate_fn=collate_batch)

    feat = XLSRFeature(cfg)
    with torch.no_grad():
        h = feat([load_audio(train_set.items[0][0], cfg)])
    model = XLSRConvConformer(cfg, in_dim=h.shape[-1]).to(device)
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
            try: auc = roc_auc_score(all_y, all_s)
            except Exception: auc = 0.5
            log(f"Epoch {ep+1} | loss {total_loss/max(1,n):.4f} | val_AUROC {auc:.4f}")
            if auc > best_auc:
                best_auc = auc; no_improve = 0
                os.makedirs(os.path.dirname(cfg["save_path"]) or ".", exist_ok=True)
                torch.save({"cfg": cfg, "model": model.state_dict()}, cfg["save_path"])
                log(f"  Saved -> {cfg['save_path']}")
            else:
                no_improve += 1
                if no_improve >= cfg["early_stop_patience"]:
                    log(f"Early stopping."); break
    else:
        model.load_state_dict(torch.load(cfg["save_path"], map_location=device)["model"])

    model.eval()
    all_y, all_s = [], []
    with torch.no_grad():
        for wavs, y in val_loader:
            all_s.extend(torch.sigmoid(model(feat(wavs).to(device))).cpu().tolist())
            all_y.extend(y.tolist())
    auroc = roc_auc_score(all_y, all_s)
    fpr, tpr, thr = roc_curve(all_y, all_s)
    best_thr = float(thr[np.argmax(tpr - fpr)])
    log(f"Final Val AUROC: {auroc:.4f}  thr: {best_thr:.4f}")
    log(classification_report(all_y, (np.array(all_s) >= best_thr).astype(int),
                               target_names=["Clean", "Watermarked"], zero_division=0))
    if log_fh: log_fh.close()


if __name__ == "__main__":
    main()
