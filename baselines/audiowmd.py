#!/usr/bin/env python3
"""
AudioWMD baseline: 2D CNN spectrogram detector + query-statistics logistic regression.

Re-implementation following:
  Sedaghati et al., "VoxWatermark: A Benchmark for Audio Watermark Detection," 2026.

Two-stage training:
  Stage 1: Train a SmallCNN binary classifier on log-mel spectrograms.
  Stage 2: For each audio, run K stochastic perturbation queries through the CNN,
           aggregate statistics (mean, std, max-min, vote rate, flip rate), and
           fit logistic regression on those features.

Usage:
    python baselines/audiowmd.py --manifest /path/to/dataset_manifest.csv
    python baselines/audiowmd.py --manifest /path/to/dataset_manifest.csv --skip_base \
        --save_base ./checkpoints/audiowmd_base.pth
"""
import os, csv, json, time, argparse, random, pickle
from typing import List, Tuple

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

CFG = {
    "manifest": None,
    "train_split": "train",
    "val_split": "validation",
    "batch_size": 64,
    "num_epochs": 5,
    "lr": 1e-3,
    "sample_rate": 16000,
    "duration": 4.0,                        # audio clip length in seconds
    "n_fft": 2048,
    "hop": 512,
    "n_mels": 64,
    "fixed_frames": 216,
    "queries": 8,
    "save_base": "./checkpoints/audiowmd_base.pth",
    "save_meta": "./checkpoints/audiowmd_meta.pkl",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_audio(path, cfg):
    y, _ = librosa.load(path, sr=cfg["sample_rate"])
    if cfg.get("duration") is None:
        return y
    tgt = int(cfg["sample_rate"] * cfg["duration"])
    return np.pad(y, (0, max(0, tgt - len(y))))[:tgt]


def wav_to_mel(y, cfg):
    m = librosa.feature.melspectrogram(
        y=y, sr=cfg["sample_rate"], n_fft=cfg["n_fft"], hop_length=cfg["hop"],
        n_mels=cfg["n_mels"], fmin=0, fmax=cfg["sample_rate"] // 2,
    )
    m = librosa.power_to_db(m, ref=np.max)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    T = cfg["fixed_frames"]
    if m.shape[1] < T:
        m = np.pad(m, ((0, 0), (0, T - m.shape[1])))
    else:
        m = m[:, :T]
    return m.astype(np.float32)


class WADDataset(Dataset):
    def __init__(self, manifest, split, cfg):
        self.items, self.cfg = [], cfg
        with open(manifest, "r") as f:
            for row in csv.DictReader(f):
                if row.get("split") != split: continue
                p = row.get("derived_path")
                if p and os.path.isfile(p):
                    self.items.append((p, int(row.get("is_watermarked", 0))))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        mel = wav_to_mel(load_audio(p, self.cfg), self.cfg)
        return torch.tensor(mel[None, ...]), torch.tensor(y, dtype=torch.long)


class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(base * 4, 1)

    def forward(self, x):
        return self.head(self.net(x).flatten(1)).squeeze(1)


def perturb_wave(y, cfg):
    sr = cfg["sample_rate"]
    p = random.random()
    if p < 0.2:
        y = np.roll(y, random.randint(-int(0.05 * len(y)), int(0.05 * len(y))))
    elif p < 0.4:
        y = librosa.effects.time_stretch(y, rate=random.uniform(0.92, 1.08))
    elif p < 0.6:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-0.4, 0.4))
    elif p < 0.75:
        y = y + np.random.randn(len(y)) * 0.005
    elif p < 0.9:
        y = y * random.uniform(0.9, 1.1)
    else:
        s = random.randint(0, len(y) - 1)
        y[s:s + int(0.01 * len(y))] = 0
    if cfg.get("duration") is not None:
        tgt = int(sr * cfg["duration"])
        y = np.pad(y, (0, max(0, tgt - len(y))))[:tgt]
    return y


def train_base(cfg):
    device = torch.device(cfg["device"])
    train_set = WADDataset(cfg["manifest"], cfg["train_split"], cfg)
    val_set = WADDataset(cfg["manifest"], cfg["val_split"], cfg)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model = SmallCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    bce = nn.BCEWithLogitsLoss()
    best_auc = 0.0

    for ep in range(cfg["num_epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            loss = bce(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        all_y, all_s = [], []
        with torch.no_grad():
            for x, y in val_loader:
                all_s.extend(torch.sigmoid(model(x.to(device))).cpu().tolist())
                all_y.extend(y.tolist())
        try:
            auc = roc_auc_score(all_y, all_s)
        except Exception:
            auc = 0.5
        print(f"Epoch {ep+1}/{cfg['num_epochs']} Val AUROC {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            os.makedirs(os.path.dirname(cfg["save_base"]) or ".", exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg}, cfg["save_base"])
            print(f"Saved base model to {cfg['save_base']}")


def query_blackbox(model, device, paths: List[str], cfg) -> Tuple[np.ndarray, np.ndarray]:
    meta = {}
    with open(cfg["manifest"], "r") as f:
        for row in csv.DictReader(f):
            meta[row["derived_path"]] = int(row.get("is_watermarked", 0))

    K = cfg["queries"]
    feats, labels = [], []
    total = len(paths)
    log_every = max(500, total // 20) if total else 1

    for i, path in enumerate(paths):
        y0 = load_audio(path, cfg)
        scores = []
        for k in range(K):
            yy = y0 if k == 0 else perturb_wave(y0, cfg)
            mel = wav_to_mel(yy, cfg)
            x = torch.tensor(mel[None, None, ...], dtype=torch.float32, device=device)
            with torch.no_grad():
                s = torch.sigmoid(model(x)).item()
            scores.append(s)
        scores = np.array(scores)
        feats.append([
            scores.mean(), scores.std(), scores.max() - scores.min(),
            float((scores > 0.5).mean()),
            float((np.sign(scores - scores.mean()) != np.sign(scores[0] - 0.5)).mean()),
        ])
        labels.append(meta.get(path, 0))
        if (i + 1) % log_every == 0 or (i + 1) == total:
            print(f"[query] {i+1}/{total} processed", flush=True)

    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description="Train or evaluate AudioWMD baseline")
    ap.add_argument("--config",      type=str,   default=None)
    ap.add_argument("--manifest",    type=str,   default=None, help="Path to dataset_manifest.csv (required)")
    ap.add_argument("--duration",    type=float, default=None, help="Audio clip length in seconds")
    ap.add_argument("--train_split", type=str,   default=None)
    ap.add_argument("--val_split",   type=str,   default=None)
    ap.add_argument("--batch_size",  type=int,   default=None)
    ap.add_argument("--num_epochs",  type=int,   default=None)
    ap.add_argument("--lr",          type=float, default=None)
    ap.add_argument("--queries",     type=int,   default=None)
    ap.add_argument("--save_base",   type=str,   default=None)
    ap.add_argument("--save_meta",   type=str,   default=None)
    ap.add_argument("--device",      type=str,   default=None)
    ap.add_argument("--skip_base",   action="store_true", help="Skip CNN training, load existing save_base")
    args = ap.parse_args()

    cfg = CFG.copy()
    if args.config:
        with open(args.config, "r") as f:
            cfg.update(json.load(f))
    for k in ["manifest", "duration", "train_split", "val_split", "batch_size",
              "num_epochs", "lr", "queries", "save_base", "save_meta", "device"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    if cfg["manifest"] is None:
        ap.error("--manifest is required (or set 'manifest' in a --config JSON file)")

    if not args.skip_base or not os.path.isfile(cfg["save_base"]):
        train_base(cfg)

    device = torch.device(cfg["device"])
    model = SmallCNN().to(device)
    state = torch.load(cfg["save_base"], map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    train_paths, val_paths = [], []
    with open(cfg["manifest"], "r") as f:
        for row in csv.DictReader(f):
            p = row.get("derived_path")
            if not p or not os.path.isfile(p): continue
            if row.get("split") == cfg["train_split"]:
                train_paths.append(p)
            elif row.get("split") == cfg["val_split"]:
                val_paths.append(p)
    print(f"Train samples: {len(train_paths)}  Val samples: {len(val_paths)}")

    t0 = time.time()
    X_train, y_train = query_blackbox(model, device, train_paths, cfg)
    X_val, y_val = query_blackbox(model, device, val_paths, cfg)
    print(f"Query time: {(time.time()-t0)/60:.2f} min")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    os.makedirs(os.path.dirname(cfg["save_meta"]) or ".", exist_ok=True)
    with open(cfg["save_meta"], "wb") as f:
        pickle.dump({"cfg": cfg, "clf": clf}, f)
    print(f"Meta saved to {cfg['save_meta']}")

    val_score = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, val_score)
    fpr, tpr, thr = roc_curve(y_val, val_score)
    best_thr = float(thr[int(np.argmax(tpr - fpr))])
    print(f"Val AUROC: {auroc:.4f}  Best thr: {best_thr:.4f}")
    pred = (val_score >= best_thr).astype(int)
    print(classification_report(y_val, pred, target_names=["Clean", "Watermarked"], zero_division=0))


if __name__ == "__main__":
    main()
