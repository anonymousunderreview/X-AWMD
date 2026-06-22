#!/usr/bin/env python3
"""
X-AWMD: Frozen XLS-R + Conv1D + BiLSTM + Attention Pooling
Binary black-box audio watermark detection (clip-level).

Usage:
    python train.py --manifest /path/to/dataset_manifest.csv --save_path ./checkpoints/xawmd.pth
    python train.py --manifest /path/to/dataset_manifest.csv --eval_only --save_path ./checkpoints/xawmd.pth
    python train.py --config config.json
"""
import os, json, time, argparse, random
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import librosa

CFG = {
    "manifest": None,                       # required: path to dataset_manifest.csv
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
    "layer_range": None,                    # e.g. "6,12" for layer averaging
    "layer_fuse": "single",                 # single | avg
    "conv_channels": 256,
    "conv_layers": 3,
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "attn_dim": 128,
    "augment": False,
    "save_path": "./checkpoints/xawmd.pth",
    "cache_dir": None,
    "eval_only": False,
    "log_every": 50,
    "early_stop_patience": 3,
    "log_file": None,
    "thr_strategy": "tpr_fpr",              # tpr_fpr | f1_watermark
    "log_every_eval": 50,
    "seed": 42,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_audio(path, cfg):
    y, _ = librosa.load(path, sr=cfg["sample_rate"])
    if cfg.get("duration") is None:
        return y
    tgt = int(cfg["sample_rate"] * cfg["duration"])
    if len(y) < tgt:
        y = np.pad(y, (0, tgt - len(y)))
    else:
        y = y[:tgt]
    return y


def perturb_wave(y, cfg):
    p = random.random()
    sr = cfg["sample_rate"]
    if p < 0.2:
        shift = random.randint(-int(0.05 * len(y)), int(0.05 * len(y)))
        y = np.roll(y, shift)
    elif p < 0.4:
        rate = random.uniform(0.92, 1.08)
        y = librosa.effects.time_stretch(y, rate=rate)
    elif p < 0.6:
        steps = random.uniform(-0.4, 0.4)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    elif p < 0.75:
        y = y + np.random.randn(len(y)) * 0.005
    elif p < 0.9:
        y = y * random.uniform(0.9, 1.1)
    else:
        start = random.randint(0, len(y) - 1)
        width = int(0.01 * len(y))
        y[start:start + width] = 0
    if cfg.get("duration") is not None:
        tgt = int(sr * cfg["duration"])
        if len(y) < tgt:
            y = np.pad(y, (0, tgt - len(y)))
        else:
            y = y[:tgt]
    return y


class WADDataset(Dataset):
    def __init__(self, manifest, split, cfg):
        import csv
        self.items = []
        with open(manifest, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") != split:
                    continue
                p = row.get("derived_path")
                if not p or not os.path.isfile(p):
                    continue
                y = int(row.get("is_watermarked", 0))
                self.items.append((p, y))
        self.cfg = cfg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        wav = load_audio(p, self.cfg)
        if self.cfg.get("augment", False):
            if random.random() < 0.6:
                wav = perturb_wave(wav, self.cfg)
        return wav.astype(np.float32), y


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

    def select_layer(self, hidden_states):
        if self.cfg["layer_fuse"] == "avg" and self.layer_range is not None:
            lo, hi = self.layer_range
            hs = torch.stack(hidden_states[lo:hi + 1], dim=0)
            return hs.mean(dim=0)
        idx = int(self.cfg["layer_idx"])
        return hidden_states[idx]

    @torch.no_grad()
    def forward(self, wavs: List[np.ndarray]) -> torch.Tensor:
        inputs = self.processor(
            wavs, sampling_rate=self.cfg["sample_rate"], return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)
        attn_mask = inputs.attention_mask.to(self.device)
        outputs = self.model(input_values, attention_mask=attn_mask, output_hidden_states=True)
        h = self.select_layer(outputs.hidden_states)  # [B, T, D]
        return h


class ConvBlock1D(nn.Module):
    def __init__(self, ch, k=3, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, k, stride=stride, padding=k // 2)
        self.bn = nn.BatchNorm1d(ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class XLSRConvLSTM(nn.Module):
    def __init__(self, cfg, in_dim):
        super().__init__()
        c = cfg["conv_channels"]
        self.proj = nn.Conv1d(in_dim, c, 1)
        self.blocks = nn.Sequential(*[ConvBlock1D(c) for _ in range(cfg["conv_layers"])])
        self.lstm = nn.LSTM(
            input_size=c,
            hidden_size=cfg["lstm_hidden"],
            num_layers=cfg["lstm_layers"],
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(cfg["lstm_hidden"] * 2, cfg["attn_dim"]),
            nn.Tanh(),
            nn.Linear(cfg["attn_dim"], 1),
        )
        self.head = nn.Linear(cfg["lstm_hidden"] * 2, 1)

    def forward(self, h):
        x = h.transpose(1, 2)          # [B, D, T]
        x = self.proj(x)
        x = self.blocks(x)
        x = x.transpose(1, 2)          # [B, T, C]
        x, _ = self.lstm(x)            # [B, T, 2H]
        e = self.attn(x).squeeze(-1)   # [B, T]
        a = torch.softmax(e, dim=1)
        pooled = torch.sum(x * a.unsqueeze(-1), dim=1)  # [B, 2H]
        logit = self.head(pooled).squeeze(-1)
        return logit


def collate_batch(batch):
    wavs, labels = zip(*batch)
    return list(wavs), torch.tensor(labels, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description="Train or evaluate X-AWMD")
    ap.add_argument("--config",              type=str,   default=None,  help="Path to JSON config (overrides CFG defaults)")
    ap.add_argument("--manifest",            type=str,   default=None,  help="Path to dataset_manifest.csv (required)")
    ap.add_argument("--duration",            type=float, default=None,  help="Audio clip length in seconds (default: 3.0)")
    ap.add_argument("--train_split",         type=str,   default=None)
    ap.add_argument("--val_split",           type=str,   default=None)
    ap.add_argument("--batch_size",          type=int,   default=None)
    ap.add_argument("--num_epochs",          type=int,   default=None)
    ap.add_argument("--lr",                  type=float, default=None)
    ap.add_argument("--xlsr_model",          type=str,   default=None)
    ap.add_argument("--layer_idx",           type=int,   default=None)
    ap.add_argument("--layer_range",         type=str,   default=None,  help="e.g. '6,12' for layer averaging")
    ap.add_argument("--layer_fuse",          type=str,   default=None,  choices=["single", "avg"])
    ap.add_argument("--device",              type=str,   default=None)
    ap.add_argument("--save_path",           type=str,   default=None)
    ap.add_argument("--cache_dir",           type=str,   default=None)
    ap.add_argument("--eval_only",           action="store_true")
    ap.add_argument("--log_every",           type=int,   default=None)
    ap.add_argument("--log_every_eval",      type=int,   default=None)
    ap.add_argument("--early_stop_patience", type=int,   default=None)
    ap.add_argument("--log_file",            type=str,   default=None)
    ap.add_argument("--thr_strategy",        type=str,   default=None,  choices=["tpr_fpr", "f1_watermark"])
    ap.add_argument("--fixed_thr",           type=float, default=None)
    ap.add_argument("--conv_layers",         type=int,   default=None)
    ap.add_argument("--conv_channels",       type=int,   default=None)
    ap.add_argument("--lstm_hidden",         type=int,   default=None)
    ap.add_argument("--lstm_layers",         type=int,   default=None)
    ap.add_argument("--seed",                type=int,   default=None)
    args = ap.parse_args()

    cfg = CFG.copy()
    if args.config:
        with open(args.config, "r") as f:
            cfg.update(json.load(f))
    for k in ["manifest", "duration", "train_split", "val_split", "batch_size", "num_epochs",
              "lr", "xlsr_model", "layer_idx", "layer_range", "layer_fuse", "device",
              "save_path", "cache_dir", "log_every", "log_every_eval", "early_stop_patience",
              "log_file", "thr_strategy", "fixed_thr", "conv_layers", "conv_channels",
              "lstm_hidden", "lstm_layers", "seed"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    if cfg["manifest"] is None:
        ap.error("--manifest is required (or set 'manifest' in a --config JSON file)")

    log_fh = None
    if cfg.get("log_file"):
        os.makedirs(os.path.dirname(cfg["log_file"]) or ".", exist_ok=True)
        log_fh = open(cfg["log_file"], "a", encoding="utf-8")

    def log(msg: str):
        print(msg, flush=True)
        if log_fh is not None:
            log_fh.write(msg + "\n")
            log_fh.flush()

    set_seed(int(cfg["seed"]))
    log(f"Seed: {int(cfg['seed'])}")
    if cfg.get("duration") is not None:
        log(f"Audio duration: {cfg['duration']}s  ({int(cfg['sample_rate'] * cfg['duration'])} samples)")
    else:
        log("Audio duration: full (no truncation)")

    device = torch.device(cfg["device"])
    if device.type == "cuda":
        log(f"Using device: cuda  gpu: {torch.cuda.get_device_name(0)}")
    else:
        log("Using device: cpu")

    train_set = WADDataset(cfg["manifest"], cfg["train_split"], cfg)
    val_set   = WADDataset(cfg["manifest"], cfg["val_split"],   cfg)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2, collate_fn=collate_batch)
    val_loader   = DataLoader(val_set,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2, collate_fn=collate_batch)

    feat = XLSRFeature(cfg)
    with torch.no_grad():
        h = feat([load_audio(train_set.items[0][0], cfg)])
    model = XLSRConvLSTM(cfg, in_dim=h.shape[-1]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    bce = nn.BCEWithLogitsLoss()
    best_auc = 0.0

    cache_path = None
    if cfg.get("cache_dir"):
        os.makedirs(cfg["cache_dir"], exist_ok=True)
        cache_path = os.path.join(cfg["cache_dir"], f"xlsr_conv_lstm_{cfg['val_split']}.npz")

    if not args.eval_only:
        patience = int(cfg["early_stop_patience"])
        no_improve = 0
        for ep in range(cfg["num_epochs"]):
            model.train()
            total_loss, n = 0.0, 0
            t0 = time.time()
            for wavs, y in train_loader:
                h = feat(wavs).to(device)
                y = y.to(device)
                logit = model(h)
                loss = bce(logit, y)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item() * y.size(0)
                n += y.size(0)
                if n > 0 and (n // cfg["batch_size"]) % cfg["log_every"] == 0:
                    elapsed = time.time() - t0
                    it_s = n / max(elapsed, 1e-6)
                    remaining = (len(train_set) - n) / max(it_s, 1e-6)
                    log(f"[train] ep {ep+1} | {n}/{len(train_set)} | loss {loss.item():.4f} | {it_s:.2f} it/s | ETA {remaining/60:.1f} min")
            train_loss = total_loss / max(1, n)

            model.eval()
            all_y, all_s = [], []
            with torch.no_grad():
                for wavs, y in val_loader:
                    h = feat(wavs).to(device)
                    logit = model(h)
                    all_s.extend(torch.sigmoid(logit).cpu().tolist())
                    all_y.extend(y.tolist())
            try:
                auc = roc_auc_score(all_y, all_s)
            except Exception:
                auc = 0.5
            log(f"Epoch {ep+1}/{cfg['num_epochs']} Train loss {train_loss:.4f} Val AUROC {auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                no_improve = 0
                os.makedirs(os.path.dirname(cfg["save_path"]) or ".", exist_ok=True)
                torch.save({"cfg": cfg, "model": model.state_dict()}, cfg["save_path"])
                log(f"Saved to {cfg['save_path']}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    log(f"Early stopping: no improvement for {patience} epochs.")
                    break
    else:
        if not os.path.isfile(cfg["save_path"]):
            raise FileNotFoundError(f"save_path not found: {cfg['save_path']}")
        state = torch.load(cfg["save_path"], map_location=device)
        model.load_state_dict(state["model"])
        model.eval()

    if cache_path and os.path.isfile(cache_path):
        data = np.load(cache_path)
        all_s = data["scores"].tolist()
        all_y = data["labels"].tolist()
        log(f"Loaded cached scores from {cache_path}")
    else:
        model.eval()
        all_y, all_s = [], []
        with torch.no_grad():
            seen = 0
            t0 = time.time()
            for wavs, y in val_loader:
                h = feat(wavs).to(device)
                logit = model(h)
                all_s.extend(torch.sigmoid(logit).cpu().tolist())
                all_y.extend(y.tolist())
                seen += len(y)
                if seen > 0 and (seen // cfg["batch_size"]) % cfg["log_every_eval"] == 0:
                    elapsed = time.time() - t0
                    it_s = seen / max(elapsed, 1e-6)
                    remaining = (len(val_set) - seen) / max(it_s, 1e-6)
                    log(f"[eval] {seen}/{len(val_set)} | {it_s:.2f} it/s | ETA {remaining/60:.1f} min")
        if cache_path:
            np.savez_compressed(cache_path, scores=np.array(all_s), labels=np.array(all_y))
            log(f"Saved cached scores to {cache_path}")

    if cfg.get("fixed_thr") is not None:
        auroc = roc_auc_score(all_y, all_s)
        log(f"AUROC: {auroc:.4f}")
        best_thr = float(cfg["fixed_thr"])
        log(f"Threshold strategy: fixed  thr={best_thr:.4f}")
    elif cfg["thr_strategy"] == "f1_watermark":
        scores = np.array(all_s)
        labels = np.array(all_y)
        thresholds = np.linspace(0.0, 1.0, 201)
        best_thr, best_f1 = 0.5, -1.0
        for t in thresholds:
            pred = (scores >= t).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fp = np.sum((pred == 1) & (labels == 0))
            fn = np.sum((pred == 0) & (labels == 1))
            precision = tp / (tp + fp + 1e-12)
            recall    = tp / (tp + fn + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(t)
        log(f"Threshold strategy: f1_watermark  best_f1={best_f1:.4f}")
    else:
        auroc = roc_auc_score(all_y, all_s)
        log(f"AUROC: {auroc:.4f}")
        fpr, tpr, thr = roc_curve(all_y, all_s)
        best_idx = int(np.argmax(tpr - fpr))
        best_thr = float(thr[best_idx])
        log("Threshold strategy: tpr_fpr")

    pred = (np.array(all_s) >= best_thr).astype(int)
    log(f"Best thr: {best_thr:.4f}")
    log(classification_report(all_y, pred, target_names=["Clean", "Watermarked"], zero_division=0))

    if log_fh is not None:
        log_fh.close()


if __name__ == "__main__":
    main()
