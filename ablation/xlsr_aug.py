#!/usr/bin/env python3
"""
X-AWMD augmentation strategy ablation.

Controls the augmentation strategy via --aug_mode:
  none          no augmentation (default)
  asymmetric    clean: heavy waveform aug; watermarked: label-safe aug only
  specaug       time-masking on XLS-R hidden states (feature-space, label-safe)
  mixup         within-class mixup on XLS-R hidden states
  codec_clean   MP3 codec simulation on clean samples only
  combined      asymmetric + specaug

These correspond to the Watermark-Aware Augmentation rows in Table VI of the X-AWMD paper.

Usage:
    python ablation/xlsr_aug.py --manifest /path/to/dataset_manifest.csv --aug_mode specaug
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
import torchaudio.functional as AF

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
    "conv_channels": 256,
    "conv_layers": 3,
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "attn_dim": 128,
    "aug_mode": "none",
    "specaug_n_masks": 2,
    "specaug_mask_len": 40,
    "mixup_alpha": 0.2,
    "save_path": "./checkpoints/xlsr_aug_none.pth",
    "early_stop_patience": 3,
    "log_every": 50,
    "log_file": None,
    "thr_strategy": "tpr_fpr",
    "seed": 42,
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def heavy_aug(wav, sr):
    p = random.random()
    if p < 0.20:
        wav = np.roll(wav, random.randint(-int(0.05 * len(wav)), int(0.05 * len(wav))))
    elif p < 0.40:
        wav = librosa.effects.time_stretch(wav, rate=random.uniform(0.92, 1.08))
    elif p < 0.60:
        wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=random.uniform(-0.4, 0.4))
    elif p < 0.75:
        wav = wav + np.random.randn(len(wav)) * 0.005
    elif p < 0.90:
        wav = wav * random.uniform(0.9, 1.1)
    else:
        s = random.randint(0, len(wav) - 1)
        wav[s:s + int(0.01 * len(wav))] = 0
    return wav


def safe_aug(wav):
    if random.random() < 0.5:
        return wav * random.uniform(0.8, 1.2)
    return wav + np.random.randn(len(wav)) * 0.001


def codec_aug(wav, sr):
    compression = random.choice([7.0, 5.0, 3.0])
    try:
        wav_t = torch.from_numpy(wav.copy()).unsqueeze(0).float()
        coded = AF.apply_codec(wav_t, sr, format="mp3", compression=compression)
        out = coded.squeeze(0).numpy()
        tgt = len(wav)
        return np.pad(out, (0, max(0, tgt - len(out))))[:tgt]
    except Exception:
        return wav


def pad_or_trim(wav, tgt):
    return np.pad(wav, (0, max(0, tgt - len(wav))))[:tgt]


class WADDataset(Dataset):
    def __init__(self, manifest, split, cfg, is_train=False):
        import csv
        self.items, self.cfg, self.is_train = [], cfg, is_train
        with open(manifest) as f:
            for row in csv.DictReader(f):
                if row.get("split") != split: continue
                p = row.get("derived_path")
                if p and os.path.isfile(p):
                    self.items.append((p, int(row.get("is_watermarked", 0))))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        wav, _ = librosa.load(path, sr=self.cfg["sample_rate"])
        if self.cfg.get("duration") is not None:
            tgt = int(self.cfg["sample_rate"] * self.cfg["duration"])
            wav = pad_or_trim(wav, tgt)
        wav = wav.astype(np.float32)
        if self.is_train:
            wav = self._apply_waveform_aug(wav, label)
        return wav.astype(np.float32), label

    def _apply_waveform_aug(self, wav, label):
        mode = self.cfg["aug_mode"]
        sr = self.cfg["sample_rate"]
        tgt = int(sr * self.cfg["duration"]) if self.cfg.get("duration") is not None else len(wav)
        if mode in ("asymmetric", "combined"):
            if random.random() < 0.6:
                wav = heavy_aug(wav, sr) if label == 0 else safe_aug(wav)
            wav = pad_or_trim(wav, tgt)
        elif mode == "codec_clean":
            if label == 0 and random.random() < 0.5:
                wav = pad_or_trim(codec_aug(wav, sr), tgt)
        return wav


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

    @torch.no_grad()
    def forward(self, wavs):
        inputs = self.processor(wavs, sampling_rate=self.cfg["sample_rate"],
                                return_tensors="pt", padding=True)
        out = self.model(inputs.input_values.to(self.device),
                         attention_mask=inputs.attention_mask.to(self.device),
                         output_hidden_states=True)
        return out.hidden_states[int(self.cfg["layer_idx"])]


class ConvBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU())

    def forward(self, x): return self.net(x)


class TemporalHead(nn.Module):
    def __init__(self, cfg, in_dim):
        super().__init__()
        self.cfg = cfg
        c = cfg["conv_channels"]
        self.proj = nn.Conv1d(in_dim, c, 1)
        self.blocks = nn.Sequential(*[ConvBlock1D(c) for _ in range(cfg["conv_layers"])])
        self.lstm = nn.LSTM(c, cfg["lstm_hidden"], cfg["lstm_layers"],
                            batch_first=True, bidirectional=True)
        h2 = cfg["lstm_hidden"] * 2
        self.attn = nn.Sequential(nn.Linear(h2, cfg["attn_dim"]), nn.Tanh(), nn.Linear(cfg["attn_dim"], 1))
        self.head = nn.Linear(h2, 1)

    def apply_specaug(self, h):
        B, T, D = h.shape
        for b in range(B):
            for _ in range(int(self.cfg["specaug_n_masks"])):
                t0 = random.randint(0, max(0, T - 1))
                tlen = random.randint(5, max(5, min(int(self.cfg["specaug_mask_len"]), T - t0)))
                h[b, t0:t0 + tlen, :] = 0.0
        return h

    def forward(self, h):
        if self.training and self.cfg.get("aug_mode", "none") in ("specaug", "combined"):
            h = self.apply_specaug(h)
        x = self.proj(h.transpose(1, 2))
        x = self.blocks(x).transpose(1, 2)
        x, _ = self.lstm(x)
        a = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        return self.head((x * a.unsqueeze(-1)).sum(1)).squeeze(-1)


def within_class_mixup(h, y, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))
    h_mixed = h.clone()
    for cls in [0, 1]:
        idx = (y == cls).nonzero(as_tuple=True)[0]
        if len(idx) < 2: continue
        perm = idx[torch.randperm(len(idx), device=h.device)]
        h_mixed[idx] = lam * h[idx] + (1.0 - lam) * h[perm]
    return h_mixed


def main():
    ap = argparse.ArgumentParser(description="Augmentation ablation (Table VI)")
    ap.add_argument("--config", type=str, default=None)
    for k in ["manifest", "train_split", "val_split", "save_path", "xlsr_model",
              "device", "log_file", "thr_strategy", "aug_mode"]:
        ap.add_argument(f"--{k}", type=str, default=None)
    for k in ["batch_size", "num_epochs", "log_every", "early_stop_patience",
              "layer_idx", "conv_layers", "conv_channels", "lstm_hidden",
              "lstm_layers", "seed", "specaug_n_masks", "specaug_mask_len"]:
        ap.add_argument(f"--{k}", type=int, default=None)
    for k in ["lr", "mixup_alpha", "duration"]:
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
    log(f"aug_mode={cfg['aug_mode']} | lr={cfg['lr']} | duration={cfg['duration']}s")

    train_set = WADDataset(cfg["manifest"], cfg["train_split"], cfg, is_train=True)
    val_set   = WADDataset(cfg["manifest"], cfg["val_split"],   cfg, is_train=False)
    train_loader = DataLoader(train_set, cfg["batch_size"], shuffle=True,  num_workers=2, collate_fn=collate_batch)
    val_loader   = DataLoader(val_set,   cfg["batch_size"], shuffle=False, num_workers=2, collate_fn=collate_batch)

    feat  = XLSRFeature(cfg)
    with torch.no_grad():
        _h = feat([train_set[0][0]])
    model = TemporalHead(cfg, in_dim=_h.shape[-1]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    bce   = nn.BCEWithLogitsLoss()

    do_mixup = cfg["aug_mode"] == "mixup"
    best_auc, no_improve = 0.0, 0

    if not args.eval_only:
        for ep in range(cfg["num_epochs"]):
            model.train()
            total_loss, n = 0.0, 0
            for wavs, y in train_loader:
                h = feat(wavs).to(device)
                y = y.to(device)
                if do_mixup:
                    h = within_class_mixup(h, y, alpha=cfg["mixup_alpha"])
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
