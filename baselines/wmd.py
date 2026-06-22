#!/usr/bin/env python3
"""
WMD baseline: ConvNeXtV2 spectrogram detector with asymmetric loss and iterative pruning.

Re-implementation of:
  Fernandez et al., "Finding a Needle in a Haystack: A Black-Box Approach to
  Invisible Watermark Detection," ECCV 2024.

Only intentional adaptation:
- Data reading uses the X-AWMD manifest CSV schema (split, derived_path, is_watermarked).
- Audio input is log-mel spectrogram; backbone is ConvNeXtV2-style with 5 blocks.

Usage:
    python baselines/wmd.py --manifest /path/to/dataset_manifest.csv
    python baselines/wmd.py --manifest /path/to/dataset_manifest.csv --eval_only --ckpt_path ./checkpoints/wmd.pth
"""
import os, csv, json, time, math, argparse, random
from typing import List, Tuple

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

CFG = {
    "manifest": None,
    "det_split": "train",
    "clean_split": "train",
    "val_split": "validation",
    "sample_rate": 16000,
    "duration": 4.0,                        # audio clip length in seconds
    "n_fft": 1024,
    "hop": 320,
    "n_mels": 128,
    "fixed_frames": 160,
    "batch_size": 32,
    "num_epochs": 50,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_dim": 96,
    "tau": 1.0,
    "pruning_rate": 0.10,
    "pruning_interval": 10,
    "log_every": 50,
    "log_file": None,
    "save_path": "./checkpoints/wmd.pth",
    "thr_strategy": "tpr_fpr",
    "fixed_thr": 0.5,
}

BLACKBOX_PERTURBATIONS = {"HSJA_signal", "HSJA_spectrogram", "square"}


def load_audio(path, cfg):
    y, _ = librosa.load(path, sr=cfg["sample_rate"])
    if cfg.get("duration") is None:
        return y
    tgt = int(cfg["sample_rate"] * cfg["duration"])
    return np.pad(y, (0, max(0, tgt - len(y))))[:tgt]


def wav_to_logmel(y, cfg):
    m = librosa.feature.melspectrogram(
        y=y, sr=cfg["sample_rate"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop"], n_mels=cfg["n_mels"],
        fmin=0, fmax=cfg["sample_rate"] // 2,
    )
    m = librosa.power_to_db(m + 1e-10, ref=np.max)
    m = (m - m.mean()) / (m.std() + 1e-6)
    t = int(cfg["fixed_frames"])
    if m.shape[1] < t:
        m = np.pad(m, ((0, 0), (0, t - m.shape[1])))
    else:
        m = m[:, :t]
    return m.astype(np.float32)


class AudioItems:
    def __init__(self, manifest: str, split: str):
        items = []
        with open(manifest, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("split") != split:
                    continue
                p = row.get("derived_path")
                if p and os.path.isfile(p):
                    items.append((p, int(row.get("is_watermarked", 0))))
        self.items = items


class DetectionSubset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], cfg, indices: List[int]):
        self.items, self.cfg, self.indices = items, cfg, list(indices)

    def set_indices(self, indices):
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        p, y = self.items[j]
        mel = wav_to_logmel(load_audio(p, self.cfg), self.cfg)
        return torch.tensor(mel[None, ...], dtype=torch.float32), float(y), j


class CleanDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], cfg):
        self.items = [(p, y) for p, y in items if y == 0]
        self.cfg = cfg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        mel = wav_to_logmel(load_audio(p, self.cfg), self.cfg)
        return torch.tensor(mel[None, ...], dtype=torch.float32), float(y)


def collate_det(batch):
    xs, ys, ids = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32), torch.tensor(ids, dtype=torch.long)


def collate_clean(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.act(self.pwconv1(x))
        x = self.grn(x)
        x = self.pwconv2(x)
        return x.permute(0, 3, 1, 2) + residual


class ConvNeXtV2Detector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = int(cfg["model_dim"])
        self.stem = nn.Sequential(nn.Conv2d(1, d, kernel_size=4, stride=4), nn.GELU())
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(d) for _ in range(5)])
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = self.norm(x.mean(dim=(2, 3)))
        return self.head(x).squeeze(-1)


def init_detector(cfg):
    model = ConvNeXtV2Detector(cfg).to(torch.device(cfg["device"]))
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        weight_decay=cfg["weight_decay"],
    )
    return model, opt


def score_detection_set(det_items, active_indices, cfg, model):
    ds = DetectionSubset(det_items, cfg, active_indices)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2, collate_fn=collate_det)
    scores = {}
    model.eval()
    device = torch.device(cfg["device"])
    with torch.no_grad():
        for xs, _, ids in dl:
            s = model(xs.to(device)).cpu().numpy().tolist()
            for j, v in zip(ids.tolist(), s):
                scores[int(j)] = float(v)
    return scores


def asymmetric_loss(pc, pd, tau):
    l_sm = tau * torch.logsumexp(pc / tau, dim=0)
    l_lin = -pd.mean()
    return l_sm + l_lin, l_sm, l_lin


def run_eval(model, cfg, log):
    device = torch.device(cfg["device"])

    def load_rows(split_name):
        rows = []
        with open(cfg["manifest"], "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("split") != split_name:
                    continue
                p = row.get("derived_path")
                if p and os.path.isfile(p):
                    rows.append({"path": p, "label": int(row.get("is_watermarked", 0)),
                                 "perturbation": row.get("perturbation", "")})
        return rows

    val_rows = load_rows(cfg["val_split"])
    eval_rows = [r for r in val_rows if r["perturbation"] in BLACKBOX_PERTURBATIONS]
    if not eval_rows:
        raise RuntimeError(f"No black-box rows in split={cfg['val_split']}")

    eval_items = [(r["path"], r["label"]) for r in eval_rows]
    eval_ds = DetectionSubset(eval_items, cfg, list(range(len(eval_items))))
    eval_dl = DataLoader(eval_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2, collate_fn=collate_det)

    model.eval()
    all_s, all_y = [], []
    with torch.no_grad():
        for xs, ys, _ in eval_dl:
            all_s.extend(model(xs.to(device)).cpu().numpy().tolist())
            all_y.extend(ys.numpy().tolist())

    auroc = roc_auc_score(all_y, all_s) if len(np.unique(all_y)) >= 2 else float("nan")
    log(f"[eval:{cfg['val_split']}] black-box AUROC: {auroc:.4f}")

    scores, labels = np.array(all_s, dtype=np.float32), np.array(all_y, dtype=np.int64)
    if cfg["thr_strategy"] == "fixed":
        best_thr = float(cfg["fixed_thr"])
    else:
        fpr, tpr, thr = roc_curve(labels, scores)
        best_thr = float(thr[int(np.argmax(tpr - fpr))])
    pred = (scores >= best_thr).astype(int)
    log(f"[eval] best_thr={best_thr:.4f}")
    log(classification_report(labels, pred, target_names=["Clean", "Watermarked"], zero_division=0))


def main():
    ap = argparse.ArgumentParser(description="Train or evaluate WMD baseline")
    ap.add_argument("--config",           type=str,   default=None)
    ap.add_argument("--manifest",         type=str,   default=None, help="Path to dataset_manifest.csv (required)")
    ap.add_argument("--duration",         type=float, default=None, help="Audio clip length in seconds")
    ap.add_argument("--det_split",        type=str,   default=None)
    ap.add_argument("--clean_split",      type=str,   default=None)
    ap.add_argument("--val_split",        type=str,   default=None)
    ap.add_argument("--batch_size",       type=int,   default=None)
    ap.add_argument("--num_epochs",       type=int,   default=None)
    ap.add_argument("--lr",               type=float, default=None)
    ap.add_argument("--weight_decay",     type=float, default=None)
    ap.add_argument("--model_dim",        type=int,   default=None)
    ap.add_argument("--device",           type=str,   default=None)
    ap.add_argument("--save_path",        type=str,   default=None)
    ap.add_argument("--tau",              type=float, default=None)
    ap.add_argument("--pruning_rate",     type=float, default=None)
    ap.add_argument("--pruning_interval", type=int,   default=None)
    ap.add_argument("--thr_strategy",     type=str,   default=None, choices=["fixed", "tpr_fpr", "f1_watermark"])
    ap.add_argument("--fixed_thr",        type=float, default=None)
    ap.add_argument("--log_every",        type=int,   default=None)
    ap.add_argument("--log_file",         type=str,   default=None)
    ap.add_argument("--eval_only",        action="store_true")
    ap.add_argument("--ckpt_path",        type=str,   default=None)
    args = ap.parse_args()

    cfg = CFG.copy()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    for k in ["manifest", "duration", "det_split", "clean_split", "val_split",
              "batch_size", "num_epochs", "lr", "weight_decay", "model_dim",
              "device", "save_path", "tau", "pruning_rate", "pruning_interval",
              "thr_strategy", "fixed_thr", "log_every", "log_file"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    if cfg["manifest"] is None:
        ap.error("--manifest is required (or set 'manifest' in a --config JSON file)")

    log_fh = None
    if cfg.get("log_file"):
        os.makedirs(os.path.dirname(cfg["log_file"]) or ".", exist_ok=True)
        log_fh = open(cfg["log_file"], "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        if log_fh:
            log_fh.write(msg + "\n"); log_fh.flush()

    device = torch.device(cfg["device"])
    log(f"Device: {device}")
    log(f"Audio duration: {cfg['duration']}s")

    if args.eval_only:
        ckpt_path = args.ckpt_path or cfg["save_path"]
        model, _ = init_detector(cfg)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))
        log(f"Loaded: {ckpt_path}")
        run_eval(model, cfg, log)
        if log_fh: log_fh.close()
        return

    det_items = AudioItems(cfg["manifest"], cfg["det_split"]).items
    clean_items = [(p, y) for p, y in AudioItems(cfg["manifest"], cfg["clean_split"]).items if y == 0]
    log(f"Detection samples: {len(det_items)}  Clean samples: {len(clean_items)}")

    model, opt = init_detector(cfg)
    active_indices = list(range(len(det_items)))
    det_ds = DetectionSubset(det_items, cfg, active_indices)
    clean_ds = CleanDataset(clean_items, cfg)

    prune_steps = 0
    t_start = time.time()

    for ep in range(1, int(cfg["num_epochs"]) + 1):
        model.train()
        det_loader = DataLoader(det_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, collate_fn=collate_det)
        clean_loader = DataLoader(clean_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, collate_fn=collate_clean)
        det_iter, clean_iter = iter(det_loader), iter(clean_loader)
        n_steps = max(len(det_loader), len(clean_loader))
        run_loss = 0.0

        for step in range(1, n_steps + 1):
            try: xs_d, _, _ = next(det_iter)
            except StopIteration: det_iter = iter(det_loader); xs_d, _, _ = next(det_iter)
            try: xs_c, _ = next(clean_iter)
            except StopIteration: clean_iter = iter(clean_loader); xs_c, _ = next(clean_iter)

            pd = model(xs_d.to(device))
            pc = model(xs_c.to(device))
            loss, lsm, llin = asymmetric_loss(pc, pd, float(cfg["tau"]))
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss.item())
            if step % max(1, int(cfg["log_every"])) == 0:
                log(f"[train] ep {ep} step {step}/{n_steps} loss {loss.item():.4f}")

        log(f"[train] ep {ep} avg_loss {run_loss/max(1,n_steps):.4f} active_det {len(active_indices)}")

        if ep % int(cfg["pruning_interval"]) == 0 and ep < int(cfg["num_epochs"]):
            prune_steps += 1
            rank_scores = score_detection_set(det_items, list(range(len(det_items))), cfg, model)
            keep_ratio = (1.0 - float(cfg["pruning_rate"])) ** prune_steps
            keep_n = max(1, int(math.floor(len(det_items) * keep_ratio)))
            ranked = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
            active_indices = [idx for idx, _ in ranked[:keep_n]]
            det_ds.set_indices(active_indices)
            log(f"[prune] ep {ep} keep_ratio {keep_ratio:.4f} keep_n {keep_n}/{len(det_items)}")
            model, opt = init_detector(cfg)

    os.makedirs(os.path.dirname(cfg["save_path"]) or ".", exist_ok=True)
    torch.save({"cfg": cfg, "model": model.state_dict(), "selected_indices": active_indices}, cfg["save_path"])
    log(f"Saved: {cfg['save_path']}")

    run_eval(model, cfg, log)
    log(f"Total time: {(time.time()-t_start)/60:.2f} min")
    if log_fh: log_fh.close()


if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    main()
