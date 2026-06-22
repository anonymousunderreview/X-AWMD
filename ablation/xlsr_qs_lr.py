#!/usr/bin/env python3
"""
XLS-R + QS-LR ablation: frozen XLS-R -> layer pooling -> query-statistics -> logistic regression.

This is the "XLS-R + QS-LR" row in Table V of the X-AWMD paper. It uses the same
frozen XLS-R backbone as X-AWMD but replaces the temporal aggregation head with
static pooling followed by query-statistics logistic regression (the AudioWMD paradigm).

Usage:
    python ablation/xlsr_qs_lr.py --manifest /path/to/dataset_manifest.csv
"""
import os, json, time, argparse, random, pickle, hashlib, csv
from typing import List, Tuple

import numpy as np
import librosa
import torch
from sklearn.linear_model import LogisticRegression
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
    "queries": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_meta": "./checkpoints/xlsr_qs_lr.pkl",
    "cache_dir": None,
    "xlsr_model": "facebook/wav2vec2-xls-r-300m",
    "layer_idx": 9,
    "layer_range": None,
    "layer_fuse": "single",
    "pooling": "mean",
    "log_every": 200,
}


def load_audio(path, cfg):
    y, _ = librosa.load(path, sr=cfg["sample_rate"])
    if cfg.get("duration") is None:
        return y
    tgt = int(cfg["sample_rate"] * cfg["duration"])
    return np.pad(y, (0, max(0, tgt - len(y))))[:tgt]


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


class XLSRExtractor:
    def __init__(self, cfg):
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

    def _select_layer(self, hidden_states):
        if self.cfg["layer_fuse"] == "avg" and self.layer_range is not None:
            lo, hi = self.layer_range
            return torch.stack(hidden_states[lo:hi + 1], dim=0).mean(0)
        return hidden_states[int(self.cfg["layer_idx"])]

    @torch.no_grad()
    def extract(self, wavs: List[np.ndarray]) -> np.ndarray:
        inputs = self.processor(wavs, sampling_rate=self.cfg["sample_rate"],
                                return_tensors="pt", padding=True)
        out = self.model(inputs.input_values.to(self.device),
                         attention_mask=inputs.attention_mask.to(self.device),
                         output_hidden_states=True)
        h = self._select_layer(out.hidden_states)  # [B, T, D]
        if self.cfg["pooling"] == "mean_std":
            mean = h.mean(1)
            std = torch.sqrt(((h - mean[:, None, :]) ** 2).mean(1) + 1e-6)
            pooled = torch.cat([mean, std], dim=-1)
        else:
            pooled = h.mean(1)
        return pooled.cpu().numpy()


def extract_features(extractor, paths, cfg) -> Tuple[np.ndarray, np.ndarray]:
    meta = {}
    with open(cfg["manifest"], "r") as f:
        for row in csv.DictReader(f):
            meta[row["derived_path"]] = int(row.get("is_watermarked", 0))

    K = cfg["queries"]
    feats, labels = [], []
    total = len(paths)
    log_every = int(cfg.get("log_every", 200))
    t_start = time.time()

    for i, path in enumerate(paths):
        y0 = load_audio(path, cfg)
        q_vecs = []
        for k in range(K):
            yy = y0 if k == 0 else perturb_wave(y0, cfg)
            vec = extractor.extract([yy])[0]
            q_vecs.append(vec)
        q_vecs = np.stack(q_vecs, axis=0)  # [K, D]
        f_vec = np.concatenate([q_vecs.mean(0), q_vecs.std(0)], axis=0)
        feats.append(f_vec)
        labels.append(meta.get(path, 0))
        if (i + 1) % log_every == 0 or (i + 1) == total:
            rate = (i + 1) / max(time.time() - t_start, 1e-6)
            eta = (total - (i + 1)) / max(rate, 1e-6)
            print(f"[feat] {i+1}/{total} | {rate:.2f} it/s | ETA {eta/60:.1f} min", flush=True)

    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description="XLS-R + QS-LR ablation (Table V backbone section)")
    ap.add_argument("--config",      type=str,   default=None)
    ap.add_argument("--manifest",    type=str,   default=None, help="Path to dataset_manifest.csv (required)")
    ap.add_argument("--duration",    type=float, default=None, help="Audio clip length in seconds")
    ap.add_argument("--train_split", type=str,   default=None)
    ap.add_argument("--val_split",   type=str,   default=None)
    ap.add_argument("--queries",     type=int,   default=None)
    ap.add_argument("--xlsr_model",  type=str,   default=None)
    ap.add_argument("--layer_idx",   type=int,   default=None)
    ap.add_argument("--layer_range", type=str,   default=None)
    ap.add_argument("--layer_fuse",  type=str,   default=None)
    ap.add_argument("--pooling",     type=str,   default=None)
    ap.add_argument("--device",      type=str,   default=None)
    ap.add_argument("--save_meta",   type=str,   default=None)
    ap.add_argument("--cache_dir",   type=str,   default=None)
    ap.add_argument("--log_every",   type=int,   default=None)
    ap.add_argument("--eval_only",   action="store_true")
    args = ap.parse_args()

    cfg = CFG.copy()
    if args.config:
        with open(args.config, "r") as f:
            cfg.update(json.load(f))
    for k in ["manifest", "duration", "train_split", "val_split", "queries",
              "xlsr_model", "layer_idx", "layer_range", "layer_fuse", "pooling",
              "device", "save_meta", "cache_dir", "log_every"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    if cfg["manifest"] is None:
        ap.error("--manifest is required (or set 'manifest' in a --config JSON file)")

    train_paths, val_paths = [], []
    with open(cfg["manifest"], "r") as f:
        for row in csv.DictReader(f):
            p = row.get("derived_path")
            if not p or not os.path.isfile(p): continue
            if row.get("split") == cfg["train_split"]:
                train_paths.append(p)
            elif row.get("split") == cfg["val_split"]:
                val_paths.append(p)
    print(f"Train: {len(train_paths)}  Val: {len(val_paths)}")

    cache_path = None
    if cfg.get("cache_dir"):
        os.makedirs(cfg["cache_dir"], exist_ok=True)
        cache_key = {k: cfg[k] for k in ["xlsr_model", "layer_idx", "layer_range", "layer_fuse",
                                          "pooling", "queries", "sample_rate", "duration"]}
        tag = hashlib.md5(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:10]
        cache_path = os.path.join(cfg["cache_dir"], f"xlsr_qs_lr_{tag}.npz")

    extractor = XLSRExtractor(cfg)

    if args.eval_only:
        if cache_path and os.path.isfile(cache_path):
            data = np.load(cache_path)
            X_val, y_val = data["X_val"], data["y_val"]
        else:
            X_val, y_val = extract_features(extractor, val_paths, cfg)
        with open(cfg["save_meta"], "rb") as f:
            meta = pickle.load(f)
        clf = meta["clf"]
    else:
        if cache_path and os.path.isfile(cache_path):
            data = np.load(cache_path)
            X_train, y_train, X_val, y_val = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
            print(f"Loaded cached features from {cache_path}")
        else:
            t0 = time.time()
            X_train, y_train = extract_features(extractor, train_paths, cfg)
            X_val, y_val = extract_features(extractor, val_paths, cfg)
            print(f"Feature extraction: {(time.time()-t0)/60:.2f} min")
            if cache_path:
                np.savez_compressed(cache_path, X_train=X_train, y_train=y_train,
                                    X_val=X_val, y_val=y_val)
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
