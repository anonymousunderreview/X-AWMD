#!/usr/bin/env python3
"""
X-AWMD Dataset Manifest Builder

Generates dataset_manifest.csv from clean audio roots and their pre-watermarked counterparts.
Expected directory layout per corpus:

    <root>/
        clean/          ← original .wav files (collected recursively)
        AudioSeal/      ← watermarked copies, same filenames
        Timbre/
        ...             ← one sub-directory per watermark method

Usage:
    python split_dataset.py \
        --librispeech /data/LibriSpeech/clean \
        --commonvoice /data/CommonVoice/wavs \
        --aishell      /data/AISHELL/wavs \
        --vctk         /data/VCTK/wavs \
        --output       dataset_manifest.csv
"""
import os, csv, random, argparse
from glob import glob
from collections import defaultdict

TRAIN_CLEAN_COUNT = 34000
VAL_RATIO         = 0.10
TEST1_COUNT       = 16000
TEST2_COUNT       = 10000

TRAIN_WM_METHODS = ["LSB", "QIM", "DSSS", "AudioSeal", "Timbre", "PhaseCoding"]
TEST_WM_METHODS  = ["Patchwork", "EchoHiding", "WavMark", "Perth"]

CHINESE_LOCALES = {"zh-CN", "zh-HK", "zh-TW", "yue"}
ENGLISH_LOCALES = {"en"}


def collect_files(root):
    audio_files = []
    for r, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(".wav"):
                audio_files.append(os.path.join(r, n))
    return sorted(audio_files)


def parse_commonvoice_lang(path):
    parts = os.path.basename(path).split("_")
    if len(parts) < 2:
        return "unknown"
    locale = parts[1]
    if locale in CHINESE_LOCALES or locale in ENGLISH_LOCALES:
        return "en_or_cn"
    if "-" in locale or locale.isalpha():
        return "other"
    return "unknown"


def pick_n(files, n):
    if len(files) >= n:
        return files[:n]
    print(f"Warning: requested {n} but only {len(files)} available")
    return files


def assign_one_wm_per_file(file_list, wm_methods):
    n = len(file_list)
    per = n // len(wm_methods)
    remainder = n % len(wm_methods)
    wm_assign = []
    for i, m in enumerate(wm_methods):
        count = per + (1 if i < remainder else 0)
        wm_assign += [m] * count
    random.shuffle(wm_assign)
    return dict(zip(file_list, wm_assign))


def find_watermarked_file(clean_path, wm_method):
    clean_dir  = os.path.dirname(clean_path)
    clean_name = os.path.splitext(os.path.basename(clean_path))[0]
    wm_dir = os.path.join(os.path.dirname(clean_dir), wm_method)
    if not os.path.exists(wm_dir):
        print(f"Warning: watermark dir not found: {wm_dir}")
        return None
    matches = glob(os.path.join(wm_dir, f"*{clean_name}*"))
    if matches:
        return matches[0]
    print(f"Warning: no watermarked file for {clean_path} in {wm_method}")
    return None


def add_split(pool, split, wm_methods, manifest):
    wm_assign = assign_one_wm_per_file(pool, wm_methods)
    for clean_file in pool:
        wm_method = wm_assign[clean_file]
        manifest.append({
            "orig_path": clean_file, "split": split, "dataset": split,
            "sampling_rate_khz": 16, "is_watermarked": 0,
            "watermark_method": "", "perturbation": "", "derived_path": clean_file,
        })
        wm_path = find_watermarked_file(clean_file, wm_method)
        if wm_path:
            manifest.append({
                "orig_path": clean_file, "split": split, "dataset": split,
                "sampling_rate_khz": 16, "is_watermarked": 1,
                "watermark_method": wm_method, "perturbation": "", "derived_path": wm_path,
            })
        else:
            print(f"Warning: skipping watermarked entry for {clean_file}")


def main():
    ap = argparse.ArgumentParser(description="Build X-AWMD dataset manifest")
    ap.add_argument("--librispeech", required=True, help="Path to LibriSpeech clean wav root")
    ap.add_argument("--commonvoice", required=True, help="Path to CommonVoice wav root")
    ap.add_argument("--aishell",     required=True, help="Path to AISHELL clean wav root")
    ap.add_argument("--vctk",        required=True, help="Path to VCTK clean wav root")
    ap.add_argument("--output",      default="dataset_manifest.csv", help="Output CSV path")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print("Collecting datasets...")
    librispeech_files = collect_files(args.librispeech)
    cv_files          = collect_files(args.commonvoice)
    aishell_files     = collect_files(args.aishell)
    vctk_files        = collect_files(args.vctk)

    print(f" LibriSpeech : {len(librispeech_files)} files")
    print(f" CommonVoice : {len(cv_files)} files")
    print(f" AISHELL     : {len(aishell_files)} files")
    print(f" VCTK        : {len(vctk_files)} files")

    cv_en_cn = [f for f in cv_files if parse_commonvoice_lang(f) == "en_or_cn"]
    cv_other = [f for f in cv_files if parse_commonvoice_lang(f) == "other"]
    print(f" CommonVoice En/Zh: {len(cv_en_cn)}, other: {len(cv_other)}")

    train_pool = pick_n(librispeech_files, 20000) + pick_n(cv_en_cn, 4000) + pick_n(aishell_files, 10000)
    random.shuffle(train_pool)
    train_pool = train_pool[:TRAIN_CLEAN_COUNT]

    val_count  = int(len(train_pool) * VAL_RATIO)
    val_pool   = train_pool[:val_count]
    train_pool = train_pool[val_count:]

    test1_pool = pick_n(cv_other,    TEST1_COUNT)
    test2_pool = pick_n(vctk_files,  TEST2_COUNT)

    print(f"Train={len(train_pool)}, Val={len(val_pool)}, Test1={len(test1_pool)}, Test2={len(test2_pool)}")

    fields = ["orig_path", "split", "dataset", "sampling_rate_khz",
              "is_watermarked", "watermark_method", "perturbation", "derived_path"]
    manifest = []

    add_split(train_pool, "train",      TRAIN_WM_METHODS, manifest)
    add_split(val_pool,   "validation", TRAIN_WM_METHODS, manifest)
    add_split(test1_pool, "test1_in",   TEST_WM_METHODS,  manifest)
    add_split(test2_pool, "test2_in",   TEST_WM_METHODS,  manifest)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in manifest:
            for field in fields:
                row.setdefault(field, "")
            writer.writerow(row)

    print(f"Manifest saved: {args.output}  ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
