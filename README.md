# X-AWMD: A Generalizable Framework for Black-Box Audio Watermark Detection under Method and Domain Shift

---

## Overview

X-AWMD detects whether an audio clip has been watermarked, without access to the watermarking algorithm or its parameters (black-box setting). It generalizes across different watermarking methods and acoustic domains.

**Architecture:** Frozen XLS-R (wav2vec2-xls-r-300m, layer 9) → Conv1D × 3 → BiLSTM × 2 → Attention Pooling → Binary classifier

---

## Repository Structure

```
.
├── train.py              # X-AWMD main model (training + evaluation)
├── split_dataset.py      # Dataset manifest generation
├── requirements.txt
├── baselines/
│   ├── wmd.py            # WMD baseline (ConvNeXtV2 + log-mel)
│   └── audiowmd.py       # AudioWMD baseline (SmallCNN + query statistics)
└── ablation/
    ├── xlsr_qs_lr.py         # XLS-R + Query-Statistics + Logistic Regression
    ├── wavlm_conv_lstm.py    # WavLM + Temporal Head (backbone ablation)
    ├── xlsr_conv_conformer.py # XLS-R + Conformer head
    ├── xlsr_conv_dprnn.py    # XLS-R + DPRNN head
    ├── xlsr_conv_transformer.py # XLS-R + Transformer head
    └── xlsr_aug.py           # Augmentation strategy ablation
```

---
## Requirements

```
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.36.0
librosa>=0.10.0
scikit-learn>=1.3.0
numpy>=1.24.0
```
---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+, PyTorch 2.0+, and a CUDA-capable GPU.

---

## Data Preparation

Prepare a CSV manifest with the following columns:

| Column | Description |
|--------|-------------|
| `split` | `train`, `validation`, or `test` |
| `derived_path` | Path to the audio file |
| `is_watermarked` | `1` = watermarked, `0` = clean |
| `perturbation` | Perturbation type applied (or `none`) |
| `watermark_method` | Name of the watermarking method |
| `orig_path` | Path to the original (pre-watermark) audio |
| `dataset` | Source dataset name |
| `sampling_rate_khz` | Sampling rate in kHz |

To generate the manifest automatically from LibriSpeech, CommonVoice, AISHELL, and VCTK:

```bash
python split_dataset.py \
  --librispeech /path/to/LibriSpeech \
  --commonvoice /path/to/CommonVoice \
  --aishell     /path/to/AISHELL \
  --vctk        /path/to/VCTK \
  --output      dataset_manifest.csv
```

---

## Training

### X-AWMD (main model)

```bash
python train.py --manifest dataset_manifest.csv
```

Key options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--manifest` | required | Path to dataset manifest CSV |
| `--duration` | `4.0` | Audio clip length in seconds (`None` = full audio) |
| `--save_path` | `./checkpoints/xawmd.pth` | Model save path |
| `--num_epochs` | `15` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--device` | auto | `cuda` or `cpu` |

### Evaluation only

```bash
python train.py --manifest dataset_manifest.csv --eval_only --save_path ./checkpoints/xawmd.pth
```

### JSON config

```bash
python train.py --config config.json
```

---

## Baselines

### WMD

ConvNeXtV2-based spectrogram classifier with asymmetric loss and iterative sample pruning.

```bash
python baselines/wmd.py --manifest dataset_manifest.csv
```

### AudioWMD

Two-stage: SmallCNN spectrogram classifier + query-statistics logistic regression meta-classifier.

```bash
python baselines/audiowmd.py --manifest dataset_manifest.csv
```

---

## Ablation Studies

### Table V — Architecture Variants

```bash
# Backbone: WavLM instead of XLS-R
python ablation/wavlm_conv_lstm.py --manifest dataset_manifest.csv

# Temporal head: Conformer
python ablation/xlsr_conv_conformer.py --manifest dataset_manifest.csv

# Temporal head: DPRNN
python ablation/xlsr_conv_dprnn.py --manifest dataset_manifest.csv

# Temporal head: Transformer
python ablation/xlsr_conv_transformer.py --manifest dataset_manifest.csv

# No neural head: query-statistics + logistic regression
python ablation/xlsr_qs_lr.py --manifest dataset_manifest.csv
```

### Table VI — Augmentation Strategy

```bash
python ablation/xlsr_aug.py --manifest dataset_manifest.csv --aug_mode none
python ablation/xlsr_aug.py --manifest dataset_manifest.csv --aug_mode asymmetric
python ablation/xlsr_aug.py --manifest dataset_manifest.csv --aug_mode specaug
python ablation/xlsr_aug.py --manifest dataset_manifest.csv --aug_mode mixup
python ablation/xlsr_aug.py --manifest dataset_manifest.csv --aug_mode codec_clean
python ablation/xlsr_aug.py --manifest dataset_manifest.csv --aug_mode combined
```
