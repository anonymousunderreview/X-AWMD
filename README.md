# X-AWMD: A Generalizable Framework for Black-Box Audio Watermark Detection under Method and Domain Shift

Official implementation of:

> **X-AWMD: A Generalizable Framework for Black-Box Audio Watermark Detection under Method and Domain Shift**

X-AWMD detects whether an audio clip contains an audio watermark in a black-box setting. The detector does not use the embedding algorithm, secret key, watermark decoder, or reference clean audio at inference time. The paper evaluates generalization to unseen watermarking methods and out-of-domain speech.

## Method

The proposed model uses signal-level STFT features rather than a frozen speech representation backbone:

```text
waveform at 16 kHz
  -> STFT
  -> 3-channel feature map:
       log(1 + |S|)
       cos(phi_t - phi_{t-1})
       sin(phi_t - phi_{t-1})
  -> 3-block 2D CNN: 32, 64, 128 channels
  -> global average pooling + linear projection
  -> binary watermark score
```

Training uses binary cross-entropy plus pair ranking over matched clean/watermarked clips from the same recording:

```text
L = L_BCE + lambda * ReLU(margin - (score_watermarked - score_clean))
```

Default release settings match the paper's proposed model: `features=stft_mag_phase`, `pair_weight=0.5`, `pair_margin=0.25`, 3-second clips, 512-point FFT, 400-sample Hann window, 160-sample hop, AdamW with learning rate `1e-4`, and early stopping on validation AUROC.

## Repository Structure

```text
.
├── train.py                  # X-AWMD main model: STFT + phase + pair ranking
├── run_multiseed.py          # Validation-threshold multi-seed evaluation runner
├── split_dataset.py          # Dataset manifest generation helper
├── requirements.txt
├── baselines/
│   ├── wmd.py                # WMD spectrogram baseline
│   └── audiowmd.py           # AudioWMD query-statistics baseline
└── ablation/
    ├── xlsr_qs_lr.py
    ├── wavlm_conv_lstm.py
    ├── xlsr_conv_conformer.py
    ├── xlsr_conv_dprnn.py
    ├── xlsr_conv_transformer.py
    └── xlsr_aug.py
```

The `ablation/` scripts are kept for legacy backbone/head comparisons. The main paper result is implemented by `train.py`.

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.8+, PyTorch 2.0+, and CUDA GPUs.

## Manifest Format

Training and evaluation use a CSV manifest. Required columns:

| Column | Description |
| --- | --- |
| `split` | Dataset split, e.g. `train`, `validation`, `test0_in`, `test1_in`, `test2_in` |
| `derived_path` | Path to the audio file |
| `is_watermarked` | `1` for watermarked audio, `0` for clean audio |
| `watermark_method` | Watermarking method name; may be empty for clean rows |
| `orig_path` | Original recording path or stable recording id used to pair clean/watermarked samples |

Optional columns such as `dataset`, `language`, `perturbation`, and `sampling_rate_khz` are preserved by the data builder and baseline scripts when available.

The proposed pair-ranking loss requires each watermarked training row to have a matching clean row with the same `orig_path` in the training split.

## Training

Train the proposed X-AWMD model:

```bash
python train.py \
  --manifest /path/to/dataset_manifest.csv \
  --save_path ./checkpoints/xawmd_stft_phase_pair.pth
```

Useful options:

| Argument | Default | Description |
| --- | --- | --- |
| `--features` | `stft_mag_phase` | Feature branch. Use `stft_mag_phase` for the proposed model |
| `--duration` | `3.0` | Clip duration in seconds |
| `--pair_weight` | `0.5` | Pair-ranking loss weight |
| `--pair_margin` | `0.25` | Required score margin between paired watermarked and clean clips |
| `--batch_size` | `3` | Number of clean/watermarked pairs per batch |
| `--num_epochs` | `15` | Maximum training epochs |
| `--lr` | `1e-4` | AdamW learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--device` | auto | `cuda` or `cpu` |

Evaluate a saved checkpoint on a split:

```bash
python train.py \
  --manifest /path/to/dataset_manifest.csv \
  --eval_only \
  --val_split test1_in \
  --save_path ./checkpoints/xawmd_stft_phase_pair.pth \
  --fixed_thr 0.189
```

If `--fixed_thr` is omitted, the script selects a threshold on the evaluation split using Youden's J. For deployment-style reporting, select the threshold on validation and reuse it for Test0/Test1/Test2.

Run multi-seed training and validation-threshold evaluation:

```bash
python run_multiseed.py \
  --manifest /path/to/dataset_manifest.csv \
  --out_dir experiments/multiseed_xawmd \
  --seeds 42 2024
```

## Paper Results

Main black-box OOD comparison. Threshold-dependent F1 uses a single validation-calibrated threshold.

| Method | Val AUROC | Val F1 | Test1 AUROC | Test1 F1 | Test2 AUROC | Test2 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WMD | 0.7201 | 0.58 | 0.5709 | 0.37 | 0.5794 | 0.49 |
| AudioWMD | 0.8830 | 0.82 | 0.6382 | 0.16 | 0.6317 | 0.43 |
| X-AWMD | 0.9154 | 0.86 | 0.9108 | 0.45 | 0.8776 | 0.48 |

Method-shift isolation for X-AWMD:

| Split | Overall AUROC | EchoHiding AUROC | Patchwork AUROC | WavMark AUROC | Recall-W | F1-W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.9154 | - | - | - | 0.80 | 0.8628 |
| Test0 | 0.7977 | 0.9575 | 0.5433 | 0.8923 | 0.48 | 0.6241 |
| Test1 | 0.9108 | 0.8588 | 0.9327 | 0.9409 | 0.29 | 0.4454 |
| Test2 | 0.8776 | 0.9446 | 0.7326 | 0.9558 | 0.32 | 0.4835 |

Design ablation:

| Setting | Val AUROC | Test0 AUROC | Test1 AUROC | Test2 AUROC |
| --- | ---: | ---: | ---: | ---: |
| XLS-R + BCE | 0.8929 | 0.6653 | 0.7477 | 0.5872 |
| STFT + phase + BCE | 0.9161 | 0.7553 | 0.8610 | 0.7913 |
| STFT + phase + pair ranking | 0.9154 | 0.7977 | 0.9108 | 0.8776 |
| + method-centroid invariance | 0.9145 | 0.8042 | 0.8954 | 0.8498 |
| + GroupDRO | 0.8991 | 0.6956 | 0.8866 | 0.7137 |

## Baselines

Train the WMD and AudioWMD baselines under the same manifest protocol:

```bash
python baselines/wmd.py --manifest /path/to/dataset_manifest.csv
python baselines/audiowmd.py --manifest /path/to/dataset_manifest.csv
```

## Ablations

The main ablation settings can be reproduced with `train.py`:

```bash
# XLS-R + BCE
python train.py --manifest /path/to/dataset_manifest.csv \
  --features xlsr --pair_weight 0.0 --save_path ./checkpoints/A_xlsr_bce.pth

# STFT + phase + BCE
python train.py --manifest /path/to/dataset_manifest.csv \
  --features stft_mag_phase --pair_weight 0.0 --save_path ./checkpoints/B_stft_phase_bce.pth

# Proposed: STFT + phase + pair ranking
python train.py --manifest /path/to/dataset_manifest.csv \
  --features stft_mag_phase --pair_weight 0.5 --pair_margin 0.25 \
  --save_path ./checkpoints/C_stft_phase_pair.pth

# Method-centroid invariance
python train.py --manifest /path/to/dataset_manifest.csv \
  --features stft_mag_phase --pair_weight 0.5 --invariant_weight 0.05 \
  --save_path ./checkpoints/D_stft_phase_pair_inv.pth

# GroupDRO
python train.py --manifest /path/to/dataset_manifest.csv \
  --features stft_mag_phase --pair_weight 0.5 --groupdro \
  --save_path ./checkpoints/E_stft_phase_groupdro.pth
```

## Notes

X-AWMD preserves strong score ranking under unseen watermark methods and domain shift, but a validation-calibrated threshold can miscalibrate on OOD splits. For this reason, the paper reports AUROC as the primary metric and uses fixed-threshold F1/recall as deployment operating-point diagnostics.
