# REEM: SCR-Guided Difficulty-Aware Optimization for Infrared Small Target Detection

---

## Overview

REEM introduces an SCR (Signal-to-Clutter Ratio) guided difficulty-aware reward mechanism on top of the MSHNet backbone. Low-SCR targets (harder to detect) receive higher training weight via a soft IoU reward term, improving detection performance without architectural changes.

Training code, evaluation code, and pretrained weights are all available below.

---

## Results

### IRSTD-1k

| Method | IoU↑ | PD↑ | FA↓ |
|---|---|---|---|
| MSHNet (baseline) | 65.60 | 93.20 | 13.51 |
| **REEM (ours)** | **68.44** | **93.88** | **6.30** |

### NUDT-SIRST

| Method | IoU↑ | PD↑ | FA↓ |
|---|---|---|---|
| MSHNet (baseline) | 74.52 | 95.37 | 29.00 |
| **REEM (ours)** | **79.86** | **97.52** | **11.21** |

---

## Training

`Reem_Training.py` reproduces the repo-exact MSHNet training pipeline (same augmentations, ImageNet normalization, repo-style mIoU/PD_FA metrics) and adds the SCR-guided reward loss on top of `SLSIoULoss`.

```bash
python Reem_Training.py \
  --dataset-dir /path/to/dataset \
  --epochs 300 --runs 10 --fixed-split --split-seed 0 \
  --base-size 256 --crop-size 256 --batch-size 24 --workers 2 \
  --warm-epoch 5 --lr 0.05 --bins 10 --blob-dist 3.0 \
  --lambda-grid <comma-separated list of length --runs> \
  --save-dir ./weight/REEM \
  --summary-csv ./weight/REEM/summary.csv \
  --target-fa-ppm 15.0
```

### Choosing `λ`

`λ` controls the strength of the SCR-guided reward term (`L = L_SLS - λ * reward`) and its optimal value depends on the dataset. We recommend sweeping a small range of `λ` values across multiple runs and selecting the value that yields the best **validation** mIoU for your dataset, then reporting that run's **test** metrics. `--runs` and `--lambda-grid` make this straightforward — set `--lambda-grid` to a comma-separated list (length equal to `--runs`) covering the range you want to explore, e.g. low/mid/high values repeated a few times each for stability.

Each run is saved under `<save-dir>/runXX_seedYY_splitZZ_lambdaW/`, with `weight.pkl` (best val mIoU checkpoint), `checkpoint.pkl` (optimizer state), and `metric.log`. The `--summary-csv` output ranks all runs by validation mIoU, making it easy to identify the best `λ` for the dataset at hand.

Key arguments:

| Argument | Meaning |
|---|---|
| `--runs` / `--lambda-grid` | Number of training runs and the SCR-reward weight `λ` used in each |
| `--fixed-split` / `--split-seed` | Keep the train/val split identical across runs |
| `--warm-epoch` | Epochs trained with plain IoU loss before SLS/shape and SCR-reward terms are enabled |
| `--target-fa-ppm` | Operating point on the test PD/FA curve is selected at the threshold closest to this FA (ppm) |

---

## Pretrained Weights

| Dataset | Download |
|---|---|
| IRSTD-1k | [REEM_IRSTD1K](https://github.com/yall-in-one/Reemm/releases/tag/REEM_IRSTD1K) |
| NUDT-SIRST | [REEM_NUDT](https://github.com/yall-in-one/Reemm/releases/tag/REEM_NUDT) |

Download `weight.pkl` from the release and place it under `weights/` before running evaluation.

---

## Evaluation

### IRSTD-1k

```bash
python Reem_results.py \
  --dataset-dir /path/to/IRSTD-1k \
  --weight weights/weight.pkl \
  --base-size 256 \
  --bins 100 \
  --blob-dist 3.0 \
  --target-fa-ppm 15.0 \
  --prefer-repo-layout
```

### NUDT-SIRST

```bash
python Reem_results.py \
  --dataset-dir /path/to/NUDT-SIRST \
  --weight weights/weight.pkl \
  --base-size 256 \
  --bins 100 \
  --blob-dist 3.0 \
  --target-fa-ppm 15.0 \
  --prefer-repo-layout
```

---

## Dataset Layout

The evaluation script auto-detects two layouts:

| Layout | Image dir | Mask dir |
|---|---|---|
| Repo style | `images/` | `masks/` |
| Paper style | `IRSTD1k_Img/` | `IRSTD1k_Label/` |

Pass `--prefer-repo-layout` to prioritize `images/masks/` if both exist.

---

## Dependencies

```
torch >= 1.10
numpy
opencv-python
scikit-image
Pillow
tqdm
```

---

## Citation

```bibtex
@InProceedings{Sevim_2026_CVPR,
    author    = {Sevim, Yunus and T\"oreyin, Beh\c{c}et U\u{g}ur},
    title     = {SCR-Guided Difficulty-Aware Optimization for Infrared Small Target Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {7181-7189}
}
```

**Paper:** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026W/PBVS/html/Sevim_SCR-Guided_Difficulty-Aware_Optimization_for_Infrared_Small_Target_Detection_CVPRW_2026_paper.html) | [arXiv](https://arxiv.org/abs/2606.18783)
