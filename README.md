# REEM: SCR-Guided Difficulty-Aware Optimization for Infrared Small Target Detection

---

## Overview

REEM introduces an SCR (Signal-to-Clutter Ratio) guided difficulty-aware reward mechanism on top of the MSHNet backbone. Low-SCR targets (harder to detect) receive higher training weight via a soft IoU reward term, improving detection performance without architectural changes.

> **Note:** Training code and full codebase will be released soon. 
> Evaluation code and pretrained weights are already available below.
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

Expected output:
```
mIoU=68.49% | PD=93.88% | FA(ppm)=6.68 | op_idx=69/100
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

Expected output:
```
mIoU=79.86% | PD=97.35% | FA(ppm)=11.21 | op_idx=10/100
```

> **Note on operating point selection:** PD and FA are reported at the threshold where
> FA is minimized subject to PD >= target. Minor differences from paper table values
> (~0.2% PD, ~0.4 ppm FA on IRSTD-1k) are due to bin resolution (bins=100).
> IoU matches exactly on both datasets; FA matches exactly on NUDT-SIRST.

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
