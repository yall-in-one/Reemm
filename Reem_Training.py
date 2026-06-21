#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSHNet Repo-Exact (paper-aligned pipeline) + SCR-Reward loss
- Same dataset pipeline as base repo-exact (train aug + ImageNet norm)
- Same metrics/selection as base (repo mIoU + repo PD_FA sweep + op by FA≈15ppm)
- Only difference: loss = SLSIoULoss - lambda * reward(SCR, softIoU)

Example:
python main_mshnet_repoexact_scrreward_multi.py \
  --dataset-dir /home/ysevim/Desktop/MSHNet/Dataset/IRSTD-1k \
  --epochs 300 --runs 10 --fixed-split --split-seed 0 \
  --base-size 256 --crop-size 256 --batch-size 24 --workers 2 \
  --warm-epoch 5 --lr 0.05 --bins 10 --blob-dist 3.0 \
  --lambda-grid 0.8,0.8,0.8,1,1,1,1.5,1.5,1.5,1.5 \
  --save-dir /home/ysevim/Desktop/MSHNet/weight/MSHNet_REPOEXACT_SCRREWARD_MULTI \
  --summary-csv /tmp/mshnet_repoexact_scrreward_summary.csv \
  --bins-csv /tmp/mshnet_repoexact_scrreward_bins.csv \
  --scr-edges 1,2,4,8
"""

import os
import sys
import time
import csv
import argparse
import os.path as osp
from typing import List, Tuple, Any, Dict

import random
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

import cv2
from skimage import measure

# =========================
# Repro
# =========================
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# Lists
# =========================
def read_list(txt_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        items = [ln.strip() for ln in f.readlines()]
    items = [x for x in items if x and (not x.startswith("#"))]
    return items

def write_list(txt_path: str, items: List[str]):
    os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
    with open(txt_path, "w") as f:
        for x in items:
            f.write(str(x).strip() + "\n")

def split_trainval(trainval_list: str, val_ratio: float, seed: int) -> Tuple[str, str]:
    names = read_list(trainval_list)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(names))
    rng.shuffle(idx)

    n_val = int(round(len(names) * float(val_ratio)))
    n_val = max(1, min(n_val, len(names) - 1))

    val_idx = set(idx[:n_val].tolist())
    train_names = [names[i] for i in range(len(names)) if i not in val_idx]
    val_names = [names[i] for i in range(len(names)) if i in val_idx]

    tag = f"vr{val_ratio}_seed{seed}"
    tr_path = f"/tmp/irstd_train_{tag}.txt"
    va_path = f"/tmp/irstd_val_{tag}.txt"
    write_list(tr_path, train_names)
    write_list(va_path, val_names)
    return tr_path, va_path

# =========================
# Path resolve helpers
# =========================
def _try_resolve_image_anyext(img_dir: str, name: str) -> str:
    base = osp.basename(name)
    root, ext = osp.splitext(base)

    if ext:  # has ext
        cand = osp.join(img_dir, base)
        if osp.exists(cand):
            return cand
        base = root

    for e in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        cand = osp.join(img_dir, base + e)
        if osp.exists(cand):
            return cand

    cand = osp.join(img_dir, base)
    if osp.exists(cand):
        return cand

    raise FileNotFoundError(f"Image not found for '{name}' under '{img_dir}'")

def _resolve_label_png(lab_dir: str, name: str) -> str:
    base = osp.basename(name)
    root, _ = osp.splitext(base)
    lab_path = osp.join(lab_dir, root + ".png")
    if not osp.exists(lab_path):
        lab_path2 = osp.join(lab_dir, base + ".png")
        if osp.exists(lab_path2):
            return lab_path2
        raise FileNotFoundError(f"Label not found: {lab_path}")
    return lab_path

# =========================
# Transforms (Repo style)
# =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def pil_to_tensor_and_norm(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std

def pil_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    m = np.array(mask, dtype=np.uint8)
    m = (m > 0).astype(np.float32)
    return torch.from_numpy(m).unsqueeze(0)

def imagenet_unnormalize(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) normalized -> [0,1] approx"""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)

# =========================
# Dataset
# =========================
class IRSTDRepoExactDataset(Data.Dataset):
    def __init__(self, dataset_dir: str, list_file: str, mode: str,
                 base_size: int = 256, crop_size: int = 256,
                 prefer_repo_layout: bool = True):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.list_file = list_file
        self.names = read_list(list_file)
        self.mode = mode
        self.base_size = int(base_size)
        self.crop_size = int(crop_size)

        repo_img = osp.join(dataset_dir, "images")
        repo_msk = osp.join(dataset_dir, "masks")
        paper_img = osp.join(dataset_dir, "IRSTD1k_Img")
        paper_msk = osp.join(dataset_dir, "IRSTD1k_Label")

        has_repo = osp.isdir(repo_img) and osp.isdir(repo_msk)
        has_paper = osp.isdir(paper_img) and osp.isdir(paper_msk)

        if prefer_repo_layout and has_repo:
            self.img_dir = repo_img
            self.msk_dir = repo_msk
            self.layout = "repo"
        elif has_paper:
            self.img_dir = paper_img
            self.msk_dir = paper_msk
            self.layout = "paper"
        elif has_repo:
            self.img_dir = repo_img
            self.msk_dir = repo_msk
            self.layout = "repo"
        else:
            raise FileNotFoundError(
                f"Cannot detect dataset layout under: {dataset_dir}\n"
                f"Expected either (images/masks) or (IRSTD1k_Img/IRSTD1k_Label)."
            )

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        crop_size = self.crop_size

        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return img, mask

    def _testval_sync_transform(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)
        return img, mask

    def __getitem__(self, i: int):
        name = self.names[i]
        img_path = _try_resolve_image_anyext(self.img_dir, name)

        if self.layout == "repo":
            base = osp.basename(name)
            root, ext = osp.splitext(base)
            lab_name = root if ext else base
            lab_path = osp.join(self.msk_dir, lab_name + ".png")
            if not osp.exists(lab_path):
                lab_path = _resolve_label_png(self.msk_dir, name)
        else:
            lab_path = _resolve_label_png(self.msk_dir, name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(lab_path).convert("L")

        if self.mode == "train":
            img, mask = self._sync_transform(img, mask)
        else:
            img, mask = self._testval_sync_transform(img, mask)

        x = pil_to_tensor_and_norm(img)
        y = pil_mask_to_tensor(mask)
        return x, y, osp.basename(name)

# =========================
# Loss (Repo SLSIoU)
# =========================
def LLoss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = torch.tensor(0.0, requires_grad=True, device=pred.device)
    B, _, H, W = pred.shape

    x_index = torch.arange(0, W, 1, device=pred.device).view(1, 1, W).repeat((1, H, 1)) / float(W)
    y_index = torch.arange(0, H, 1, device=pred.device).view(1, H, 1).repeat((1, 1, W)) / float(H)

    smooth = 1e-8
    for i in range(B):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()
        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4.0 / (torch.pi ** 2)) * (
            torch.square(
                torch.arctan((pred_centery) / (pred_centerx + smooth)) -
                torch.arctan((target_centery) / (target_centerx + smooth))
            )
        )

        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)

        length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)
        loss = loss + (1.0 - length_loss + angle_loss) / float(B)

    return loss

class SLSIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_log: torch.Tensor, target: torch.Tensor,
                warm_epoch: int, epoch: int, with_shape: bool = True) -> torch.Tensor:
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        dis = torch.pow((pred_sum - target_sum) / 2.0, 2)
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth)
        iou = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)

        if epoch > warm_epoch:
            siou = alpha * iou
            if with_shape:
                lloss = LLoss(pred, target)
                loss = 1.0 - siou.mean() + lloss
            else:
                loss = 1.0 - siou.mean()
        else:
            loss = 1.0 - iou.mean()

        return loss

# =========================
# SCR-Reward wrapper (repo-exact input)
# =========================
class HybridSLSWithSCRReward(nn.Module):
    """
    total_loss = SLS - lambda_reward * mean( w(scr_gt_local) * softIoU(pred,gt) )

    NOTE:
    - input_img is ImageNet-normalized (B,3,H,W)
    - For SCR, we unnormalize -> gray in [0,1]
    """
    def __init__(
        self,
        sls_loss: nn.Module,
        lambda_reward: float = 1.0,
        eps: float = 1e-6,
        scr_clip_max: float = 12.0,
        scr_k: float = 2.0,
        alpha: float = 2.0,
        start_reward_after: int = 5,
        detach_w: bool = True,
        outer_scale: float = 4.0,
        inner_scale: float = 1.5,
        min_outer: int = 15,
    ):
        super().__init__()
        self.sls_loss = sls_loss
        self.lambda_reward = float(lambda_reward)
        self.eps = float(eps)
        self.scr_clip_max = float(scr_clip_max)
        self.scr_k = float(scr_k)
        self.alpha = float(alpha)
        self.start_reward_after = int(start_reward_after)
        self.detach_w = bool(detach_w)
        self.outer_scale = float(outer_scale)
        self.inner_scale = float(inner_scale)
        self.min_outer = int(min_outer)

    @staticmethod
    def _soft_iou(prob: torch.Tensor, gt01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        prob = prob.clamp(0.0, 1.0)
        gt01 = gt01.clamp(0.0, 1.0)
        inter = (prob * gt01).flatten(1).sum(dim=1)
        union = (prob + gt01 - prob * gt01).flatten(1).sum(dim=1)
        return inter / (union + eps)

    def _scr_weight(self, scr: torch.Tensor) -> torch.Tensor:
        scr = scr.clamp(0.0, self.scr_clip_max)
        scr_norm = scr / (scr + self.scr_k)          # [0,1)
        w = 1.0 + self.alpha * (1.0 - scr_norm)      # [1,1+alpha]
        return w

    def _scr_local_from_gt_batch(self, gt01: torch.Tensor, input_img_norm: torch.Tensor) -> torch.Tensor:
        gt01 = ensure_b1hw(gt01)  # (B,1,H,W)
        # unnormalize then gray
        x01 = imagenet_unnormalize(input_img_norm)  # (B,3,H,W) in [0,1]
        gray = x01.mean(dim=1)                      # (B,H,W)
        B, H, W = gray.shape
        eps: float = 1e-6

        scr_list = []
        for b in range(B):
            g = (gt01[b, 0] > 0.5)
            area = int(g.sum().item())
            if area < 9:
                scr_list.append(torch.tensor(0.0, device=gray.device, dtype=gray.dtype))
                continue

            ys, xs = torch.where(g)
            y0 = int(ys.min().item()); y1 = int(ys.max().item())
            x0 = int(xs.min().item()); x1 = int(xs.max().item())

            th = (y1 - y0 + 1)
            tw = (x1 - x0 + 1)

            inner_h = max(1, int(round(th * self.inner_scale / 2.0)))
            inner_w = max(1, int(round(tw * self.inner_scale / 2.0)))
            outer_h = max(1, int(round(th * self.outer_scale / 2.0)))
            outer_w = max(1, int(round(tw * self.outer_scale / 2.0)))

            min_outer_h = max(1, (int(self.min_outer) // 2))
            min_outer_w = max(1, (int(self.min_outer) // 2))
            outer_h = max(outer_h, min_outer_h)
            outer_w = max(outer_w, min_outer_w)

            cy = int(round((y0 + y1) / 2.0))
            cx = int(round((x0 + x1) / 2.0))

            oy0 = max(0, cy - outer_h); oy1 = min(H - 1, cy + outer_h)
            ox0 = max(0, cx - outer_w); ox1 = min(W - 1, cx + outer_w)

            iy0 = max(0, cy - inner_h); iy1 = min(H - 1, cy + inner_h)
            ix0 = max(0, cx - inner_w); ix1 = min(W - 1, cx + inner_w)

            ring = torch.zeros((H, W), dtype=torch.bool, device=gray.device)
            ring[oy0:oy1+1, ox0:ox1+1] = True
            ring[iy0:iy1+1, ix0:ix1+1] = False
            ring = ring & (~g)

            ring_n = int(ring.sum().item())
            gb = gray[b]
            if ring_n < 9:
                bg = (~g)
                if int(bg.sum().item()) < 9:
                    scr_list.append(torch.tensor(0.0, device=gray.device, dtype=gray.dtype))
                    continue
                mu_t = gb[g].mean()
                mu_b = gb[bg].mean()
                sigma_b = gb[bg].std() + self.eps
            else:
                mu_t = gb[g].mean()
                mu_b = gb[ring].mean()
                sigma_b = gb[ring].std() + self.eps

            scr = torch.abs(mu_t - mu_b) / (sigma_b + eps)
            scr = torch.clamp(scr, min=0.0, max=self.scr_clip_max)
            scr_list.append(scr)

        return torch.stack(scr_list, dim=0)  # (B,)

    def forward(self, pred_logits, gt, warm_epoch, epoch, input_img=None):
        sls = self.sls_loss(pred_logits, gt, warm_epoch, epoch)

        if input_img is None:
            return sls
        if epoch <= (warm_epoch + self.start_reward_after):
            return sls

        pred_prob = torch.sigmoid(pred_logits)
        pred_prob = ensure_b1hw(pred_prob)
        gt01 = ensure_b1hw(gt)

        iou = self._soft_iou(pred_prob, gt01, eps=self.eps)                 # (B,)
        scr_gt = self._scr_local_from_gt_batch(gt01, input_img)             # (B,)
        w = self._scr_weight(scr_gt)                                        # (B,)
        if self.detach_w:
            w = w.detach()

        reward = (w * iou).mean()
        return sls - self.lambda_reward * reward

# =========================
# Metrics (Repo style)
# =========================
class mIoU:
    def __init__(self, nclass: int = 1):
        self.nclass = nclass
        self.reset()

    def reset(self):
        self.total_inter = 0.0
        self.total_union = 0.0
        self.total_correct = 0.0
        self.total_label = 0.0

    def update(self, output: torch.Tensor, target: torch.Tensor):
        if len(target.shape) == 3:
            target = target.float().unsqueeze(1)
        else:
            target = target.float()

        assert output.shape == target.shape, "Predict and Label Shape Don't Match"
        predict = (output > 0).float()
        pixel_labeled = (target > 0).float().sum()
        pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()
        self.total_correct += float(pixel_correct.item())
        self.total_label += float(pixel_labeled.item())

        intersection = predict * ((predict == target).float())
        area_inter = float(intersection.sum().item())
        area_pred = float(predict.sum().item())
        area_lab  = float(target.sum().item())
        area_union = area_pred + area_lab - area_inter

        self.total_inter += area_inter
        self.total_union += area_union

    def get(self):
        pixAcc = (self.total_correct / (self.total_label + 1e-12))
        IoU = (self.total_inter / (self.total_union + 1e-12))
        return pixAcc, IoU

class PD_FA:
    def __init__(self, nclass: int, bins: int, size: int, dist_thr: float = 3.0):
        self.nclass = nclass
        self.bins = bins
        self.size = size
        self.dist_thr = float(dist_thr)
        self.FA = np.zeros(self.bins + 1, dtype=np.float64)
        self.PD = np.zeros(self.bins + 1, dtype=np.float64)
        self.target = np.zeros(self.bins + 1, dtype=np.float64)

    def reset(self):
        self.FA[:] = 0.0
        self.PD[:] = 0.0
        self.target[:] = 0.0

    def update(self, preds_logits: torch.Tensor, labels: torch.Tensor):
        probs = torch.sigmoid(preds_logits)
        preds_u8 = (probs * 255.0).detach().cpu().numpy()
        lab = labels.detach().cpu().numpy()

        B = preds_u8.shape[0]
        for b in range(B):
            pred_map = preds_u8[b, 0]
            lab_map = lab[b, 0]

            if pred_map.shape[0] != self.size or pred_map.shape[1] != self.size:
                pred_map = cv2.resize(pred_map, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            if lab_map.shape[0] != self.size or lab_map.shape[1] != self.size:
                lab_map = cv2.resize(lab_map.astype(np.float32), (self.size, self.size), interpolation=cv2.INTER_NEAREST)

            for iBin in range(self.bins + 1):
                score_thresh = iBin * (255.0 / self.bins)
                predits = (pred_map > score_thresh).astype(np.int64)
                labelss = (lab_map > 0.5).astype(np.int64)

                image = measure.label(predits, connectivity=2)
                coord_image = list(measure.regionprops(image))
                label = measure.label(labelss, connectivity=2)
                coord_label = list(measure.regionprops(label))

                self.target[iBin] += len(coord_label)

                image_area_total = [int(r.area) for r in coord_image]
                image_area_match = []
                distance_match = []

                m = 0
                while m < len(coord_label):
                    centroid_label = np.array(list(coord_label[m].centroid), dtype=np.float64)
                    for k in range(len(coord_image)):
                        centroid_image = np.array(list(coord_image[k].centroid), dtype=np.float64)
                        distance = np.linalg.norm(centroid_image - centroid_label)
                        area_image = int(coord_image[k].area)
                        if distance < self.dist_thr:
                            distance_match.append(distance)
                            image_area_match.append(area_image)
                            del coord_image[k]
                            break
                    m += 1

                dismatch = list(image_area_total)
                for a in image_area_match:
                    if a in dismatch:
                        dismatch.remove(a)

                self.FA[iBin] += float(np.sum(dismatch))
                self.PD[iBin] += float(len(distance_match))

    def get(self, img_num: int):
        img_num = max(int(img_num), 1)
        Final_FA_ratio = self.FA / ((self.size * self.size) * img_num + 1e-12)
        Final_FA_ppm = Final_FA_ratio * 1e6
        Final_PD = self.PD / (self.target + 1e-12)
        return Final_FA_ratio, Final_FA_ppm, Final_PD

# =========================
# SCR bins reporting (optional)
# =========================
def compute_scr_gt_local_from_normed_rgb(
    img_rgb_norm_1: torch.Tensor,  # (3,H,W) ImageNet norm
    gt_1: torch.Tensor,            # (1,H,W)
    eps: float = 1e-6,
    scr_clip_max: float = 12.0,
    outer_scale: float = 4.0,
    inner_scale: float = 1.5,
    min_outer: int = 15,
) -> float:
    x = img_rgb_norm_1.detach().float()
    x01 = imagenet_unnormalize(x[None, ...])[0]  # (3,H,W) [0,1]
    gray = x01.mean(dim=0)

    g = (gt_1[0].detach().float() > 0.5)
    area = int(g.sum().item())
    if area < 9:
        return 0.0

    ys, xs = torch.where(g)
    y0 = int(ys.min().item()); y1 = int(ys.max().item())
    x0 = int(xs.min().item()); x1 = int(xs.max().item())

    H, W = gray.shape
    th = (y1 - y0 + 1)
    tw = (x1 - x0 + 1)

    inner_h = max(1, int(round(th * inner_scale / 2.0)))
    inner_w = max(1, int(round(tw * inner_scale / 2.0)))
    outer_h = max(1, int(round(th * outer_scale / 2.0)))
    outer_w = max(1, int(round(tw * outer_scale / 2.0)))

    outer_h = max(outer_h, max(1, int(min_outer) // 2))
    outer_w = max(outer_w, max(1, int(min_outer) // 2))

    cy = int(round((y0 + y1) / 2.0))
    cx = int(round((x0 + x1) / 2.0))

    oy0 = max(0, cy - outer_h); oy1 = min(H - 1, cy + outer_h)
    ox0 = max(0, cx - outer_w); ox1 = min(W - 1, cx + outer_w)

    iy0 = max(0, cy - inner_h); iy1 = min(H - 1, cy + inner_h)
    ix0 = max(0, cx - inner_w); ix1 = min(W - 1, cx + inner_w)

    ring = torch.zeros((H, W), dtype=torch.bool)
    ring[oy0:oy1+1, ox0:ox1+1] = True
    ring[iy0:iy1+1, ix0:ix1+1] = False
    ring = ring & (~g)

    ring_n = int(ring.sum().item())
    if ring_n < 9:
        bg = (~g)
        if int(bg.sum().item()) < 9:
            return 0.0
        mu_t = gray[g].mean()
        mu_b = gray[bg].mean()
        sigma_b = gray[bg].std() + eps
    else:
        mu_t = gray[g].mean()
        mu_b = gray[ring].mean()
        sigma_b = gray[ring].std() + eps

    scr = torch.abs(mu_t - mu_b) / (sigma_b + eps)
    scr = torch.clamp(scr, min=0.0, max=scr_clip_max)
    return float(scr.item())

def scr_bin_index(scr: float, edges: List[float]) -> int:
    for i, e in enumerate(edges):
        if scr < e:
            return i
    return len(edges)

def scr_bin_name(i: int, edges: List[float]) -> str:
    if i == 0:
        return f"<{edges[0]:g}"
    if i == len(edges):
        return f">={edges[-1]:g}"
    return f"[{edges[i-1]:g},{edges[i]:g})"

@torch.no_grad()
def eval_test_by_scr_bins_paperlike(model, loader, device, scr_edges: List[float], scr_clip_max: float, blob_dist: float):
    model.eval()
    nbins = len(scr_edges) + 1
    bins = [{"iou_sum": 0.0, "pd_sum": 0.0, "fa_sum": 0.0, "n": 0} for _ in range(nbins)]
    overall = {"iou_sum": 0.0, "pd_sum": 0.0, "fa_sum": 0.0, "n": 0}

    def cc_centroids_and_areas(bin_img: np.ndarray):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img.astype(np.uint8), connectivity=8)
        cents, areas = [], []
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            ys, xs = np.where(labels == k)
            if ys.size == 0:
                continue
            cents.append((float(ys.mean()), float(xs.mean())))
            areas.append(area)
        return cents, areas

    def pd_fa_ppm_paper(pred_bin: np.ndarray, gt_bin: np.ndarray, max_dist: float = 3.0) -> Tuple[float, float]:
        H, W = gt_bin.shape[:2]
        total_area = float(H * W)

        gt_c, _ = cc_centroids_and_areas(gt_bin)
        pr_c, pr_area = cc_centroids_and_areas(pred_bin)

        if len(gt_c) == 0:
            fa_ppm = (float(sum(pr_area)) / (total_area + 1e-9)) * 1e6
            return 0.0, float(fa_ppm)

        matched_pr = set()
        matched_gt = 0
        for (gy, gx) in gt_c:
            best_j, best_d = None, 1e9
            for j, (py, px) in enumerate(pr_c):
                if j in matched_pr:
                    continue
                d = ((py - gy) ** 2 + (px - gx) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is not None and best_d <= max_dist:
                matched_gt += 1
                matched_pr.add(best_j)

        unmatched_area = 0
        for j, a in enumerate(pr_area):
            if j not in matched_pr:
                unmatched_area += int(a)

        pd = matched_gt / (len(gt_c) + 1e-9)
        fa_ppm = (float(unmatched_area) / (total_area + 1e-9)) * 1e6
        return float(pd), float(fa_ppm)

    for x, y, _ in tqdm(loader, dynamic_ncols=True, desc="Eval(test_by_SCR)"):
        x = x.to(device, non_blocking=True)
        y = ensure_b1hw(y.to(device, non_blocking=True))

        out = model(x, True)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            _, logits = out
        else:
            logits = out

        prob = torch.sigmoid(logits)
        pred_bin = (prob >= 0.5).to(torch.uint8).cpu().numpy()
        gt_bin = (y > 0.5).to(torch.uint8).cpu().numpy()

        B = pred_bin.shape[0]
        for b in range(B):
            p = pred_bin[b, 0]
            g = gt_bin[b, 0]
            inter = np.logical_and(p > 0, g > 0).sum()
            uni = np.logical_or(p > 0, g > 0).sum()
            iou = float(inter / (uni + 1e-9))
            pd, fa_ppm = pd_fa_ppm_paper(p, g, max_dist=blob_dist)

            scr = compute_scr_gt_local_from_normed_rgb(x[b].detach().cpu(), y[b].detach().cpu(), scr_clip_max=scr_clip_max)
            bi = scr_bin_index(scr, scr_edges)

            overall["iou_sum"] += iou
            overall["pd_sum"] += pd
            overall["fa_sum"] += fa_ppm
            overall["n"] += 1

            bins[bi]["iou_sum"] += iou
            bins[bi]["pd_sum"] += pd
            bins[bi]["fa_sum"] += fa_ppm
            bins[bi]["n"] += 1

    def finalize(acc):
        n = max(acc["n"], 1)
        return {"n": int(acc["n"]), "iou": acc["iou_sum"] / n, "pd": acc["pd_sum"] / n, "fa_ppm": acc["fa_sum"] / n}

    out = {"overall": finalize(overall), "bins": []}
    for i in range(nbins):
        out["bins"].append({"bin": scr_bin_name(i, scr_edges), **finalize(bins[i])})
    return out

# =========================
# Model helpers
# =========================
def load_init_weight(model, weight_path: str, device: str, strict: bool = False):
    if not weight_path:
        print("[INIT] no init weight (random init).")
        return model.to(device)

    state = torch.load(weight_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "net" in state:
            state = state["net"]
        elif "model" in state:
            state = state["model"]

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[INIT LOAD] {osp.basename(weight_path)} strict={strict} missing={len(missing)} unexpected={len(unexpected)}")

    miss_ratio = len(missing) / (len(model.state_dict()) + 1e-9)
    if miss_ratio > 0.05:
        raise RuntimeError(
            f"[INIT FAIL] Too many missing keys ({len(missing)}/{len(model.state_dict())} ~ {miss_ratio*100:.1f}%). "
            f"Wrong init checkpoint? path={weight_path}"
        )

    return model.to(device)

def ensure_b1hw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x[None, None, ...]
    if x.ndim == 3:
        return x[:, None, :, :]
    return x

def forward_mshnet_logits(model, x: torch.Tensor, warm_flag: bool) -> torch.Tensor:
    out = model(x, warm_flag)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        _, pred = out
        return pred
    return out

# =========================
# Train/Eval (repo metrics)
# =========================
def train_one_epoch(model, loss_fn, optimizer, loader, device, warm_epoch, epoch):
    model.train()
    tbar = tqdm(loader, dynamic_ncols=True, desc=f"Train {epoch:03d}")
    down = nn.MaxPool2d(2, 2)

    tag = bool(epoch > warm_epoch)
    loss_avg = 0.0
    n = 0

    for x, y, _ in tbar:
        x = x.to(device, non_blocking=True)
        y = ensure_b1hw(y.to(device, non_blocking=True))

        masks, pred = model(x, tag)

        # === FINAL OUTPUT ===
        loss = loss_fn(
            pred,
            y,
            warm_epoch,
            epoch,
            input_img=x        # ✅ SCR sadece burada
        )

        # === DEEP SUPERVISION (NO SCR) ===
        yy = y
        for j in range(len(masks)):
            if j > 0:
                yy = down(yy)

            loss = loss + loss_fn(
                masks[j],
                yy,
                warm_epoch,
                epoch,
                input_img=None   # ❌ SCR kapalı
            )

        loss = loss / (len(masks) + 1)


        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        loss_avg += float(loss.item()) * bs
        n += bs
        tbar.set_postfix(loss=loss_avg / max(n, 1))

    return loss_avg / max(n, 1)

@torch.no_grad()
def eval_val_repo_metrics(model, loss_fn, loader, device, warm_epoch: int, epoch_for_loss: int,
                         bins: int, size: int, blob_dist: float):
    model.eval()
    miou = mIoU(nclass=1)
    pd_fa = PD_FA(nclass=1, bins=bins, size=size, dist_thr=blob_dist)

    loss_sum = 0.0
    n_img = 0

    for x, y, _ in tqdm(loader, dynamic_ncols=True, desc="Eval(val)"):
        x = x.to(device, non_blocking=True)
        y = ensure_b1hw(y.to(device, non_blocking=True))

        logits = forward_mshnet_logits(model, x, warm_flag=True)

        loss = loss_fn(logits, y, warm_epoch, epoch_for_loss, input_img=x)
        loss_sum += float(loss.item()) * x.size(0)

        miou.update(logits.detach().cpu(), y.detach().cpu())
        pd_fa.update(logits.detach(), y.detach())
        n_img += x.size(0)

    pixAcc, IoU = miou.get()
    FA_ratio, FA_ppm, PD = pd_fa.get(img_num=n_img)
    mid = bins // 2
    pd_mid = float(PD[mid]) if mid < len(PD) else float(PD[-1])
    fa_ppm_mid = float(FA_ppm[mid]) if mid < len(FA_ppm) else float(FA_ppm[-1])

    return {
        "loss": loss_sum / max(n_img, 1),
        "pixAcc": float(pixAcc),
        "iou": float(IoU),
        "pd_mid": pd_mid,
        "fa_ppm_mid": fa_ppm_mid,
        "n": int(n_img),
        "FA_ppm_curve": FA_ppm,
        "PD_curve": PD,
    }

@torch.no_grad()
def eval_test_bestop(model, loss_fn, loader, device, warm_epoch: int, epoch_for_loss: int,
                    bins: int, size: int, blob_dist: float, target_fa_ppm: float):
    model.eval()
    miou = mIoU(nclass=1)
    pd_fa = PD_FA(nclass=1, bins=bins, size=size, dist_thr=blob_dist)

    loss_sum = 0.0
    n_img = 0

    for x, y, _ in tqdm(loader, dynamic_ncols=True, desc="Eval(test)"):
        x = x.to(device, non_blocking=True)
        y = ensure_b1hw(y.to(device, non_blocking=True))

        logits = forward_mshnet_logits(model, x, warm_flag=True)

        loss = loss_fn(logits, y, warm_epoch, epoch_for_loss, input_img=x)
        loss_sum += float(loss.item()) * x.size(0)

        miou.update(logits.detach().cpu(), y.detach().cpu())
        pd_fa.update(logits.detach(), y.detach())
        n_img += x.size(0)

    pixAcc, IoU = miou.get()
    FA_ratio, FA_ppm, PD = pd_fa.get(img_num=n_img)

    # ❗ SON BIN'İ AT (thr=255 → boş maske)
    fa_curve = np.array(FA_ppm[:-1], dtype=np.float64)
    pd_curve = np.array(PD[:-1], dtype=np.float64)

    target_fa_ppm = float(target_fa_ppm)
    idx = int(np.argmin(np.abs(fa_curve - target_fa_ppm)))
    idx = max(0, min(idx, len(fa_curve) - 1))

    return {
        "loss": loss_sum / max(n_img, 1),
        "pixAcc": float(pixAcc),
        "iou": float(IoU),
        "pd": float(pd_curve[idx]),
        "fa_ppm": float(fa_curve[idx]),
        "fa_ratio": float(FA_ratio[idx]),
        "op_idx": int(idx),
        "n": int(n_img),
        "FA_ppm_curve": FA_ppm,
        "PD_curve": PD,
    }

# =========================
# IO
# =========================
def save_checkpoint(save_dir: str, model, optimizer, epoch: int, best_iou: float):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), osp.join(save_dir, "weight.pkl"))
    torch.save(
        {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "iou": best_iou},
        osp.join(save_dir, "checkpoint.pkl")
    )

def append_log(save_dir: str, line: str):
    os.makedirs(save_dir, exist_ok=True)
    with open(osp.join(save_dir, "metric.log"), "a") as f:
        f.write(line.rstrip() + "\n")

def write_csv(path: str, header: List[str], rows: List[List[Any]]):
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# =========================
# Args
# =========================
def parse_args():
    ap = argparse.ArgumentParser("MSHNet IRSTD-1k Repo-Exact + SCR Reward (fair) multi-run")

    ap.add_argument("--dataset-dir", type=str, required=True)

    ap.add_argument("--base-size", type=int, default=256)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--workers", type=int, default=2)

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--warm-epoch", type=int, default=5)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--multi-gpus", action="store_true")

    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--fixed-split", action="store_true")

    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--run-seed-base", type=int, default=0)

    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--blob-dist", type=float, default=3.0)

    # SCR reward params
    ap.add_argument("--lambda-grid", type=str, required=True,
                    help="Comma list length==runs. Example: 0.8,0.8,0.8,1,1,1,1.1,1.1,1.2,1.2")
    ap.add_argument("--scr-k", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--start-reward-after", type=int, default=5)
    ap.add_argument("--detach-w", action="store_true")
    ap.add_argument("--no-detach-w", action="store_true")

    ap.add_argument("--scr-edges", type=str, default="1,2,4,8")
    ap.add_argument("--scr-clip-max", type=float, default=12.0)

    ap.add_argument("--prefer-repo-layout", action="store_true")

    ap.add_argument("--save-dir", type=str, default="")
    ap.add_argument("--summary-csv", type=str, default="")
    ap.add_argument("--bins-csv", type=str, default="")

    # optional extra SCR-bin report
    ap.add_argument("--with-scr-bins", action="store_true", help="also print paper-like SCR bins report (thr=0.5)")

    ap.add_argument("--trainval-list", type=str, default="", help="Optional path to trainval list file")
    ap.add_argument("--test-list", type=str, default="", help="Optional path to test list file")

    ap.add_argument("--init-weight", type=str, default="", help="Optional init .pth/.pkl for model")
    ap.add_argument("--init-strict", action="store_true", help="Strict load init weight")
    ap.add_argument("--target-fa-ppm", type=float, default=15.0,
                help="Pick op_idx by FA(ppm) closest to this value (paper: NUDT=11.77, IRSTD=15.03 etc.)")




    return ap.parse_args()

# =========================
# Main
# =========================
def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # lambda grid
    lambda_grid = [float(x.strip()) for x in args.lambda_grid.split(",") if x.strip()]
    if len(lambda_grid) != int(args.runs):
        raise ValueError(f"--lambda-grid length ({len(lambda_grid)}) must equal --runs ({args.runs})")

    # lists
    trainval_list = args.trainval_list or osp.join(args.dataset_dir, "trainval.txt")
    test_list     = args.test_list     or osp.join(args.dataset_dir, "test.txt")

    if not osp.exists(trainval_list):
        raise FileNotFoundError(trainval_list)
    if not osp.exists(test_list):
        raise FileNotFoundError(test_list)

    scr_edges = [float(x.strip()) for x in args.scr_edges.split(",") if x.strip()]
    scr_edges = sorted(scr_edges)

    # model import from repo
    repo_root = osp.dirname(osp.abspath(__file__))
    sys.path.insert(0, repo_root)
    from model.MSHNet import MSHNet  # noqa

    weight_root = osp.join(repo_root, "weight")
    os.makedirs(weight_root, exist_ok=True)
    base_out = args.save_dir or osp.join(weight_root, "MSHNet_REPOEXACT_SCRREWARD_MULTI_%s" % time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(base_out, exist_ok=True)

    # shared test loader
    ds_test = IRSTDRepoExactDataset(args.dataset_dir, test_list, "test",
                                    base_size=args.base_size, crop_size=args.crop_size,
                                    prefer_repo_layout=bool(args.prefer_repo_layout))
    test_loader = Data.DataLoader(ds_test, batch_size=1, shuffle=False,
                                  num_workers=args.workers, pin_memory=(device == "cuda"), drop_last=False)

    print("\n==== MULTI-RUN TRAIN (Repo-Exact pipeline) + SCR-Reward loss (FAIR) ====")
    print(f"runs: {args.runs} | epochs: {args.epochs} | base_out: {base_out}")
    print(f"fixed_split: {bool(args.fixed_split)} | split_seed: {args.split_seed} | val_ratio: {args.val_ratio}")
    print(f"img_size(base)={args.base_size} | crop_size(train)={args.crop_size} | batch_size={args.batch_size}")
    print(f"bins={args.bins} | blob_dist={args.blob_dist}")
    print(f"SCR edges: {scr_edges} | scr_clip_max: {args.scr_clip_max}")
    print(f"lambda grid: {lambda_grid}")
    print(f"Test N={len(ds_test)}\n")

    detach_w = True
    if args.no_detach_w:
        detach_w = False
    if args.detach_w:
        detach_w = True

    all_run_rows = []
    all_bins_rows = []

    best_by_val = -1.0
    best_overall = None

    for run_id in range(int(args.runs)):
        run_seed = int(args.run_seed_base) + run_id
        split_seed = int(args.split_seed) if args.fixed_split else run_seed
        run_lambda = float(lambda_grid[run_id])

        set_seed(run_seed)
        train_list, val_list = split_trainval(trainval_list, args.val_ratio, split_seed)

        ds_train = IRSTDRepoExactDataset(args.dataset_dir, train_list, "train",
                                         base_size=args.base_size, crop_size=args.crop_size,
                                         prefer_repo_layout=bool(args.prefer_repo_layout))
        ds_val = IRSTDRepoExactDataset(args.dataset_dir, val_list, "val",
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       prefer_repo_layout=bool(args.prefer_repo_layout))

        train_loader = Data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.workers, pin_memory=(device == "cuda"), drop_last=True)
        val_loader = Data.DataLoader(ds_val, batch_size=1, shuffle=False,
                                     num_workers=args.workers, pin_memory=(device == "cuda"), drop_last=False)

        model = MSHNet(3)
        model = load_init_weight(model, args.init_weight, device=device, strict=bool(args.init_strict))

        if args.multi_gpus and device == "cuda" and torch.cuda.device_count() > 1:
            print(f"use {torch.cuda.device_count()} gpus")
            model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)
        else:
            model = model.to(device)

        sls = SLSIoULoss()
        loss_fn = HybridSLSWithSCRReward(
            sls_loss=sls,
            lambda_reward=run_lambda,
            eps=1e-6,
            scr_clip_max=float(args.scr_clip_max),
            scr_k=float(args.scr_k),
            alpha=float(args.alpha),
            start_reward_after=int(args.start_reward_after),
            detach_w=detach_w,
        )

        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        run_dir = osp.join(base_out, f"run{run_id:02d}_seed{run_seed}_split{split_seed}_lambda{run_lambda:g}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n[RUN {run_id:02d}] seed={run_seed} split_seed={split_seed} | lambda={run_lambda:g} | Train={len(ds_train)} Val={len(ds_val)}")
        print(f"Save dir: {run_dir}")

        best_val_iou = -1.0
        best_epoch = -1

        for epoch in range(int(args.epochs)):
            tr_loss = train_one_epoch(model, loss_fn, optimizer, train_loader, device, args.warm_epoch, epoch)

            val_metrics = eval_val_repo_metrics(
                model=model, loss_fn=loss_fn, loader=val_loader, device=device,
                warm_epoch=args.warm_epoch, epoch_for_loss=epoch,
                bins=args.bins, size=args.base_size, blob_dist=args.blob_dist
            )

            v_iou = val_metrics["iou"]
            line = (
                f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())} - "
                f"RUN{run_id:02d} EP{epoch:04d}\t"
                f"lambda {run_lambda:g}\t"
                f"train_loss {tr_loss:.6f}\t"
                f"val_loss {val_metrics['loss']:.6f}\t"
                f"mIoU {v_iou*100:6.2f}%\t"
                f"PD@mid {val_metrics['pd_mid']*100:6.2f}%\t"
                f"FAppm@mid {val_metrics['fa_ppm_mid']:8.2f}"
            )
            print(line)
            append_log(run_dir, line)

            if v_iou > best_val_iou:
                best_val_iou = v_iou
                best_epoch = epoch
                save_checkpoint(run_dir, model, optimizer, epoch, best_val_iou)

        # load best weights
        best_weight = osp.join(run_dir, "weight.pkl")
        state = torch.load(best_weight, map_location="cpu")
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model = model.to(device)

        # TEST (repo metrics)
        test_metrics = eval_test_bestop(
            model=model, loss_fn=loss_fn, loader=test_loader, device=device,
            warm_epoch=args.warm_epoch, epoch_for_loss=best_epoch,
            bins=args.bins, size=args.base_size, blob_dist=args.blob_dist,
            target_fa_ppm=float(args.target_fa_ppm)
        )


        print(f"\n[RUN {run_id:02d}] BEST VAL: epoch={best_epoch} | val_mIoU={best_val_iou*100:.2f}% | lambda={run_lambda:g}")
        print(
            f"[RUN {run_id:02d}] TEST (op by FA≈15ppm): "
            f"mIoU={test_metrics['iou']*100:.2f}% | "
            f"PD={test_metrics['pd']*100:.2f}% | "
            f"FA(ppm)={test_metrics['fa_ppm']:.2f} | "
            f"(idx={test_metrics['op_idx']}/{args.bins}) | N={test_metrics['n']}"
        )

        if args.with_scr_bins:
            scr_report = eval_test_by_scr_bins_paperlike(
                model=model, loader=test_loader, device=device,
                scr_edges=scr_edges, scr_clip_max=float(args.scr_clip_max), blob_dist=float(args.blob_dist)
            )
            print(f"[RUN {run_id:02d}] TEST by SCR bins (paper-like, thr=0.5 on prob):")
            for row in scr_report["bins"]:
                print(
                    f"  SCR {row['bin']:>8s} | N={row['n']:4d} | "
                    f"IoU={row['iou']*100:6.2f}% | PD={row['pd']*100:6.2f}% | FA(ppm)={row['fa_ppm']:8.2f}"
                )

            for b in scr_report["bins"]:
                all_bins_rows.append([
                    run_id, run_seed, split_seed, run_lambda,
                    best_epoch, best_val_iou,
                    b["bin"], b["n"], b["iou"], b["pd"], b["fa_ppm"], run_dir
                ])

        all_run_rows.append([
            run_id, run_seed, split_seed, run_lambda,
            best_epoch, best_val_iou,
            test_metrics["loss"], test_metrics["iou"], test_metrics["pd"], test_metrics["fa_ppm"], test_metrics["n"],
            run_dir
        ])

        if best_val_iou > best_by_val:
            best_by_val = best_val_iou
            best_overall = (run_id, run_dir, run_lambda, best_epoch, best_val_iou, test_metrics)

    # summary
    print("\n==================== SUMMARY (all runs) ====================")
    all_run_rows_sorted = sorted(all_run_rows, key=lambda r: float(r[5]), reverse=True)
    for r in all_run_rows_sorted:
        run_id, run_seed, split_seed, run_lambda, best_epoch, best_val_iou, tloss, tiou, tpd, tfa, tn, run_dir = r
        print(
            f"RUN{run_id:02d} | lambda={run_lambda:g} | best_ep={best_epoch:4d} | val_mIoU={best_val_iou*100:6.2f}% | "
            f"TEST mIoU={tiou*100:6.2f}% PD={tpd*100:6.2f}% FA(ppm)={tfa:8.2f} N={tn:4d} | {run_dir}"
        )

    if best_overall is not None:
        run_id, run_dir, run_lambda, best_epoch, best_val_iou, test_metrics = best_overall
        print("\n==================== BEST RUN (by VAL mIoU) ====================")
        print(f"RUN{run_id:02d} | dir={run_dir}")
        print(f"lambda={run_lambda:g} | best_epoch={best_epoch} | val_mIoU={best_val_iou*100:.2f}%")
        print(
            f"TEST(op by FA≈15ppm): mIoU={test_metrics['iou']*100:.2f}% | "
            f"PD={test_metrics['pd']*100:.2f}% | FA(ppm)={test_metrics['fa_ppm']:.2f} | N={test_metrics['n']}"
        )

    if args.summary_csv:
        header = ["run_id", "run_seed", "split_seed", "lambda_reward",
                  "best_epoch", "best_val_miou",
                  "test_loss", "test_miou", "test_pd", "test_fa_ppm", "test_n", "run_dir"]
        write_csv(args.summary_csv, header, all_run_rows_sorted)
        print(f"\nWrote summary CSV: {args.summary_csv}")

    if args.bins_csv and args.with_scr_bins:
        header = ["run_id", "run_seed", "split_seed", "lambda_reward",
                  "best_epoch", "best_val_miou",
                  "scr_bin", "n", "iou", "pd", "fa_ppm", "run_dir"]
        write_csv(args.bins_csv, header, all_bins_rows)
        print(f"Wrote bins CSV: {args.bins_csv}")

if __name__ == "__main__":
    main()
