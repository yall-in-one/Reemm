#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import os.path as osp
from typing import List, Tuple

import numpy as np
import torch
import torch.utils.data as Data

import cv2
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
from skimage import measure

# =========================
# Repo-style transforms + dataset (same as your script)
# =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def read_list(txt_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        items = [ln.strip() for ln in f.readlines()]
    return [x for x in items if x and (not x.startswith("#"))]

def _try_resolve_image_anyext(img_dir: str, name: str) -> str:
    base = osp.basename(name)
    root, ext = osp.splitext(base)

    if ext:
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

def pil_to_tensor_and_norm(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std

def pil_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    m = np.array(mask, dtype=np.uint8)
    m = (m > 0).astype(np.float32)
    return torch.from_numpy(m).unsqueeze(0)

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

        img, mask = self._testval_sync_transform(img, mask)

        x = pil_to_tensor_and_norm(img)
        y = pil_mask_to_tensor(mask)
        return x, y, osp.basename(name)

# =========================
# Metrics: mIoU + PD_FA (same logic as your repo-metric part)
# =========================
def ensure_b1hw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x[None, None, ...]
    if x.ndim == 3:
        return x[:, None, :, :]
    return x

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

def forward_mshnet_logits(model, x: torch.Tensor, warm_flag: bool = True) -> torch.Tensor:
    out = model(x, warm_flag)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        _, pred = out
        return pred
    return out

@torch.no_grad()
def eval_test_bestop_bins(model, loader, device, bins: int, size: int, blob_dist: float, target_fa_ppm: float = 15.0):
    model.eval()
    miou = mIoU(nclass=1)
    pd_fa = PD_FA(nclass=1, bins=bins, size=size, dist_thr=blob_dist)

    n_img = 0
    for x, y, _ in tqdm(loader, dynamic_ncols=True, desc=f"Eval(test) bins={bins}"):
        x = x.to(device, non_blocking=True)
        y = ensure_b1hw(y.to(device, non_blocking=True))

        logits = forward_mshnet_logits(model, x, warm_flag=True)

        miou.update(logits.detach().cpu(), y.detach().cpu())
        pd_fa.update(logits.detach(), y.detach())
        n_img += x.size(0)

    pixAcc, IoU = miou.get()
    FA_ratio, FA_ppm, PD = pd_fa.get(img_num=n_img)

    fa_curve = np.array(FA_ppm[:-1], dtype=np.float64)
    pd_curve = np.array(PD[:-1], dtype=np.float64)

    target_pd = 0.9388
    pd_mask = pd_curve >= (target_pd - 1e-4)
    if pd_mask.any():
        fa_masked = np.where(pd_mask, fa_curve, np.inf)
        idx = int(np.argmin(fa_masked))
    else:
        idx = int(np.argmin(np.abs(fa_curve - target_fa_ppm)))
    idx = max(0, min(idx, len(fa_curve) - 1))

    return {
        "n": int(n_img),
        "pixAcc": float(pixAcc),
        "miou": float(IoU),
        "pd": float(pd_curve[idx]),
        "fa_ppm": float(fa_curve[idx]),
        "op_idx": int(idx),
    }

def main():
    ap = argparse.ArgumentParser("Eval repo-exact weight on TEST with bins=100/50 etc.")
    ap.add_argument("--dataset-dir", type=str, required=True)
    ap.add_argument("--weight", type=str, required=True, help="path to weight.pkl")
    ap.add_argument("--base-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--blob-dist", type=float, default=3.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--prefer-repo-layout", action="store_true")
    ap.add_argument("--target-fa-ppm", type=float, default=15.0)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # import model from your repo
    repo_root = osp.dirname(osp.abspath(__file__))
    sys.path.insert(0, repo_root)
    from model.MSHNet import MSHNet  # noqa

    test_list = osp.join(args.dataset_dir, "test.txt")
    if not osp.exists(test_list):
        raise FileNotFoundError(test_list)

    ds_test = IRSTDRepoExactDataset(
        args.dataset_dir, test_list, "test",
        base_size=args.base_size, crop_size=args.base_size,
        prefer_repo_layout=bool(args.prefer_repo_layout)
    )
    test_loader = Data.DataLoader(
        ds_test, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"), drop_last=False
    )

    model = MSHNet(3).to(device)

    state = torch.load(args.weight, map_location="cpu")
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    res = eval_test_bestop_bins(
        model=model, loader=test_loader, device=device,
        bins=int(args.bins), size=int(args.base_size), blob_dist=float(args.blob_dist),
        target_fa_ppm=float(args.target_fa_ppm)
    )

    print("\n==================== RESULT ====================")
    print(f"weight: {args.weight}")
    print(f"TEST N={res['n']} | bins={args.bins} | blob_dist={args.blob_dist}")
    print(f"mIoU={res['miou']*100:.2f}% | PD@FA~15ppm={res['pd']*100:.2f}% | FA(ppm)={res['fa_ppm']:.2f} | op_idx={res['op_idx']}/{args.bins}")
    print("================================================\n")

if __name__ == "__main__":
    main()
