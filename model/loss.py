# model/loss.py
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def ensure_b1hw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x[None, None, ...]
    if x.ndim == 3:
        return x[:, None, :, :]
    return x

def imagenet_unnormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)

def LLoss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    B, _, H, W = pred.shape

    x_index = (torch.arange(0, W, device=pred.device, dtype=pred.dtype)
               .view(1, 1, W).repeat(1, H, 1)) / float(W)
    y_index = (torch.arange(0, H, device=pred.device, dtype=pred.dtype)
               .view(1, H, 1).repeat(1, 1, W)) / float(H)

    smooth = 1e-8
    for i in range(B):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()
        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4.0 / (torch.pi ** 2)) * (
            (torch.atan(pred_centery / (pred_centerx + smooth)) -
             torch.atan(target_centery / (target_centerx + smooth))) ** 2
        )

        pred_length = torch.sqrt(pred_centerx**2 + pred_centery**2 + smooth)
        target_length = torch.sqrt(target_centerx**2 + target_centery**2 + smooth)
        length_loss = torch.min(pred_length, target_length) / (torch.max(pred_length, target_length) + smooth)

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
                return 1.0 - siou.mean() + lloss
            return 1.0 - siou.mean()

        return 1.0 - iou.mean()

class HybridSLSWithSCRReward(nn.Module):
    """
    total_loss = SLS - lambda_reward * mean( w(scr_gt_local) * softIoU(pred,gt) )
    input_img: ImageNet-normalized (B,3,H,W)
    """
    
