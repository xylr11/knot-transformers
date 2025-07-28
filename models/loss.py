import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class ChamferLoss(nn.Module):
    """
    Chamfer loss for point sets with confidence.
    Parameters:
    - lambda_coord (float): Weight for coordinate loss
    - lambda_conf (float): Weight for confidence loss
    - lambda_reg (float): Weight for regularization loss
    - alpha (float): Sharpness for soft targets
    """
    def __init__(self, lambda_coord=1.0, lambda_conf=1.0, lambda_reg=0.1, alpha=10.0):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_conf = lambda_conf
        self.lambda_reg = lambda_reg
        self.alpha = alpha  # sharpness for soft targets

    def forward(self, pred_tensor, target_tensor):
        # Separate coordinates and confidences
        pred_xy = pred_tensor[..., :2]                # (B, M, 2)
        pred_conf = pred_tensor[..., 2]  # (B, M) in [0, 1]

        target_xy = target_tensor[..., :2]            # (B, N, 2)
        target_conf = target_tensor[..., 2]           # (B, N) in [0, 1]

        # Pairwise Euclidean distances
        dists = torch.cdist(pred_xy, target_xy, p=2)  # (B, M, N)

        # Mask: which target points are valid
        mask_target = (target_conf > 0.5)  # (B, N)

        # Mask out padded targets by setting distances to +inf
        dists_masked = dists.masked_fill(~mask_target.unsqueeze(1), float('inf'))

        # Chamfer direction: pred -> target (each pred finds closest real target)
        min_pred_to_target, _ = dists_masked.min(dim=2)  # (B, M)

        # Chamfer direction: target -> pred (each real target finds closest pred)
        min_target_to_pred, _ = dists_masked.min(dim=1)  # (B, N)
        
        min_target_to_pred = torch.nan_to_num(min_target_to_pred * mask_target)  # zero out padded

        # Weighted Chamfer loss (mean over valid points)
        chamfer_pred = (pred_conf * min_pred_to_target).sum(dim=1) / (pred_conf.sum(dim=1) + 1e-8)
        chamfer_target = min_target_to_pred.sum(dim=1) / (mask_target.sum(dim=1) + 1e-8)

        chamfer_loss = chamfer_pred.mean() + chamfer_target.mean()

        # Soft target confidence: encourage pred_conf ~ max similarity to real targets
        soft_target_conf = torch.exp(-self.alpha * dists**2)  # (B, M, N)
        soft_target_conf = soft_target_conf * mask_target.unsqueeze(1)  # zero out padded
        conf_targets = soft_target_conf.max(dim=2)[0]  # (B, M)

        conf_loss = F.binary_cross_entropy(pred_conf, conf_targets)

        # Penalize high conf
        reg_loss = (pred_conf ** 2).mean()

        # Total loss
        total_loss = (self.lambda_coord * chamfer_loss
                      + self.lambda_conf * conf_loss
                      + self.lambda_reg * reg_loss)

        return total_loss
    
class HungarianLoss(nn.Module):
    """
    Hungarian matched set loss for point sets with confidence.
    Args:
        lambda_coord (float): Weight for coordinate loss.
        lambda_conf (float): Weight for confidence loss.
    """
    def __init__(self, lambda_coord=1.0, lambda_conf=1.0):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_conf = lambda_conf

    def forward(self, pred, target):
        B, M, _ = pred.shape
        _, N, _ = target.shape

        pred_xy = pred[..., :2]
        pred_conf = pred[..., 2]

        target_xy = target[..., :2]
        target_conf = target[..., 2]

        loss_coord = 0.0
        loss_conf = 0.0

        for b in range(B):
            # Keep only valid targets
            valid = target_conf[b] > 0.5
            tgt_xy = target_xy[b][valid]
            tgt_conf = target_conf[b][valid]

            if tgt_xy.shape[0] == 0:
                continue  # no valid target points

            # Cost matrix: Euclidean distance
            dists = torch.cdist(pred_xy[b].unsqueeze(0), tgt_xy.unsqueeze(0), p=2).squeeze(0)  # (M, T)
            dists = dists.detach().cpu().numpy()

            # Solve assignment
            row_ind, col_ind = linear_sum_assignment(dists)

            # Matched pairs
            matched_pred_xy = pred_xy[b][row_ind]
            matched_tgt_xy = tgt_xy[col_ind]

            loss_coord += F.mse_loss(matched_pred_xy, matched_tgt_xy)

            # For confidence: match pred conf to target conf (optional)
            matched_pred_conf = pred_conf[b][row_ind]
            matched_tgt_conf = tgt_conf[col_ind]
            loss_conf += F.binary_cross_entropy(matched_pred_conf, matched_tgt_conf)

        total_loss = self.lambda_coord * loss_coord / B + self.lambda_conf * loss_conf / B
        return total_loss