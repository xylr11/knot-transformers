import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianLoss(nn.Module):
    """
    Hungarian matched set loss for point sets with confidence.
    Args:
        lambda_coord (float): Weight for coordinate loss.
        lambda_conf (float): Weight for confidence loss.
    """
    def __init__(self, lambda_coord=1.0, lambda_conf=1.0, lambda_reg=0.1):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_conf = lambda_conf
        self.lambda_reg = lambda_reg

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

        # Penalize high conf
        reg_loss = (pred_conf ** 2).mean()

        total_loss = self.lambda_coord * loss_coord / B + self.lambda_conf * loss_conf / B + self.lambda_reg * reg_loss
        return total_loss