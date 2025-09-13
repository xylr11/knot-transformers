import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianLoss(nn.Module):
    """
    Hungarian-matched set loss with confidence supervision and penalty for unmatched predictions.
    
    Args:
        lambda_coord (float): Weight for coordinate (MSE) loss.
        lambda_conf (float): Weight for confidence BCE loss.
        lambda_unmatched (float): Weight for unmatched prediction penalty (push conf -> 0).
    """
    def __init__(self, lambda_coord=1.0, lambda_conf=1.0, lambda_unmatched=1.0):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_conf = lambda_conf
        self.lambda_unmatched = lambda_unmatched

    def forward(self, pred, target):
        """
        pred: (B, M, 3) - [x, y, conf]
        target: (B, N, 3) - [x, y, conf]
        """
        B, M, _ = pred.shape
        _, N, _ = target.shape

        pred_xy = pred[..., :2]
        pred_conf = pred[..., 2]
        target_xy = target[..., :2]
        target_conf = target[..., 2]

        total_coord_loss = 0.0
        total_conf_loss = 0.0
        total_unmatched_loss = 0.0
        total_matches = 0

        for b in range(B):
            valid = target_conf[b] > 0.5
            tgt_xy = target_xy[b][valid]
            tgt_conf = target_conf[b][valid]

            if tgt_xy.shape[0] == 0:
                # no targets, penalize all predictions for having confidence > 0
                total_unmatched_loss += F.binary_cross_entropy(pred_conf[b], torch.zeros_like(pred_conf[b]))
                continue

            # Cost matrix: pairwise Euclidean distances (M x T)
            dists = torch.cdist(pred_xy[b].unsqueeze(0), tgt_xy.unsqueeze(0), p=2).squeeze(0)
            dists_np = dists.detach().cpu().numpy()

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(dists_np)

            matched_pred_xy = pred_xy[b][row_ind]
            matched_pred_conf = pred_conf[b][row_ind]
            matched_tgt_xy = tgt_xy[col_ind]

            # Coordinate loss
            coord_loss = F.mse_loss(matched_pred_xy, matched_tgt_xy, reduction='mean')
            total_coord_loss += coord_loss * matched_pred_xy.shape[0]

            # Confidence loss: matched -> 1
            conf_loss = F.binary_cross_entropy(matched_pred_conf, torch.ones_like(matched_pred_conf))
            total_conf_loss += conf_loss * matched_pred_conf.shape[0]

            # Unmatched predictions: confidence -> 0
            unmatched_mask = torch.ones(M, dtype=torch.bool, device=pred.device)
            unmatched_mask[row_ind] = False
            unmatched_pred_conf = pred_conf[b][unmatched_mask]

            if unmatched_pred_conf.numel() > 0:
                unmatched_loss = F.binary_cross_entropy(unmatched_pred_conf, torch.zeros_like(unmatched_pred_conf))
                total_unmatched_loss += unmatched_loss * unmatched_pred_conf.shape[0]

            total_matches += matched_pred_xy.shape[0]

        # Normalize by total number of matches to keep scale stable
        if total_matches > 0:
            total_coord_loss /= total_matches
            total_conf_loss /= total_matches
            total_unmatched_loss /= total_matches

        total_loss = (
            self.lambda_coord * total_coord_loss +
            self.lambda_conf * total_conf_loss +
            self.lambda_unmatched * total_unmatched_loss
        )
        return total_loss
