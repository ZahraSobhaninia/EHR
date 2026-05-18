import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", 
                 use_pos_weight=False, pos_weight=None):  # ← اضافه شد
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.use_pos_weight = use_pos_weight
        self.pos_weight = pos_weight

    def forward(self, logits, targets, pos_weight=None):
        targets = targets.float()
        
        pw = pos_weight if pos_weight is not None else self.pos_weight
        
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pw,
            reduction="none"
        )
        probas = torch.sigmoid(logits)
        pt = probas * targets + (1 - probas) * (1 - targets)
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            loss = self.alpha * focal_weight * bce_loss
        else:
            loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss