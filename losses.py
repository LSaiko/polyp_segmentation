"""
losses.py — Loss Functions and Evaluation Metrics
==================================================
PURPOSE: Define HOW the model measures its own mistakes (loss)
         and HOW we measure quality (metrics like IoU and Dice).

WHY NOT just use CrossEntropyLoss?
  Polyps are small — they might be only 5% of the image pixels.
  A model that predicts "nothing" everywhere would get 95% pixel accuracy!
  Dice and IoU losses are designed for exactly this class imbalance problem.
  They measure OVERLAP between prediction and ground truth, not pixel count.

KEY CONCEPTS:
  - Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)   (ranges 0→1, higher=better)
  - IoU (Jaccard):    |A ∩ B| / |A ∪ B|            (ranges 0→1, higher=better)
  - Loss = 1 - metric, because optimizers MINIMIZE, so we flip the sign.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Dice Loss ────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Formula: Dice = (2 * |pred ∩ target| + smooth) / (|pred| + |target| + smooth)
    Loss     = 1 - Dice   (we minimize loss, so maximizing dice = minimizing this)

    The 'smooth' term (epsilon) prevents division by zero when both
    pred and target are all zeros (no polyp in the image).

    HOW IT WORKS:
        pred   = model output AFTER sigmoid, values between 0.0 and 1.0
        target = ground truth mask, values 0.0 or 1.0

        |pred ∩ target| = sum(pred * target)  — how much they overlap
        |pred|          = sum(pred)            — how much pred covers
        |target|        = sum(target)          — how much target covers
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits  : raw model output, shape (B, 1, H, W) — NOT yet sigmoid'd
            targets : binary masks,     shape (B, 1, H, W) — values 0.0 or 1.0

        Returns:
            scalar loss value (lower = better overlap)
        """
        # Apply sigmoid to convert raw logits (any value) to probabilities (0-1)
        # We do this INSIDE the loss so the model doesn't need a final sigmoid layer
        pred = torch.sigmoid(logits)

        # Flatten spatial dimensions: (B, 1, H, W) → (B, H*W)
        # We compute dice per-image then average over the batch
        pred    = pred.view(pred.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Numerator: 2 * overlap
        intersection = (pred * targets).sum(dim=1)
        numerator    = 2.0 * intersection + self.smooth

        # Denominator: sum of both areas
        denominator  = pred.sum(dim=1) + targets.sum(dim=1) + self.smooth

        # Dice score per image, then mean over batch
        dice = numerator / denominator        # shape: (B,)
        loss = 1.0 - dice.mean()             # scalar

        return loss


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss — the most common choice for medical segmentation.

    WHY COMBINE THEM?
        BCE (Binary Cross Entropy) is good at fine-grained pixel-level training.
        Dice is good at handling class imbalance (tiny polyps vs large background).
        Together they get benefits of both.

    Loss = BCE_weight * BCE_loss + Dice_weight * Dice_loss
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce         = nn.BCEWithLogitsLoss()   # numerically stable BCE
        self.dice        = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ── Metrics ──────────────────────────────────────────────────────────────────

def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Compute Dice coefficient (quality metric, NOT a loss).

    Higher is better. Perfect overlap = 1.0. No overlap = 0.0.
    A score of 0.80+ is generally considered good for polyp segmentation.

    Args:
        logits    : raw model output (B, 1, H, W)
        targets   : ground truth masks (B, 1, H, W)
        threshold : pixels above this probability are classified as polyp
        smooth    : prevents division by zero

    Returns:
        float: mean dice score over the batch
    """
    # Convert logits → binary predictions
    with torch.no_grad():
        pred = (torch.sigmoid(logits) > threshold).float()

    pred    = pred.view(pred.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (pred * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=1) + targets.sum(dim=1) + smooth)

    return dice.mean().item()   # return Python float


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Compute IoU / Jaccard Index (the other standard segmentation metric).

    IoU = |prediction ∩ ground_truth| / |prediction ∪ ground_truth|

    Interpretation:
        0.0 = no overlap at all
        0.5 = reasonable overlap
        0.7 = good
        0.8+ = excellent

    Provation and other medical device companies will ask you about IoU
    during technical interviews — know how to interpret these numbers.

    Args:
        logits    : raw model output (B, 1, H, W)
        targets   : ground truth masks (B, 1, H, W)
        threshold : probability threshold for binary prediction
        smooth    : prevents division by zero

    Returns:
        float: mean IoU score over the batch
    """
    with torch.no_grad():
        pred = (torch.sigmoid(logits) > threshold).float()

    pred    = pred.view(pred.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (pred * targets).sum(dim=1)          # pixels in BOTH
    union        = pred.sum(dim=1) + targets.sum(dim=1) - intersection  # pixels in EITHER

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def pixel_accuracy(logits, targets, threshold=0.5):
    """
    Simple pixel-level accuracy. Shown here so you understand WHY we don't
    rely on it: a model predicting nothing gets ~90% accuracy on most scans.

    Always report Dice AND IoU alongside pixel accuracy.
    """
    with torch.no_grad():
        pred = (torch.sigmoid(logits) > threshold).float()
    correct = (pred == targets).float().sum()
    total   = torch.numel(targets)
    return (correct / total).item()
