# Run 1 — Baseline Configuration

**Date:** Initial training run  
**Model:** LightUNet (custom, ~7.7M parameters)  
**Hardware:** CPU only  
**Dataset:** 80 synthetic Kvasir-SEG-style images (64 train / 16 val)

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | 0.001 | Adam optimizer default |
| Image size | 64 × 64 | Small — fast CPU training |
| Batch size | 4 | Conservative for CPU |
| Epochs | 8 | Stopped early — loss still decreasing |
| BCE loss weight | 0.5 | Equal weighting |
| Dice loss weight | 0.5 | Equal weighting |
| Val split | 20% | 16 images for validation |
| LR scheduler | ReduceLROnPlateau (patience=3, factor=0.5) | |

## Results

| Metric | Best val score | Epoch achieved |
|---|---|---|
| IoU | 0.8796 | 7 |
| Dice | 0.9354 | 7 |
| Loss | 0.1967 | 8 |

## Per-epoch training log

| Epoch | Train loss | Val Dice | Val IoU | New best? |
|---|---|---|---|---|
| 1 | 0.6525 | 0.9163 | 0.8472 | Yes |
| 2 | 0.5534 | 0.9328 | 0.8747 | Yes |
| 3 | 0.4845 | 0.9322 | 0.8735 | No |
| 4 | 0.4335 | 0.6704 | 0.5664 | No |
| 5 | 0.3891 | 0.8899 | 0.8051 | No |
| 6 | 0.3471 | 0.9051 | 0.8280 | No |
| 7 | 0.3065 | 0.9354 | 0.8796 | Yes |
| 8 | 0.2636 | 0.9312 | 0.8720 | No |

## Observations

- Epoch 4 showed a significant dip (val IoU 0.87 → 0.57) — likely a difficult batch in validation. This is a sign that 64×64 image size may be losing important spatial detail that makes some polyp shapes ambiguous.
- Training loss decreased steadily throughout — model was not done learning at epoch 8. More epochs likely beneficial.
- Equal BCE/Dice weighting is a reasonable default but may under-optimize for boundary quality.

## What to try next

- Reduce learning rate for more stable convergence
- Increase image size to preserve more spatial detail
- Increase Dice loss weight to focus more on overlap quality
- Train for more epochs
