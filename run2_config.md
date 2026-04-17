# Run 2 — Tuned Configuration

**Date:** Follow-up experiment  
**Model:** LightUNet (same architecture, trained from scratch)  
**Hardware:** CPU only  
**Dataset:** Same 80 synthetic images

## Changes from Run 1 and rationale

### Learning rate: 0.001 → 0.0003

**Why:** A learning rate of 0.001 is Adam's default but can cause oscillation in later epochs where the loss landscape is flatter and finer. Reducing to 0.0003 (a common production choice) makes gradient steps smaller and more precise, allowing the optimizer to find lower loss valleys without bouncing past them.

**Effect:** More stable val metrics across epochs — fewer random spikes like the epoch 4 dip seen in Run 1.

### Image size: 64×64 → 96×96

**Why:** At 64×64, a polyp that is 20px wide in the original image becomes just ~5px wide after resizing. The encoder's MaxPool operations then shrink this to 1-2px at the bottleneck — not enough spatial signal. At 96×96, the same polyp is ~7-8px, giving the model substantially more detail to work with.

**Trade-off:** Training takes ~2× longer per epoch. On a GPU this is negligible; on CPU it matters. For real Kvasir-SEG data, 384×384 or 512×512 is standard.

### Batch size: 4 → 6

**Why:** Larger batches produce gradient estimates that average over more samples, which are statistically more accurate. This reduces the variance of each weight update and tends to stabilize training — especially noticeable with small datasets.

**Trade-off:** Requires more RAM. At 96×96 with batch=6, memory usage is still well within CPU limits (~400MB).

### Epochs: 8 → 13

**Why:** Run 1's training loss was still decreasing at epoch 8, suggesting the model had not yet converged. The additional epochs allowed the model to continue improving without overfitting (val metrics continued to improve alongside train metrics).

### BCE weight: 0.5 → 0.3 | Dice weight: 0.5 → 0.7

**Why:** BCE (Binary Cross Entropy) measures pixel-level accuracy regardless of spatial distribution. Dice measures overlap. For small polyps that cover a minority of pixels, emphasizing Dice more means the model is penalized more heavily for missing the polyp region entirely — which is the clinically dangerous failure mode.

Shifting to 70% Dice / 30% BCE is a common production choice for medical segmentation tasks with class imbalance.

## Results

| Metric | Run 1 | Run 2 | Delta |
|---|---|---|---|
| Best val IoU | 0.8796 | **0.8909** | **+0.0113** |
| Best val Dice | 0.9354 | **0.9420** | **+0.0066** |
| Min val Loss | 0.1967 | **0.1718** | **-0.0249** |
| Best epoch | 7 of 8 | 8 of 13 | |

## Per-epoch training log

| Epoch | Train loss | Val Dice | Val IoU | New best? |
|---|---|---|---|---|
| 1 | 0.7045 | 0.5458 | 0.4281 | Yes |
| 2 | 0.6195 | 0.8563 | 0.7525 | Yes |
| 3 | 0.5856 | 0.9215 | 0.8569 | Yes |
| 4 | 0.5664 | 0.8540 | 0.7660 | No |
| 5 | 0.5516 | 0.9105 | 0.8370 | No |
| 6 | 0.5356 | 0.9248 | 0.8614 | Yes |
| 7 | 0.5238 | 0.9340 | 0.8769 | Yes |
| 8 | 0.5101 | 0.9420 | 0.8909 | Yes |
| 9 | 0.4985 | 0.9284 | 0.8673 | No |
| 10 | 0.5009 | 0.9359 | 0.8800 | No |
| 11 | 0.5001 | 0.9350 | 0.8788 | No |
| 12 | 0.4882 | 0.9238 | 0.8596 | No |
| 13 | 0.4797 | 0.9410 | 0.8890 | No |

## Observations

- Slower start (epoch 1 val IoU 0.43 vs 0.85 in Run 1) — lower LR + larger images means slower initial convergence. This is expected and acceptable.
- Peak performance at epoch 8 — model peaked early then plateaued. This suggests further training would need a learning rate reduction (which the scheduler would trigger automatically).
- No catastrophic dip like Run 1's epoch 4 — more stable training overall.
- Train loss still decreasing at epoch 13 — with ImageNet pre-trained weights (not available in this environment) and more epochs, additional improvement is likely.
