"""
visualize.py — Overlay Predictions on Colonoscopy Images
=========================================================
PURPOSE: Show WHAT the model learned by overlaying its predictions
         on the original colonoscopy images. This is essential for:
         - Debugging (is the model looking at the right region?)
         - Presentations (hiring managers want to SEE it working)
         - FDA/regulatory submissions (explainability is required)

THREE VISUALIZATION TYPES:
  1. Side-by-side grid  → image | ground truth | prediction | overlay
  2. Blended overlay    → semi-transparent colored mask on original image
  3. Training curves    → loss/dice/IoU plotted over epochs

RUN THIS FILE:
  python visualize.py
  # Outputs saved to: outputs/
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # use non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE: Overlay predicted mask on original image
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def overlay_mask_on_image(
    image_np,       # original colonoscopy image as numpy array (H, W, 3), uint8
    pred_mask_np,   # predicted binary mask (H, W), values 0 or 1, float32
    gt_mask_np=None,# ground truth mask (H, W), optional, for comparison
    pred_color=(0, 200, 100),   # green overlay for prediction
    gt_color=(255, 80, 80),     # red overlay for ground truth
    alpha=0.45,     # transparency: 0=invisible, 1=opaque
):
    """
    Blend a colored segmentation mask onto the original image.

    HOW IT WORKS (the "alpha blending" formula):
        blended_pixel = original_pixel * (1 - alpha) + mask_color * alpha

        When alpha = 0.45:
            You see 55% of the original image + 45% of the colored overlay
            This lets you see both the tissue texture AND the mask simultaneously

    Args:
        image_np    : RGB colonoscopy frame, dtype uint8, values 0-255
        pred_mask_np: predicted mask, dtype float32, values 0.0 or 1.0
        gt_mask_np  : ground truth mask (optional — for comparison overlays)
        pred_color  : RGB tuple for prediction overlay (default: green)
        gt_color    : RGB tuple for ground truth overlay (default: red)
        alpha       : blend transparency (0.3-0.5 works best)

    Returns:
        numpy array (H, W, 3) uint8: the blended image
    """
    # Work on a copy so we don't modify the original
    overlay = image_np.copy().astype(np.float32)
    H, W    = image_np.shape[:2]

    # ── Apply ground truth mask (red) if provided ──
    if gt_mask_np is not None:
        gt_bool = gt_mask_np > 0.5   # boolean mask: True where polyp
        for c, color_val in enumerate(gt_color):
            overlay[gt_bool, c] = (
                overlay[gt_bool, c] * (1 - alpha) + color_val * alpha
            )

    # ── Apply prediction mask (green) ──
    pred_bool = pred_mask_np > 0.5
    for c, color_val in enumerate(pred_color):
        overlay[pred_bool, c] = (
            overlay[pred_bool, c] * (1 - alpha) + color_val * alpha
        )

    return overlay.astype(np.uint8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VISUALIZATION: 4-panel comparison grid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_prediction_grid(
    images_np,       # list of original images (H, W, 3) uint8
    gt_masks_np,     # list of ground truth masks (H, W) float32
    pred_masks_np,   # list of predicted masks (H, W) float32
    dice_scores,     # list of per-image dice scores
    iou_scores,      # list of per-image IoU scores
    save_path,       # where to save the output PNG
    n_samples=4,     # how many rows to show
):
    """
    Creates a grid with 4 columns per sample:
        [Original] [Ground Truth] [Prediction] [Overlay]

    This is the standard visualization for medical segmentation papers
    and portfolio presentations.
    """
    n = min(n_samples, len(images_np))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    # Handle case where n=1 (axes would be 1D, not 2D)
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original image", "Ground truth mask", "Predicted mask", "Overlay (pred=green)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for row in range(n):
        img  = images_np[row]           # (H, W, 3)
        gt   = gt_masks_np[row]         # (H, W)
        pred = pred_masks_np[row]       # (H, W)
        d    = dice_scores[row]
        iou  = iou_scores[row]

        # Column 0: Original colonoscopy image
        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f"Sample {row+1}", fontsize=9)

        # Column 1: Ground truth mask (white=polyp, black=background)
        axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)

        # Column 2: Predicted mask
        # Color-code by correctness:
        #   True Positive  (both pred & gt = 1) → green
        #   False Positive (pred=1, gt=0)        → red (over-predicting)
        #   False Negative (pred=0, gt=1)        → blue (missed polyp)
        #   True Negative  (both = 0)            → black
        tp = ((pred > 0.5) & (gt > 0.5)).astype(float)  # correctly found
        fp = ((pred > 0.5) & (gt < 0.5)).astype(float)  # false alarm
        fn = ((pred < 0.5) & (gt > 0.5)).astype(float)  # missed polyp

        error_map = np.zeros((*gt.shape, 3))
        error_map[..., 1] += tp        # green channel: true positives
        error_map[..., 0] += fp        # red channel: false positives
        error_map[..., 2] += fn        # blue channel: false negatives

        axes[row, 2].imshow(error_map, vmin=0, vmax=1)

        # Column 3: Blended overlay on original
        blended = overlay_mask_on_image(img, pred, gt_mask_np=gt)
        axes[row, 3].imshow(blended)

        # Add metric scores to the prediction column title
        axes[row, 2].set_xlabel(
            f"Dice={d:.3f}  IoU={iou:.3f}",
            fontsize=9, color="#333333"
        )

    # Add color legend for the error map column
    legend_elements = [
        mpatches.Patch(color="green", label="True positive (correct)"),
        mpatches.Patch(color="red",   label="False positive (over-predicted)"),
        mpatches.Patch(color="blue",  label="False negative (missed polyp)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01)
    )

    # Remove individual axis ticks/labels (cleaner look)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "U-Net Polyp Segmentation — Prediction Analysis",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Prediction grid saved → {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VISUALIZATION: Training curves
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_training_curves(history, save_path):
    """
    Plot loss, Dice, and IoU over training epochs for both
    training and validation sets.

    HOW TO READ THESE CURVES:
        - Training loss going DOWN = model is learning
        - Validation loss going DOWN = generalization is improving
        - Val loss RISING while train loss falls = OVERFITTING (bad!)
          → solution: more augmentation, less epochs, more data

    Args:
        history   : list of epoch dicts from train.py
        save_path : output PNG file path
    """
    epochs     = [h["epoch"]         for h in history]
    train_loss = [h["train"]["loss"]  for h in history]
    val_loss   = [h["val"]["loss"]    for h in history]
    train_dice = [h["train"]["dice"]  for h in history]
    val_dice   = [h["val"]["dice"]    for h in history]
    train_iou  = [h["train"]["iou"]   for h in history]
    val_iou    = [h["val"]["iou"]     for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Color scheme: blue=train, orange=validation
    train_color = "#2563eb"
    val_color   = "#f97316"

    # Panel 1: Loss
    axes[0].plot(epochs, train_loss, color=train_color, linewidth=2, label="Train")
    axes[0].plot(epochs, val_loss,   color=val_color,   linewidth=2, label="Validation", linestyle="--")
    axes[0].set_title("Loss (lower is better)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE + Dice Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Dice Score
    axes[1].plot(epochs, train_dice, color=train_color, linewidth=2, label="Train")
    axes[1].plot(epochs, val_dice,   color=val_color,   linewidth=2, label="Validation", linestyle="--")
    axes[1].set_title("Dice Score (higher is better)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Coefficient")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    # Mark best val dice
    best_epoch = epochs[val_dice.index(max(val_dice))]
    best_val   = max(val_dice)
    axes[1].axvline(x=best_epoch, color="gray", linestyle=":", alpha=0.7)
    axes[1].annotate(
        f"Best: {best_val:.3f}\n(epoch {best_epoch})",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 0.5, best_val - 0.05),
        fontsize=8, color="gray"
    )

    # Panel 3: IoU Score
    axes[2].plot(epochs, train_iou, color=train_color, linewidth=2, label="Train")
    axes[2].plot(epochs, val_iou,   color=val_color,   linewidth=2, label="Validation", linestyle="--")
    axes[2].set_title("IoU Score (higher is better)", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Intersection over Union")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Shade the gap between train and val on IoU (shows generalization gap)
    axes[2].fill_between(
        epochs, train_iou, val_iou,
        alpha=0.1, color="red",
        label="Generalization gap"
    )

    plt.suptitle("U-Net Training Curves — Polyp Segmentation", fontweight="bold", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved → {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INFERENCE HELPER: run model on a single image file
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def predict_single_image(model, image_path, img_size=256, device="cpu", threshold=0.5):
    """
    Run inference on one image and return the raw image + predicted mask.

    This is what you'd wrap in a FastAPI endpoint for production:
        POST /predict → returns mask JSON or overlay image

    Args:
        model      : loaded U-Net model (in eval mode)
        image_path : path to a colonoscopy image file
        img_size   : must match the size used during training
        device     : "cuda" or "cpu"
        threshold  : probability cutoff for polyp vs background

    Returns:
        image_np  : original image as numpy (H, W, 3) uint8
        pred_mask : binary predicted mask (img_size, img_size) float32
    """
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Load and preprocess
    image_np = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    tensor   = transform(image=image_np)["image"].unsqueeze(0).to(device)
    # shape: (1, 3, H, W) — the unsqueeze adds the batch dimension

    # Inference
    model.eval()
    with torch.no_grad():
        logits    = model(tensor)                               # (1, 1, H, W)
        prob_map  = torch.sigmoid(logits).squeeze().cpu().numpy()   # (H, W) values 0-1
        pred_mask = (prob_map > threshold).astype(np.float32)      # (H, W) binary

    # Resize original image to match output
    image_resized = np.array(
        Image.fromarray(image_np).resize((img_size, img_size)),
        dtype=np.uint8
    )

    return image_resized, pred_mask, prob_map


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN: Run full visualization pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_visualization(
    checkpoint_path = "checkpoints/best_model.pt",
    img_dir         = "data/kvasir-seg/images",
    mask_dir        = "data/kvasir-seg/masks",
    history_path    = "logs/history.json",
    output_dir      = "outputs",
    n_samples       = 4,
    img_size        = 256,
):
    """
    Load the best trained model and generate:
        1. outputs/prediction_grid.png  — 4-panel comparison grid
        2. outputs/training_curves.png  — loss/dice/IoU over epochs

    Args:
        checkpoint_path : path to saved .pt checkpoint
        img_dir         : folder of colonoscopy images
        mask_dir        : folder of binary masks
        history_path    : JSON log from training
        output_dir      : where to save visualization PNGs
        n_samples       : how many images to visualize
        img_size        : must match training image size
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    print(f"\nLoading model from {checkpoint_path}...")
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,   # don't re-download ImageNet weights
        in_channels     = 3,
        classes         = 1,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    saved_epoch = checkpoint["epoch"]
    saved_iou   = checkpoint["metrics"].get("iou", 0)
    print(f"  Loaded checkpoint: epoch {saved_epoch}, val_iou={saved_iou:.4f}")

    # ── Run inference on sample images ──
    print(f"\nRunning inference on {n_samples} validation images...")
    filenames = sorted(os.listdir(img_dir))[-n_samples:]   # use last n (val set)

    all_images = []
    all_gt     = []
    all_pred   = []
    all_dice   = []
    all_iou    = []

    for fname in filenames:
        img_path  = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        img_np, pred_mask, prob_map = predict_single_image(
            model, img_path, img_size=img_size, device=device
        )

        # Load ground truth mask
        gt_np = np.array(
            Image.open(mask_path).convert("L").resize((img_size, img_size)),
            dtype=np.float32
        )
        gt_np = (gt_np > 127).astype(np.float32)

        # Compute metrics
        eps = 1e-6
        pred_b = pred_mask.flatten()
        gt_b   = gt_np.flatten()
        intersection = (pred_b * gt_b).sum()
        dice = (2 * intersection + eps) / (pred_b.sum() + gt_b.sum() + eps)
        union = pred_b.sum() + gt_b.sum() - intersection
        iou  = (intersection + eps) / (union + eps)

        all_images.append(img_np)
        all_gt.append(gt_np)
        all_pred.append(pred_mask)
        all_dice.append(float(dice))
        all_iou.append(float(iou))

        print(f"  {fname}: dice={dice:.4f}  iou={iou:.4f}")

    # ── Prediction grid ──
    print("\nGenerating prediction grid...")
    plot_prediction_grid(
        all_images, all_gt, all_pred,
        all_dice, all_iou,
        save_path=os.path.join(output_dir, "prediction_grid.png"),
        n_samples=n_samples,
    )

    # ── Training curves ──
    if os.path.exists(history_path):
        print("Generating training curves...")
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(
            history,
            save_path=os.path.join(output_dir, "training_curves.png")
        )
    else:
        print(f"  Skipping training curves (history not found at {history_path})")

    print(f"\nAll outputs saved to ./{output_dir}/")


if __name__ == "__main__":
    run_visualization()
