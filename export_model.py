"""
export_model.py — Export trained checkpoint to a deployable model.pt
=====================================================================
PURPOSE:
    train_light.py saves a full checkpoint dict:
        {"epoch": N, "model_state": OrderedDict, "optimizer_state": ..., ...}

    The FastAPI server (main.py) can load EITHER format, but plain state-dict
    files are smaller, portable, and the convention expected by most serving
    frameworks (TorchServe, ONNX export, etc.).

    This script extracts just the weights and saves them to model.pt,
    ready to be mounted into the Docker container.

USAGE:
    # Basic — uses default paths
    python export_model.py

    # Custom paths
    python export_model.py --checkpoint checkpoints/best_model.pt --out model.pt

    # Also export to ONNX for cross-platform serving (TensorRT, ORT, etc.)
    python export_model.py --onnx

WHAT model.pt contains after export:
    OrderedDict of parameter name → weight tensor
    (exactly what model.load_state_dict() expects)
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from model import LightUNet


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = Path("checkpoints/best_model.pt")
DEFAULT_OUTPUT     = Path("model.pt")
DEFAULT_ONNX_OUT   = Path("model.onnx")
IMG_SIZE           = 256   # must match the size used during training


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint and return as dict with 'model_state' and metadata."""
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found at {checkpoint_path.resolve()}")
        sys.exit(1)

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and "model_state" in raw:
        # Full checkpoint saved by train_light.py
        return raw
    else:
        # Already a plain state dict — wrap it so callers are consistent
        return {"model_state": raw, "epoch": "?", "metrics": {}}


def export_state_dict(checkpoint: dict, output_path: Path) -> None:
    """Save just the model weights to output_path."""
    torch.save(checkpoint["model_state"], output_path)
    size_mb = output_path.stat().st_size / 1_024 / 1_024
    epoch   = checkpoint.get("epoch", "?")
    metrics = checkpoint.get("metrics", {})
    iou     = metrics.get("iou", "N/A")
    dice    = metrics.get("dice", "N/A")

    print(f"\n  Exported weights  → {output_path}  ({size_mb:.1f} MB)")
    print(f"  Source epoch      : {epoch}")
    if iou != "N/A":
        print(f"  Val IoU           : {float(iou):.4f}")
    if dice != "N/A":
        print(f"  Val Dice          : {float(dice):.4f}")
    print(f"\n  Mount into Docker with:")
    print(f"    docker run -p 8000:8000 -v $(pwd)/{output_path}:/app/model.pt \\")
    print(f"      ghcr.io/lsaiko/polyp_segmentation:latest")


def export_onnx(checkpoint: dict, output_path: Path) -> None:
    """Export to ONNX for cross-platform / TensorRT serving."""
    model = LightUNet(in_channels=3, out_channels=1)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version    = 17,
        input_names      = ["image"],          # (B, 3, H, W)  normalised RGB
        output_names     = ["logits"],         # (B, 1, H, W)  raw logits — apply sigmoid
        dynamic_axes     = {
            "image":  {0: "batch"},
            "logits": {0: "batch"},
        },
        export_params    = True,
        do_constant_folding = True,
    )

    size_mb = output_path.stat().st_size / 1_024 / 1_024
    print(f"\n  Exported ONNX     → {output_path}  ({size_mb:.1f} MB)")
    print(f"  Input  : image   (batch, 3, {IMG_SIZE}, {IMG_SIZE})  float32  ImageNet-normalised")
    print(f"  Output : logits  (batch, 1, {IMG_SIZE}, {IMG_SIZE})  float32  — apply sigmoid for probabilities")

    # Quick shape sanity check
    try:
        import onnx
        m = onnx.load(str(output_path))
        onnx.checker.check_model(m)
        print("  ONNX model check  : PASSED")
    except ImportError:
        print("  (install onnx to run model check: pip install onnx)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LightUNet checkpoint to a deployable model.pt (or ONNX)."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help=f"Path to the .pt checkpoint file (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output path for the plain state-dict file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--onnx", action="store_true",
        help="Also export an ONNX model alongside the state-dict",
    )
    parser.add_argument(
        "--onnx-out", type=Path, default=DEFAULT_ONNX_OUT,
        help=f"Output path for the ONNX file (default: {DEFAULT_ONNX_OUT})",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)

    export_state_dict(checkpoint, args.out)

    if args.onnx:
        export_onnx(checkpoint, args.onnx_out)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
