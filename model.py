"""
model.py — U-Net Architecture (Two Versions)
=============================================
PURPOSE: Define the model we train. Two options are provided:

  1. LightUNet  — built from scratch in pure PyTorch (~500K parameters)
                  Great for LEARNING because every layer is visible and explained.
                  Runs on CPU with minimal RAM. Use this to understand U-Net.

  2. build_smp_unet() — production U-Net via segmentation_models_pytorch
                  Uses a pre-trained ResNet34 backbone (~24M parameters).
                  Use this on a GPU machine for real Kvasir-SEG performance.

WHAT IS U-NET?
  Invented in 2015 at University of Freiburg specifically for medical images.
  Two-part design:
    Encoder (left side): shrinks the image, extracts what features are present
    Decoder (right side): grows back to full size, places features spatially
    Skip connections: copy feature maps from encoder directly to decoder
                      so fine spatial detail isn't lost during shrinking

SKIP CONNECTIONS — the key insight:
  Without them: the decoder only knows "a polyp exists somewhere"
  With them:    the decoder knows "a polyp exists AND here are the edge pixels"
  This is why U-Net outperformed everything else in 2015 for segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILDING BLOCK: Double Convolution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DoubleConv(nn.Module):
    """
    Two consecutive Conv → BatchNorm → ReLU blocks.

    This is the basic repeating unit of U-Net. Every encoder and decoder
    step uses this block.

    WHY TWO CONVOLUTIONS?
        One convolution extracts local features.
        A second convolution combines those features, giving a larger
        effective receptive field without increasing the kernel size.

    WHY BATCHNORM?
        Normalizes activations across a batch, stabilizing training.
        Without it, deep networks have wildly varying activation magnitudes
        that make gradient descent unstable.

    WHY RELU?
        Activation function: sets all negative values to zero.
        This introduces non-linearity — without it, stacking conv layers
        would be mathematically equivalent to a single linear transformation.

    Args:
        in_ch  : number of input feature channels
        out_ch : number of output feature channels
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # Conv 1
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=3,   # 3x3 sliding window
                      padding=1,       # padding=1 keeps spatial size same
                      bias=False),     # bias=False because BatchNorm handles offset
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),     # inplace=True saves a little memory

            # Conv 2
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENCODER STEP: Downsample + DoubleConv
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EncoderBlock(nn.Module):
    """
    One step DOWN in the U-Net encoder:
        MaxPool (halve spatial size) → DoubleConv (extract features)

    After each encoder step:
        - Image gets SMALLER (256→128→64→32→16)
        - Feature depth gets DEEPER (32→64→128→256)

    Think of it like zooming out and extracting progressively more abstract
    patterns: edges → textures → shapes → semantic regions.

    MaxPool2d(2): takes the MAX value in each 2x2 block.
        Why max? Preserves the most activated (most feature-present) pixels.
        Equivalent to saying "something interesting was in this region."
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)         # halve H and W
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DECODER STEP: Upsample + concatenate skip + DoubleConv
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DecoderBlock(nn.Module):
    """
    One step UP in the U-Net decoder:
        ConvTranspose2d (double spatial size) → concat skip → DoubleConv

    ConvTranspose2d = "learnable upsampling"
        Inserts zeros between pixels then convolves — the model learns
        how to interpolate, rather than using a fixed formula like bilinear.

    SKIP CONNECTION:
        The encoder saved feature maps at this resolution.
        We concatenate (torch.cat) them with the upsampled decoder features.
        This gives the decoder BOTH:
          - Abstract semantic info from deep encoder (what it is)
          - Fine spatial detail from early encoder (exactly where it is)

    Args:
        in_ch  : channels coming IN from the layer below (upsampled)
        out_ch : channels going OUT after this block
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Upsample: double H and W, halve channels
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # After concat with skip, channels = in_ch // 2 + skip_ch
        # skip_ch == in_ch // 2 by our architecture design, so total = in_ch
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        """
        Args:
            x    : feature map from the layer below (smaller, deeper)
            skip : feature map from the matching encoder level (larger, shallower)
        """
        x = self.up(x)   # double the spatial size

        # Handle size mismatch: if encoder feature map is slightly bigger
        # (can happen when input size isn't a clean power of 2)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        # Concatenate along channel dimension (dim=1)
        # x shape: (B, C/2, H, W)  +  skip shape: (B, C/2, H, W)
        # → combined: (B, C, H, W)
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FULL LIGHT U-NET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LightUNet(nn.Module):
    """
    Lightweight U-Net for CPU training and learning.

    Architecture (for 256x256 input):

    ENCODER                          DECODER
    ──────────────────────────────   ──────────────────────────────
    Input  (3, 256, 256)
    enc1   (32, 256, 256)  ───────────────────────────► dec4 (32, 256, 256)
    enc2   (64, 128, 128)  ─────────────────────► dec3 (32, 128, 128)
    enc3   (128, 64, 64)   ─────────────────► dec2 (64, 64, 64)
    enc4   (256, 32, 32)   ─────────────► dec1 (128, 32, 32)
    bottle (256, 16, 16)   (bottleneck — most compressed representation)

    Final: Conv1x1 → (1, 256, 256) raw logit mask

    Total parameters: ~500K (vs 24M for ResNet34 U-Net)
    RAM needed: ~200MB during training
    """

    def __init__(self, in_channels=3, out_channels=1, features=(32, 64, 128, 256)):
        """
        Args:
            in_channels  : 3 for RGB images
            out_channels : 1 for binary segmentation (polyp vs background)
            features     : channel sizes at each encoder level
                           Larger = more capacity but more RAM and slower
        """
        super().__init__()

        # ── Encoder ──
        # First block: no pooling (input is already the right size)
        self.enc1 = DoubleConv(in_channels, features[0])   # 3  → 32
        self.enc2 = EncoderBlock(features[0], features[1]) # 32 → 64
        self.enc3 = EncoderBlock(features[1], features[2]) # 64 → 128
        self.enc4 = EncoderBlock(features[2], features[3]) # 128 → 256

        # Bottleneck: most compressed, most abstract representation
        self.bottleneck = EncoderBlock(features[3], features[3] * 2)  # 256 → 512

        # ── Decoder ──
        self.dec1 = DecoderBlock(features[3] * 2, features[3])  # 512 → 256
        self.dec2 = DecoderBlock(features[3],     features[2])  # 256 → 128
        self.dec3 = DecoderBlock(features[2],     features[1])  # 128 → 64
        self.dec4 = DecoderBlock(features[1],     features[0])  # 64  → 32

        # Final 1x1 convolution: map feature channels to output classes
        # 1x1 conv = applies a learned linear combination of channels per pixel
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Full forward pass through encoder → bottleneck → decoder.

        Args:
            x : input image tensor, shape (B, 3, H, W)

        Returns:
            raw logit mask, shape (B, 1, H, W)
            Values are NOT yet passed through sigmoid — that happens in the loss.
        """
        # ── Encoder: save skip connections ──
        s1 = self.enc1(x)           # (B, 32,  H,   W)
        s2 = self.enc2(s1)          # (B, 64,  H/2, W/2)
        s3 = self.enc3(s2)          # (B, 128, H/4, W/4)
        s4 = self.enc4(s3)          # (B, 256, H/8, W/8)

        # ── Bottleneck ──
        b  = self.bottleneck(s4)    # (B, 512, H/16, W/16)

        # ── Decoder: pass skip connections from matching encoder levels ──
        d  = self.dec1(b,  s4)      # (B, 256, H/8, W/8)
        d  = self.dec2(d,  s3)      # (B, 128, H/4, W/4)
        d  = self.dec3(d,  s2)      # (B, 64,  H/2, W/2)
        d  = self.dec4(d,  s1)      # (B, 32,  H,   W)

        # ── Final classification layer ──
        return self.final_conv(d)   # (B, 1, H, W)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FACTORY: choose model based on environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_model(use_lightweight=True):
    """
    Return the appropriate U-Net variant.

    Args:
        use_lightweight : True  → LightUNet (CPU-friendly, for learning)
                          False → segmentation_models_pytorch U-Net (GPU, production)

    Usage:
        model = build_model(use_lightweight=True)   # on your laptop/CPU
        model = build_model(use_lightweight=False)  # on Colab GPU or workstation
    """
    if use_lightweight:
        model = LightUNet(in_channels=3, out_channels=1)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  LightUNet loaded — {n_params:,} parameters (~CPU-friendly)")
        return model
    else:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name    = "resnet34",
            encoder_weights = "imagenet",
            in_channels     = 3,
            classes         = 1,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ResNet34 U-Net loaded — {n_params:,} parameters (GPU recommended)")
        return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quick sanity check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    model = LightUNet()
    dummy = torch.randn(2, 3, 256, 256)   # batch of 2 fake images
    out   = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")          # expect (2, 1, 256, 256)
    assert out.shape == (2, 1, 256, 256), "Shape mismatch!"
    print("Model forward pass: OK")
