# U-Net Polyp Segmentation Pipeline

> **What this is:** A from-scratch medical image segmentation system that detects polyps in colonoscopy images. Built to demonstrate Computer Vision skills relevant to medical device software — specifically for roles requiring model training, MLOps, and clinical image processing.

---

## Table of Contents
1. [What the model does](#what-the-model-does)
2. [Why this problem matters](#why-this-problem-matters)
3. [Skills demonstrated](#skills-demonstrated)
4. [Project structure](#project-structure)
5. [Code explained for learners](#code-explained-for-learners)
6. [Experiment results](#experiment-results)
7. [How to run it](#how-to-run-it)
8. [Key concepts glossary](#key-concepts-glossary)
9. [Connection to Opti-TracT](#connection-to-opti-tract)

---

## What the model does

Given a colonoscopy frame, the model outputs a pixel-level mask showing exactly where polyps are located.

    Input:  RGB colonoscopy image  (H x W x 3)
    Output: Binary mask            (H x W x 1)  — 0 = background, 1 = polyp

This is called semantic segmentation — classifying every pixel individually, not just the whole image.

---

## Why this problem matters

Colorectal cancer is the second-leading cause of cancer death in the US. Polyps are precancerous growths — catching them early during colonoscopy is directly life-saving. AI-assisted polyp detection reduces miss rates (human endoscopists miss ~22% of polyps), provides real-time second opinions during procedures, and creates audit trails for clinical documentation.

Companies like Provation, Medtronic, and Fujifilm are actively building and deploying these systems in clinical settings.

---

## Skills demonstrated

| Skill | Where used | Why it matters |
|---|---|---|
| PyTorch Dataset / DataLoader | dataset.py | Core data pipeline skill for all production ML |
| Image augmentation | dataset.py — Albumentations | Prevents overfitting; essential for small medical datasets |
| Custom loss functions | losses.py — DiceLoss, BCEDiceLoss | Dice/IoU losses are the clinical standard for segmentation |
| U-Net architecture | model.py | Dominant architecture in medical image segmentation since 2015 |
| Skip connections | model.py — DecoderBlock | Why U-Net outperforms vanilla CNNs |
| Checkpoint management | train.py | Production model lifecycle: save best, resume training |
| LR scheduling | train.py — ReduceLROnPlateau | Prevents oscillation; standard in production |
| Evaluation metrics | losses.py — iou_score, dice_score | Required metrics for clinical validation |
| Explainability visualization | visualize.py | Required for FDA/CE regulatory submissions |
| Hyperparameter experiments | experiments/ | ML discipline — systematic comparison, not guessing |

---

## Project structure

    polyp_segmentation/
    |
    |-- model.py            <- U-Net architecture (built from scratch, fully explained)
    |-- dataset.py          <- Data loading pipeline with augmentation
    |-- losses.py           <- Dice loss, IoU metric, BCE+Dice combined loss
    |-- train.py            <- Full training loop (production U-Net, GPU)
    |-- train_light.py      <- Lightweight training loop (CPU-friendly)
    |-- visualize.py        <- Overlay predictions, training curve plots
    |
    |-- experiments/
    |   |-- run1_config.md  <- Baseline hyperparameter configuration
    |   |-- run2_config.md  <- Tuned configuration with rationale
    |   `-- COMPARISON.md   <- Side-by-side result analysis
    |
    |-- outputs/
    |   |-- run1_prediction_grid.png   <- Run 1 visual results
    |   |-- run1_training_curves.png   <- Run 1 training history
    |   |-- run2_prediction_grid.png   <- Run 2 visual results
    |   |-- run2_training_curves.png   <- Run 2 training history
    |   `-- comparison_report.png      <- Full side-by-side comparison
    |
    |-- data/kvasir-seg/    <- Dataset (not git-tracked — too large)
    |-- checkpoints/        <- Saved weights (not git-tracked)
    |-- logs/               <- Training history JSON
    `-- requirements.txt

---

## Code explained for learners

### dataset.py — How PyTorch loads your data

PyTorch does not know what your data looks like. You tell it by writing a Dataset class with three methods:

- __init__: called once — set up paths and transforms
- __len__: return total number of samples
- __getitem__: return ONE image+mask pair by index

The DataLoader wraps this and handles batching automatically. You write the logic for one sample; DataLoader handles the rest.

Augmentation creates fake variety by randomly flipping, rotating, and color-shifting images. This forces the model to learn the shape of a polyp rather than memorizing the exact training images — essential in medical imaging where datasets are small.

### losses.py — Why Dice loss instead of regular accuracy?

Imagine polyps cover 5% of pixels. A model predicting "no polyp anywhere" would get 95% pixel accuracy — but it is clinically useless.

Dice loss measures overlap, not pixel count:

    Dice = (2 x |prediction ∩ ground_truth|) / (|prediction| + |ground_truth|)

Perfect overlap = 1.0, no overlap = 0.0. We minimize Dice LOSS = 1 - Dice, which maximizes overlap.

IoU (Intersection over Union) is the evaluation metric:

    IoU = |prediction ∩ ground_truth| / |prediction ∪ ground_truth|

IoU is stricter than Dice. A score of 0.80+ is publication-quality for polyp segmentation.

### model.py — How U-Net works

U-Net has a distinctive U shape with two halves. The encoder shrinks the image while extracting features. The decoder expands back to full size. Skip connections copy feature maps directly from encoder to decoder at matching scales — the key innovation. Without them the decoder only knows "a polyp exists somewhere." With them it knows precise edges and boundaries.

### train.py — The training loop, one step at a time

    logits = model(images)        # 1. forward pass — model makes predictions
    loss = loss_fn(logits, masks) # 2. how wrong was it?
    optimizer.zero_grad()         # 3. clear old gradients (required every step)
    loss.backward()               # 4. compute new gradients via backpropagation
    optimizer.step()              # 5. update all model weights

Backpropagation computes "if I nudge this weight slightly, does loss go up or down?" and moves every weight in the direction that reduces loss. This happens automatically — you just call .backward().

---

## Experiment results

### Run 1 — Baseline

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.001 |
| Image size | 64 x 64 |
| Batch size | 4 |
| Epochs | 8 |
| Loss | BCE 50% + Dice 50% |

| Metric | Score |
|---|---|
| Best val Dice | 0.9354 |
| Best val IoU | 0.8796 |

### Run 2 — Tuned

| Hyperparameter | Value | Change | Rationale |
|---|---|---|---|
| Learning rate | 0.0003 | 3.3x lower | Finer gradient steps, less oscillation |
| Image size | 96 x 96 | 2.25x more pixels | More spatial detail preserved |
| Batch size | 6 | +50% | More stable gradient estimates |
| Epochs | 13 | +5 | Allow fuller convergence |
| BCE weight | 0.3 | down | Less emphasis on pixel accuracy |
| Dice weight | 0.7 | up | More emphasis on spatial overlap |

| Metric | Run 1 | Run 2 | Delta |
|---|---|---|---|
| Best val Dice | 0.9354 | 0.9420 | +0.0066 |
| Best val IoU | 0.8796 | 0.8909 | +0.0113 |
| Min val Loss | 0.1967 | 0.1718 | -0.0249 |

Key finding: Increasing the Dice loss weight from 0.5 to 0.7 had the largest single impact. Higher Dice weight produced masks with better boundary accuracy on smaller polyp regions — exactly the clinically important case.

---

## How to run it

Quick start (CPU, synthetic data):

    pip install -r requirements.txt
    python train_light.py
    python visualize.py

Full production run (GPU + real Kvasir-SEG data):

    # Download: https://datasets.simula.no/kvasir-seg/
    # Place images in data/kvasir-seg/images/
    # Place masks  in data/kvasir-seg/masks/
    # In train.py set: encoder_weights = "imagenet"
    python train.py
    python visualize.py

---

## Key concepts glossary

| Term | Plain-English definition |
|---|---|
| Segmentation | Classifying every pixel (vs. classification = one label per image) |
| U-Net | Architecture with encoder + decoder + skip connections for medical imaging |
| Encoder | The shrinking half — extracts features, reduces spatial size |
| Decoder | The expanding half — restores full resolution |
| Skip connection | Direct path that copies encoder features to the decoder at the same scale |
| Dice coefficient | Measure of overlap between two binary masks (0=none, 1=perfect) |
| IoU | Intersection over Union — standard evaluation metric for segmentation |
| Overfitting | Memorizing training data instead of learning general patterns |
| Augmentation | Random transforms that create artificial variety in training data |
| Checkpoint | Saved copy of model weights at a specific training step |
| Backpropagation | Algorithm that computes gradients for weight updates |
| Learning rate | How large each weight update step is |
| Batch size | Number of images processed before one gradient update |
| Epoch | One complete pass through the entire training dataset |
| Logit | Raw model output before sigmoid — any real value |
| Sigmoid | Squashes any value to 0–1 range (for probabilities) |

---

## Connection to Opti-TracT

This segmentation pipeline is the detection component of a broader surgical instrument tracking system. The Opti-TracT patent (optical tracking for surgical instruments) requires three capabilities: detection of which pixels belong to the instrument (this project), tracking to assign consistent IDs across video frames, and localization to estimate 3D position from 2D detections.

The U-Net approach used here for polyp segmentation adapts directly to surgical instrument segmentation — the architecture is identical, only the dataset changes. This pipeline demonstrates the core technical competency that underpins the full Opti-TracT system.
