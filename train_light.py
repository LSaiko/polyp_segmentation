"""
train_light.py — Training loop using LightUNet (CPU-safe)
==========================================================
Same logic as train.py but uses our custom LightUNet instead of
the heavy ResNet34 backbone. Runs on any machine with no GPU required.

WHEN TO USE WHICH:
    train_light.py    → learning, debugging, CPU-only machines
    train.py          → production run, GPU machine, real Kvasir-SEG data

RUN:
    python train_light.py
"""

import os, time, json, torch
import torch.optim as optim

from model   import LightUNet
from dataset import create_dataloaders
from losses  import BCEDiceLoss, dice_score, iou_score, pixel_accuracy

CONFIG = {
    "data_root"          : "data/kvasir-seg",
    "checkpoint_dir"     : "checkpoints",
    "log_dir"            : "logs",
    "img_size"           : 128,   # smaller than full 256 — faster on CPU
    "batch_size"         : 4,     # small batch for CPU
    "epochs"             : 15,
    "lr"                 : 1e-3,  # slightly higher lr for training from scratch
    "val_split"          : 0.2,
    "bce_weight"         : 0.5,
    "dice_weight"        : 0.5,
    "scheduler_patience" : 4,
    "scheduler_factor"   : 0.5,
}


class RunningMetrics:
    def __init__(self):
        self.totals, self.counts = {}, {}
    def update(self, **kw):
        for k, v in kw.items():
            self.totals[k] = self.totals.get(k, 0.0) + float(v)
            self.counts[k] = self.counts.get(k, 0)   + 1
    def averages(self):
        return {k: self.totals[k] / self.counts[k] for k in self.totals}


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    m = RunningMetrics()
    for i, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss   = loss_fn(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            m.update(loss=loss.item(),
                     dice=dice_score(logits, masks),
                     iou=iou_score(logits, masks))
        if (i+1) % max(1, len(loader)//3) == 0 or (i+1)==len(loader):
            a = m.averages()
            print(f"    step {i+1}/{len(loader)} | loss={a['loss']:.4f}  dice={a['dice']:.4f}  iou={a['iou']:.4f}")
    return m.averages()


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    m = RunningMetrics()
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        m.update(loss=loss_fn(logits, masks).item(),
                 dice=dice_score(logits, masks),
                 iou=iou_score(logits, masks),
                 accuracy=pixel_accuracy(logits, masks))
    return m.averages()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  LightUNet Training  |  device={device}  |  img_size={CONFIG['img_size']}")
    print(f"{'='*60}\n")

    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        CONFIG["data_root"], CONFIG["img_size"],
        CONFIG["batch_size"], CONFIG["val_split"]
    )

    print("\nBuilding LightUNet...")
    model     = LightUNet().to(device)
    loss_fn   = BCEDiceLoss(CONFIG["bce_weight"], CONFIG["dice_weight"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=CONFIG["scheduler_patience"],
        factor=CONFIG["scheduler_factor"]
    )

    best_iou, history = 0.0, []
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    best_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")
    last_path = os.path.join(CONFIG["checkpoint_dir"], "last_model.pt")

    print(f"\nTraining for {CONFIG['epochs']} epochs...\n")

    for epoch in range(1, CONFIG["epochs"]+1):
        t0 = time.time()
        print(f"Epoch {epoch}/{CONFIG['epochs']}")
        print("-"*50)

        print("  [Train]")
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print("  [Validate]")
        vl = validate(model, val_loader, loss_fn, device)
        scheduler.step(vl["loss"])
        lr = optimizer.param_groups[0]["lr"]

        print(f"\n  {'Metric':<12} {'Train':>8} {'Val':>8}")
        print(f"  {'-'*30}")
        for k in ["loss","dice","iou"]:
            print(f"  {k:<12} {tr.get(k,0):>8.4f} {vl.get(k,0):>8.4f}")
        print(f"  lr={lr:.2e}  time={time.time()-t0:.1f}s")

        # Save last checkpoint always
        torch.save({"epoch":epoch,"model_state":model.state_dict(),
                    "optimizer_state":optimizer.state_dict(),
                    "metrics":vl,"config":CONFIG}, last_path)

        # Save best checkpoint if val IoU improved
        is_best = vl["iou"] > best_iou
        if is_best:
            best_iou = vl["iou"]
            torch.save({"epoch":epoch,"model_state":model.state_dict(),
                        "optimizer_state":optimizer.state_dict(),
                        "metrics":vl,"config":CONFIG}, best_path)
            print(f"  ✓ New best! val_iou={best_iou:.4f} saved → {best_path}")

        history.append({"epoch":epoch,"lr":lr,"train":tr,"val":vl,"is_best":is_best})
        with open(os.path.join(CONFIG["log_dir"],"history.json"),"w") as f:
            json.dump(history, f, indent=2)
        print()

    print("="*60)
    print(f"Done. Best val IoU: {best_iou:.4f}")
    print(f"Best checkpoint: {best_path}")
    print("="*60)
    return model, history


if __name__ == "__main__":
    train()
