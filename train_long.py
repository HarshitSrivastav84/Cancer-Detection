"""
Long-run fine-tuning script.  Usage (in Colab):

    !python train_long.py --epochs 30 --tag ft30
    !python train_long.py --epochs 40 --tag ft40
    !python train_long.py --epochs 50 --tag ft50

What it does differently from train_finetune.py:
  - the number of epochs comes from the command line
  - cosine learning-rate decay, so the model settles instead of bouncing
  - the "best" model is chosen by validation AUC-ROC (steadier than accuracy)
  - everything is saved straight to Google Drive after every epoch, so a
    Colab disconnect cannot lose your work; add --resume to continue a run
  - per-epoch history is saved as CSV and as a curves PNG

Every run starts from the same weights (--init), so 30 vs 40 vs 50 epochs
is a fair comparison.  A 40-epoch run is NOT "30 epochs plus 10 more"; the
learning-rate schedule is stretched over the chosen number of epochs.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import GastricDataset, train_transform, eval_transform
from model import build_model

# ---- ARGUMENTS ----
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, required=True, help="total epochs, e.g. 30")
ap.add_argument("--tag", type=str, default=None, help="name for this run, e.g. ft30")
ap.add_argument("--lr", type=float, default=1e-5, help="starting learning rate")
ap.add_argument("--init", type=str, default="gastric_cancer_model.pth",
                help="weights to start from (the frozen-phase model)")
ap.add_argument("--save_dir", type=str, default=None,
                help="where to save; defaults to Google Drive if mounted")
ap.add_argument("--patience", type=int, default=0,
                help="stop if val AUC has not improved for this many epochs (0 = never stop early)")
ap.add_argument("--resume", action="store_true", help="continue from the last saved epoch")
ap.add_argument("--batch_size", type=int, default=32)
args = ap.parse_args()

TAG = args.tag or f"ft{args.epochs}"
CSV_PATH = "dataset_labels.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"          # mixed precision: faster on the T4

DRIVE_DIR = "/content/drive/My Drive/gastric_runs"
if args.save_dir:
    SAVE_DIR = args.save_dir
elif os.path.isdir("/content/drive/My Drive"):
    SAVE_DIR = DRIVE_DIR
else:
    SAVE_DIR = "runs"
    print("WARNING: Google Drive is not mounted, saving to ./runs (lost if the session ends).")
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_PATH = os.path.join(SAVE_DIR, f"gastric_cancer_model_{TAG}_best.pth")
LAST_PATH = os.path.join(SAVE_DIR, f"{TAG}_last.pt")
CSV_OUT = os.path.join(SAVE_DIR, f"{TAG}_history.csv")
PNG_OUT = os.path.join(SAVE_DIR, f"{TAG}_curves.png")

print(f"Run: {TAG} | epochs: {args.epochs} | lr: {args.lr} | device: {DEVICE} | AMP: {USE_AMP}")
print(f"Saving to: {SAVE_DIR}")
torch.manual_seed(42)

# ---- DATA ----
train_ds = GastricDataset(CSV_PATH, split="train", transform=train_transform)
val_ds = GastricDataset(CSV_PATH, split="val", transform=eval_transform)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=USE_AMP)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=USE_AMP)

# ---- CLASS WEIGHTS (same idea as before) ----
df = pd.read_csv(CSV_PATH)
y_train = df[df["split"] == "train"]["label"].map({"non_cancer": 0, "cancer": 1}).values
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
weights_t = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights (non_cancer, cancer): {weights}")

# ---- MODEL: unfrozen, starting from the frozen-phase weights ----
model = build_model(num_classes=2, freeze_backbone=False)
if os.path.exists(args.init):
    model.load_state_dict(torch.load(args.init, map_location=DEVICE))
    print(f"Started from {args.init}")
else:
    print(f"WARNING: {args.init} not found, starting from ImageNet weights only.")
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights_t)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# ---- RESUME ----
history, best_auc, no_improve, start_epoch = [], 0.0, 0, 0
if args.resume and os.path.exists(LAST_PATH):
    ck = torch.load(LAST_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    scaler.load_state_dict(ck["scaler"])
    history, best_auc = ck["history"], ck["best_auc"]
    no_improve, start_epoch = ck["no_improve"], ck["epoch"] + 1
    print(f"Resumed from epoch {start_epoch} (best val AUC so far {best_auc:.4f})")
elif args.resume:
    print("No checkpoint found for this tag, starting fresh.")

# ---- TRAIN ----
for epoch in range(start_epoch, args.epochs):
    t0 = time.time()
    lr_now = optimizer.param_groups[0]["lr"]

    model.train()
    running = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
    train_loss = running / len(train_loader)
    scheduler.step()

    # ---- validate ----
    model.eval()
    val_loss, ys, ps = 0.0, [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                out = model(images)
                val_loss += criterion(out, labels).item()
            ps.extend(torch.softmax(out.float(), dim=1)[:, 1].cpu().numpy())
            ys.extend(labels.cpu().numpy())
    val_loss /= len(val_loader)

    y, pr = np.array(ys), np.array(ps)
    pred = (pr >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    auc = roc_auc_score(y, pr)

    history.append(dict(epoch=epoch + 1, lr=lr_now, train_loss=train_loss, val_loss=val_loss,
                        val_acc=acc * 100, val_sens=sens * 100, val_spec=spec * 100, val_auc=auc))

    marker = ""
    if auc > best_auc:
        best_auc, no_improve = auc, 0
        torch.save(model.state_dict(), BEST_PATH)
        marker = "  <- new best (saved)"
    else:
        no_improve += 1

    print(f"Epoch {epoch+1}/{args.epochs} | lr {lr_now:.2e} | train {train_loss:.4f} | "
          f"val {val_loss:.4f} | acc {acc*100:.2f}% | sens {sens*100:.2f}% | "
          f"spec {spec*100:.2f}% | AUC {auc:.4f} | {time.time()-t0:.0f}s{marker}")

    # save progress every epoch (so a disconnect costs at most one epoch)
    pd.DataFrame(history).to_csv(CSV_OUT, index=False)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                    history=history, best_auc=best_auc, no_improve=no_improve,
                    epoch=epoch), LAST_PATH)

    if args.patience and no_improve >= args.patience:
        print(f"Early stop: val AUC did not improve for {args.patience} epochs.")
        break

# ---- CURVES ----
h = pd.DataFrame(history)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(h.epoch, h.train_loss, label="Train loss"); ax[0].plot(h.epoch, h.val_loss, label="Val loss")
ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(True)
ax[1].plot(h.epoch, h.val_auc, color="purple"); ax[1].set_title("Validation AUC-ROC")
ax[1].set_xlabel("Epoch"); ax[1].grid(True)
ax[2].plot(h.epoch, h.val_acc, label="Accuracy"); ax[2].plot(h.epoch, h.val_sens, label="Sensitivity")
ax[2].plot(h.epoch, h.val_spec, label="Specificity")
ax[2].set_title("Validation metrics at 0.50 threshold (%)"); ax[2].set_xlabel("Epoch")
ax[2].legend(); ax[2].grid(True)
plt.tight_layout(); plt.savefig(PNG_OUT, dpi=150)

best_row = h.loc[h.val_auc.idxmax()]
print(f"\nDone. Best validation AUC {best_row.val_auc:.4f} at epoch {int(best_row.epoch)}.")
print(f"Best weights: {BEST_PATH}")
print(f"Curves:       {PNG_OUT}")
print(f"Next: python evaluate_run.py --weights \"{BEST_PATH}\"")