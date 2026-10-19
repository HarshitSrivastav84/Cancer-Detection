"""
Final evaluation for the report.

1. Loads the best-AUC weights.
2. Chooses the decision threshold on the VALIDATION set only.
3. Evaluates the TEST set once, at the default 0.50 threshold and at the
   chosen threshold, and prints accuracy, sensitivity, specificity, AUC.

Usage (Colab):
    !python final_eval.py --weights "/content/drive/My Drive/gastric_runs/gastric_cancer_model_ft40_best.pth"
    !python final_eval.py --weights "..." --target_acc 0.90
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import GastricDataset, eval_transform
from model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--target_acc", type=float, default=0.91,
                help="validation accuracy to reach; the threshold with the "
                     "highest sensitivity that still reaches it is chosen")
args = ap.parse_args()

CSV_PATH = "dataset_labels.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(num_classes=2, freeze_backbone=False)
model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
model = model.to(DEVICE).eval()


def get_probs(split):
    ds = GastricDataset(CSV_PATH, split=split, transform=eval_transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    ys, ps = [], []
    with torch.no_grad():
        for images, labels in loader:
            out = model(images.to(DEVICE))
            ps.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            ys.extend(labels.numpy())
    return np.array(ys), np.array(ps)


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return dict(acc=(tp + tn) / len(y),
                sens=tp / (tp + fn) if tp + fn else 0.0,
                spec=tn / (tn + fp) if tn + fp else 0.0,
                prec=tp / (tp + fp) if tp + fp else 0.0,
                tp=tp, tn=tn, fp=fp, fn=fn)


# ---- 1. choose the threshold on VALIDATION ----
yv, pv = get_probs("val")
best, fallback = None, None
for t in np.arange(0.05, 0.96, 0.01):
    m = metrics(yv, pv, t)
    if fallback is None or m["acc"] > fallback[1]["acc"]:
        fallback = (t, m)
    if m["acc"] >= args.target_acc and (best is None or m["sens"] > best[1]["sens"]):
        best = (t, m)

if best is None:
    thr, mv = fallback
    print(f"\nNo threshold reached {args.target_acc*100:.0f}% validation accuracy. "
          f"Using the most accurate one instead.")
else:
    thr, mv = best
print(f"\nChosen threshold (from validation): {thr:.2f}")
print(f"  validation accuracy {mv['acc']*100:.2f}% | sensitivity {mv['sens']*100:.2f}% "
      f"| specificity {mv['spec']*100:.2f}%")

# ---- 2. evaluate TEST once ----
yt, pt = get_probs("test")
auc = roc_auc_score(yt, pt)
baseline = max((yt == 0).mean(), (yt == 1).mean())

print("\n" + "=" * 60)
print("TEST SET RESULTS (unseen data)")
print("=" * 60)
print(f"AUC-ROC: {auc:.4f}")
print(f"Majority-class baseline (always predict non-cancer): {baseline*100:.2f}%\n")
for name, t in [("Default threshold 0.50", 0.50), (f"Chosen threshold {thr:.2f}", thr)]:
    m = metrics(yt, pt, t)
    print(f"{name}")
    print(f"  Accuracy    {m['acc']*100:.2f}%")
    print(f"  Sensitivity {m['sens']*100:.2f}%   (cancer cases caught)")
    print(f"  Specificity {m['spec']*100:.2f}%")
    print(f"  Precision   {m['prec']*100:.2f}%")
    print(f"  TN {m['tn']}  FP {m['fp']}  FN {m['fn']}  TP {m['tp']}\n")

# ---- 3. confusion matrix at the chosen threshold ----
m = metrics(yt, pt, thr)
cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["non_cancer", "cancer"]); ax.set_yticklabels(["non_cancer", "cancer"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Test confusion matrix (threshold {thr:.2f})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im); plt.tight_layout()
plt.savefig("confusion_matrix_final.png", dpi=150)
print("Saved confusion_matrix_final.png")
print(f"Use this threshold in the app:  $env:THRESHOLD=\"{thr:.2f}\"; python app.py")