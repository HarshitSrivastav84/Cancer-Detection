"""
Evaluate any saved model on the TEST set at one or more thresholds.

    !python evaluate_run.py --weights "/content/drive/My Drive/gastric_runs/gastric_cancer_model_ft30_best.pth"
    !python evaluate_run.py --weights gastric_cancer_model_finetuned.pth --thresholds 0.5 0.55 0.65

Run this once per finished run and compare the results side by side.
Choose your final model and threshold using VALIDATION results, then
report the test numbers once.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dataset import GastricDataset, eval_transform
from model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--thresholds", type=float, nargs="+", default=[0.50, 0.65])
ap.add_argument("--csv", default="dataset_labels.csv")
args = ap.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = GastricDataset(args.csv, split="test", transform=eval_transform)
loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

model = build_model(num_classes=2, freeze_backbone=False)
model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
model = model.to(DEVICE).eval()

ys, ps = [], []
with torch.no_grad():
    for images, labels in loader:
        out = model(images.to(DEVICE))
        ps.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
        ys.extend(labels.numpy())
y, pr = np.array(ys), np.array(ps)

print(f"\nWeights: {args.weights}")
print(f"AUC-ROC (test): {roc_auc_score(y, pr):.4f}\n")
print(f"{'Threshold':<10}{'Accuracy':<10}{'Sens.':<9}{'Spec.':<9}{'Precision':<11}{'Missed':<8}{'False alarms'}")
for t in args.thresholds:
    pred = (pr >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    print(f"{t:<10.2f}{acc*100:<10.2f}{sens*100:<9.2f}{spec*100:<9.2f}{prec*100:<11.2f}{fn:<8d}{fp}")