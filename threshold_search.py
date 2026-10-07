"""
Searches for the best classification threshold on the VALIDATION set
(not test set, to keep the final test evaluation honest).

Finds the threshold that gives the BEST ACCURACY while keeping
sensitivity above a minimum bar you set - rather than blindly using
the default 50% cutoff.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

from dataset import GastricDataset, eval_transform
from model import build_model

# ---- CONFIG ----
CSV_PATH = "dataset_labels.csv"
MODEL_PATH = "gastric_cancer_model_finetuned.pth"
BATCH_SIZE = 32
MIN_SENSITIVITY = 0.80  # the minimum sensitivity we're willing to accept
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ---- LOAD VALIDATION SET (searching here, not on test set) ----
val_dataset = GastricDataset(CSV_PATH, split="val", transform=eval_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---- LOAD FINE-TUNED MODEL ----
model = build_model(num_classes=2, freeze_backbone=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---- GET PROBABILITIES ON VALIDATION SET ----
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # probability of "cancer"

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ---- TRY MANY THRESHOLDS, FIND THE BEST ONE ----
print(f"\nSearching thresholds (minimum sensitivity required: {MIN_SENSITIVITY*100:.0f}%)...\n")
print(f"{'Threshold':<12}{'Accuracy':<12}{'Sensitivity':<14}{'Specificity':<12}")
print("-" * 50)

best_threshold = 0.5
best_accuracy = 0.0
results = []

for threshold in np.arange(0.10, 0.91, 0.05):
    preds = (all_probs >= threshold).astype(int)
    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    results.append((threshold, accuracy, sensitivity, specificity))
    print(f"{threshold:<12.2f}{accuracy*100:<12.2f}{sensitivity*100:<14.2f}{specificity*100:<12.2f}")

    # Only consider this threshold if it meets our minimum sensitivity requirement
    if sensitivity >= MIN_SENSITIVITY and accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print("\n" + "="*50)
print(f"BEST THRESHOLD (meeting {MIN_SENSITIVITY*100:.0f}%+ sensitivity): {best_threshold:.2f}")
print(f"Resulting Accuracy: {best_accuracy*100:.2f}%")
print("="*50)
print(f"\nUpdate your app.py / evaluate.py to use threshold={best_threshold:.2f}")
print("instead of the default 0.5, then re-run evaluate.py on the TEST set")
print("to confirm this threshold's real performance on unseen data.")