"""
Evaluates the fine-tuned model on the TEST set using the optimized
threshold (0.65) found via threshold_search.py on the validation set.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np

from dataset import GastricDataset, eval_transform
from model import build_model

# ---- CONFIG ----
CSV_PATH = "dataset_labels.csv"
MODEL_PATH = "gastric_cancer_model_finetuned.pth"
BATCH_SIZE = 32
THRESHOLD = 0.65  # found via threshold_search.py on the validation set
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")
print(f"Using classification threshold: {THRESHOLD}")

test_dataset = GastricDataset(CSV_PATH, split="test", transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = build_model(num_classes=2, freeze_backbone=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
all_preds = (all_probs >= THRESHOLD).astype(int)

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
auc = roc_auc_score(all_labels, all_probs)

print("\n" + "="*50)
print(f"CONFUSION MATRIX (threshold={THRESHOLD})")
print("="*50)
print(f"                 Predicted Non-Cancer   Predicted Cancer")
print(f"Actual Non-Cancer      {tn:6d}                {fp:6d}")
print(f"Actual Cancer          {fn:6d}                {tp:6d}")

print("\n" + "="*50)
print("KEY METRICS ON TEST SET (final, honest result)")
print("="*50)
print(f"Accuracy:              {accuracy*100:.2f}%")
print(f"Sensitivity (Recall):  {sensitivity*100:.2f}%")
print(f"Specificity:           {specificity*100:.2f}%")
print(f"Precision:             {precision*100:.2f}%")
print(f"F1-Score:               {f1:.4f}")
print(f"AUC-ROC:                {auc:.4f}")