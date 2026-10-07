"""
Evaluates the trained model on the TEST set (data never used during
training or validation) - producing the metrics that actually matter
for a medical AI project: confusion matrix, sensitivity, specificity,
precision, F1-score, and AUC-ROC.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

from dataset import GastricDataset, eval_transform
from model import build_model

# ---- CONFIG ----
CSV_PATH = "dataset_labels.csv"
MODEL_PATH = "gastric_cancer_model_finetuned.pth"
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ---- LOAD TEST DATASET ----
test_dataset = GastricDataset(CSV_PATH, split="test", transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---- LOAD TRAINED MODEL ----
model = build_model(num_classes=2, freeze_backbone=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---- RUN PREDICTIONS ON TEST SET ----
all_preds = []
all_labels = []
all_probs = []  # probability of "cancer" class, needed for AUC-ROC

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # index 1 = cancer class

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ---- CONFUSION MATRIX ----
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)
print(f"                 Predicted Non-Cancer   Predicted Cancer")
print(f"Actual Non-Cancer      {tn:6d}                {fp:6d}")
print(f"Actual Cancer          {fn:6d}                {tp:6d}")

# ---- CALCULATE METRICS ----
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall for cancer class
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
auc = roc_auc_score(all_labels, all_probs)

print("\n" + "="*50)
print("KEY METRICS")
print("="*50)
print(f"Accuracy:              {accuracy*100:.2f}%")
print(f"Sensitivity (Recall):  {sensitivity*100:.2f}%   <- % of actual cancer cases correctly caught")
print(f"Specificity:           {specificity*100:.2f}%   <- % of actual non-cancer cases correctly cleared")
print(f"Precision:             {precision*100:.2f}%   <- of flagged 'cancer' predictions, % actually correct")
print(f"F1-Score:               {f1:.4f}")
print(f"AUC-ROC:                {auc:.4f}   <- 1.0 = perfect, 0.5 = random guessing")

print("\n" + "="*50)
print("FULL CLASSIFICATION REPORT")
print("="*50)
print(classification_report(all_labels, all_preds, target_names=['non_cancer', 'cancer']))

# ---- PLOT CONFUSION MATRIX AS A HEATMAP ----
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')

labels_text = ['non_cancer', 'cancer']
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels_text)
ax.set_yticklabels(labels_text)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('Actual Label')
ax.set_title('Confusion Matrix - Test Set')

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)

plt.colorbar(im)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("\nConfusion matrix saved as confusion_matrix.png")
print("\nEvaluation complete.")