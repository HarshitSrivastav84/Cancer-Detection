"""
Phase 2 training: Full fine-tuning.
Unfreezes the entire backbone and continues training with a much smaller
learning rate, starting from your already-trained model.

Run this AFTER your original train.py has already produced
gastric_cancer_model.pth - this script loads that model and improves it
further, rather than starting from scratch.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

from dataset import GastricDataset, train_transform, eval_transform
from model import build_model

# ---- CONFIG ----
CSV_PATH = "dataset_labels.csv"
PREVIOUS_MODEL_PATH = "gastric_cancer_model.pth"  # your already-trained model
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.00001  # MUCH smaller than before - avoids destroying pretrained knowledge
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ---- LOAD DATASETS (same as before) ----
train_dataset = GastricDataset(CSV_PATH, split="train", transform=train_transform)
val_dataset = GastricDataset(CSV_PATH, split="val", transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---- CLASS WEIGHTS (same as before) ----
full_df = pd.read_csv(CSV_PATH)
train_labels = full_df[full_df["split"] == "train"]["label"].map(
    {"non_cancer": 0, "cancer": 1}
).values

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights (non_cancer, cancer): {class_weights}")

# ---- BUILD MODEL WITH UNFROZEN BACKBONE, LOAD YOUR PREVIOUS WEIGHTS ----
model = build_model(num_classes=2, freeze_backbone=False)  # KEY CHANGE: unfrozen
model.load_state_dict(torch.load(PREVIOUS_MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)

print("Loaded previous model weights. Backbone is now UNFROZEN for fine-tuning.")

# ---- LOSS AND OPTIMIZER (lower learning rate this time) ----
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---- TRAINING LOOP (same structure as before) ----
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    correct, total = 0, 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    avg_val_loss = val_loss / len(val_loader)

    print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} Summary ===")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.2f}%\n")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "gastric_cancer_model_finetuned.pth")
        print(f"New best fine-tuned model saved! (Val Accuracy: {val_acc:.2f}%)\n")

print(f"\nFine-tuning complete. Best validation accuracy: {best_val_acc:.2f}%")
print("Model saved as gastric_cancer_model_finetuned.pth")