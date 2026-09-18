"""
Training script for gastric cancer detection model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

from dataset import GastricDataset, train_transform, eval_transform
from model import build_model

# CONFIG
CSV_PATH = "dataset_labels.csv"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# LOAD DATASETS
train_dataset = GastricDataset(CSV_PATH, split="train", transform=train_transform)
val_dataset = GastricDataset(CSV_PATH, split="val", transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# HANDLE CLASS IMBALANCE
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

# ---- BUILD MODEL ----
model = build_model(num_classes=2, freeze_backbone=True)
model = model.to(DEVICE)

# ---- LOSS AND OPTIMIZER ----
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---- TRAINING LOOP ----
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # --- Training phase ---
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

    # --- Validation phase ---
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

    # Save the best model (based on validation accuracy)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "gastric_cancer_model.pth")
        print(f"New best model saved! (Val Accuracy: {val_acc:.2f}%)\n")

print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")
print("Model saved as gastric_cancer_model.pth")