"""
Plots training/validation loss and accuracy curves from your completed
10-epoch training run.
"""

import matplotlib.pyplot as plt

# ---- Actual results from training ----
train_losses = [0.5451, 0.4607, 0.4374, 0.4269, 0.4231, 0.4231, 0.4195, 0.4180, 0.4068, 0.4083]
val_losses   = [0.4777, 0.4406, 0.4283, 0.4185, 0.4084, 0.4024, 0.4048, 0.3921, 0.3989, 0.3882]
val_accuracies = [77.47, 79.57, 82.31, 82.10, 79.80, 79.50, 82.98, 82.38, 83.73, 80.60]

epochs = list(range(1, len(train_losses) + 1))

# ---- Plot loss curve ----
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker='o', label='Train Loss')
plt.plot(epochs, val_losses, marker='o', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)

# ---- Plot accuracy curve ----
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies, marker='o', color='green', label='Validation Accuracy')
plt.axhline(y=87.5, color='red', linestyle='--', label='Baseline (always predict non-cancer)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

print("Graph saved as training_curves.png")
print(f"\nBest validation accuracy: {max(val_accuracies):.2f}% (Epoch {val_accuracies.index(max(val_accuracies)) + 1})")