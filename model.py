"""
CNN model definition for gastric cancer detection.
Uses a pretrained EfficientNet-B0.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, freeze_backbone=True):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the final classification layer
    # EfficientNet-B0's classifier expects 1280 input features
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    # Quick test: build the model and check it runs on a fake image
    model = build_model()
    print(model.classifier)

    dummy_input = torch.randn(1, 3, 224, 224)  # batch of 1 fake image
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")  # should be [1, 2]
    print("Model built and tested successfully!")