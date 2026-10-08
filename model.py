"""
Using EfficientNet-B0.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, freeze_backbone=True):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# Testing on fake image
if __name__ == "__main__":
    model = build_model()
    print(model.classifier)

    dummy_input = torch.randn(1, 3, 224, 224)

    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    
    print("Model built and tested successfully!")