"""Neural architectures used in the Kaggle notebook (scratch CNN + pretrained EfficientNet-B0)."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


class TinyCNN(nn.Module):
    """Small CNN on 3-channel mel + deltas (from-scratch baseline)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class EfficientNetGenre(nn.Module):
    """EfficientNet-B0 trunk with ImageNet weights and a custom classifier head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.base = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.base(x)
