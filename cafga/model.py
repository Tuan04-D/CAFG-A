"""EEGNet-8,2 (Lawhern et al., 2018), including its max-norm constraints."""

import torch
from torch import nn


class _Constrained(nn.Module):
    def __init__(self, module, max_norm, dim):
        super().__init__()
        self.module = module
        self.max_norm = max_norm
        self.dim = dim

    def forward(self, x):
        with torch.no_grad():
            w = self.module.weight
            norm = w.norm(2, dim=self.dim, keepdim=True).clamp_min(1e-8)
            w.mul_(norm.clamp_max(self.max_norm) / norm)
        return self.module(x)


class EEGNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_samples, f1=8, depth=2,
                 kernel_length=128, dropout=0.5):
        super().__init__()
        f2 = f1 * depth
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.depthwise = _Constrained(
            nn.Conv2d(f1, f2, (n_channels, 1), groups=f1, bias=False),
            max_norm=1.0, dim=(1, 2, 3),
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(f2, f2, (1, 16), padding=(0, 8), groups=f2, bias=False),
            nn.Conv2d(f2, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.flatten = nn.Flatten()
        self.n_features = f2 * (((n_samples + 1) // 4) // 8)
        self.classifier = _Constrained(
            nn.Linear(self.n_features, n_classes), max_norm=0.25, dim=1)

    def features(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.flatten(x)

    def forward(self, x):
        return self.classifier(self.features(x))
