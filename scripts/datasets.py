"""PyTorch datasets for cached mel patches and AST feature tensors."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import spec_augment


class MelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["path"])
        if self.augment:
            mel = spec_augment(mel)
            if random.random() < 0.4:
                mel = mel * random.uniform(0.85, 1.15)
        return torch.tensor(mel, dtype=torch.float32), int(row["label"])


class ASTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = np.load(row["path"])
        if self.augment:
            feat = feat.copy()
            t = random.randint(1, 80)
            t0 = random.randint(0, max(0, feat.shape[0] - t - 1))
            feat[t0 : t0 + t, :] = 0
            f = random.randint(1, 20)
            f0 = random.randint(0, max(0, feat.shape[1] - f - 1))
            feat[:, f0 : f0 + f] = 0
        return torch.tensor(feat, dtype=torch.float32), int(row["label"])
