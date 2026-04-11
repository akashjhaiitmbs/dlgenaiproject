"""Paths and hyperparameters shared across preprocessing, training, and inference."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

SR = 22_050
SR_AST = 16_000
DURATION = 30
SEG_LEN = 5
SEG_HOP = 2
N_MELS = 128
HOP = 512
N_MFCC = 40

SEED = 42
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "22f2000701-t12026")

_BASE = os.environ.get(
    "MESSY_MASHUP_ROOT",
    "/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup",
)
_WORKING = os.environ.get("WORKING_ROOT", "/kaggle/working")

STEMS_DIR = os.path.join(_BASE, "genres_stems")
MASHUP_DIR = _BASE
ESC_DIR = os.path.join(_BASE, "ESC-50-master", "audio")
TEST_CSV = os.path.join(_BASE, "test.csv")
SAMPLE_SUB = os.path.join(_BASE, "sample_submission.csv")
EFF_CACHE = os.path.join(_WORKING, "eff_cache")
AST_CACHE = os.path.join(_WORKING, "ast_cache")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
