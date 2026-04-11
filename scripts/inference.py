"""Ensemble inference on test mashups (EfficientNet + AST), same logic as the Kaggle notebook."""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from . import audio, config


class GenreEnsemble:
    """Loads no weights here — pass already-loaded ``eff_model`` and ``ast_model``."""

    def __init__(
        self,
        eff_model: torch.nn.Module,
        ast_model: torch.nn.Module,
        ast_extractor,
        device: torch.device,
        weight_eff: float,
        weight_ast: float,
        mashup_dir: str | None = None,
    ):
        self.eff_model = eff_model
        self.ast_model = ast_model
        self.ast_extractor = ast_extractor
        self.device = device
        self.w_eff = weight_eff
        self.w_ast = weight_ast
        self.mashup_dir = mashup_dir or config.MASHUP_DIR

    def predict_file_probs(self, fp: str) -> np.ndarray:
        p_eff = self._eff_probs(fp)
        p_ast = self._ast_probs(fp)
        return self.w_eff * p_eff + self.w_ast * p_ast

    def _eff_probs(self, fp: str) -> np.ndarray:
        self.eff_model.eval()
        probs = np.zeros(10)
        y = audio.load_audio(fp, sr=config.SR, duration=config.DURATION)
        seg_samples = config.SR * config.SEG_LEN
        for _ in range(5):
            start = random.randint(0, max(0, len(y) - seg_samples))
            seg = y[start : start + seg_samples]
            if random.random() < 0.5:
                seg = np.roll(seg, random.randint(0, config.SR * 2))
            mel = audio.to_3ch_mel(seg, sr=config.SR)
            x = torch.tensor(mel[np.newaxis]).to(self.device)
            with torch.no_grad():
                probs += F.softmax(self.eff_model(x), dim=1).cpu().numpy()[0]
        return probs / 5

    def _ast_probs(self, fp: str) -> np.ndarray:
        self.ast_model.eval()
        probs = np.zeros(10)
        try:
            y = audio.load_audio(fp, sr=config.SR_AST, duration=config.DURATION)
            seg_samples = config.SR_AST * 10
            for _ in range(3):
                start = random.randint(0, max(0, len(y) - seg_samples))
                seg = y[start : start + seg_samples]
                inputs = self.ast_extractor(
                    seg,
                    sampling_rate=config.SR_AST,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=1024,
                )
                x = inputs["input_values"].to(self.device)
                with torch.no_grad():
                    probs += F.softmax(self.ast_model(input_values=x).logits, dim=1).cpu().numpy()[0]
            return probs / 3
        except Exception:
            return probs

    def predict_test_df(
        self,
        test_df: pd.DataFrame,
        filename_col: str,
    ) -> list[int]:
        preds: list[int] = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            fp = os.path.join(self.mashup_dir, row[filename_col])
            try:
                combined = self.predict_file_probs(fp)
                preds.append(int(np.argmax(combined)))
            except Exception as e:
                print(f"Error: {fp} | {e}")
                preds.append(0)
        return preds


def write_submission(
    sample_sub_path: str,
    genre_labels: list[str],
    out_path: str,
) -> pd.DataFrame:
    sub = pd.read_csv(sample_sub_path)
    sub["genre"] = genre_labels
    sub.to_csv(out_path, index=False)
    return sub
