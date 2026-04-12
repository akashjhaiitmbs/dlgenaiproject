

from __future__ import annotations

import json
import random
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ROOT = Path(__file__).resolve().parent
CONFIG_NAME = "ensemble_config.json"
N_MELS = 128
HOP_LENGTH = 512

_predictor = None


class EfficientNetGenre(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.base = models.efficientnet_b0(weights=None)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.base(x)


def load_audio(fp: str, sr: int, duration_sec: int) -> np.ndarray:
    y, _ = librosa.load(fp, sr=sr, duration=duration_sec, mono=True)
    target = int(sr * duration_sec)
    if len(y) < target:
        y = np.tile(y, int(np.ceil(target / len(y))))[:target]
    return y[:target].astype(np.float32)


def to_3ch_mel(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    d1 = librosa.feature.delta(mel_db)
    d2 = librosa.feature.delta(mel_db, order=2)
    out = np.stack([mel_db, d1, d2], axis=0)
    for i in range(3):
        out[i] = (out[i] - out[i].mean()) / (out[i].std() + 1e-9)
    return out.astype(np.float32)


class EnsemblePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg: dict = {}
        self.eff_model: EfficientNetGenre | None = None
        self.ast_model: ASTForAudioClassification | None = None
        self.ast_extractor = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        cfg_path = ROOT / CONFIG_NAME
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"Missing {CONFIG_NAME} in {ROOT}. Upload ensemble_config.json from Kaggle."
            )
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = json.load(f)

        genres = self.cfg["genres"]
        n_cls = len(genres)
        eff_name = self.cfg.get("efficientnet_ckpt", "efficientnet_final.pt")
        ast_name = self.cfg.get("ast_ckpt", "ast_final.pt")
        eff_pt = ROOT / eff_name
        ast_pt = ROOT / ast_name
        if not eff_pt.is_file():
            raise FileNotFoundError(f"Missing {eff_pt}")
        if not ast_pt.is_file():
            raise FileNotFoundError(f"Missing {ast_pt}")

        ast_id = self.cfg["ast_model_id"]

        self.eff_model = EfficientNetGenre(num_classes=n_cls).to(self.device)
        eff_sd = torch.load(eff_pt, map_location=self.device)
        self.eff_model.load_state_dict(eff_sd, strict=True)
        self.eff_model.eval()

        self.ast_extractor = ASTFeatureExtractor.from_pretrained(ast_id)
        self.ast_model = ASTForAudioClassification.from_pretrained(
            ast_id,
            num_labels=n_cls,
            ignore_mismatched_sizes=True,
        ).to(self.device)
        ast_sd = torch.load(ast_pt, map_location=self.device)
        self.ast_model.load_state_dict(ast_sd, strict=True)
        self.ast_model.eval()

        self._loaded = True

    def predict_path(self, audio_path: str | None) -> tuple[str, dict]:
        if not audio_path:
            return "Upload a WAV file.", {}
        self.load()

        random.seed(42)
        sr = int(self.cfg["sr"])
        sr_ast = int(self.cfg["sr_ast"])
        dur = int(self.cfg["duration_sec"])
        seg_eff = int(self.cfg["eff_seg_len_sec"])
        n_eff = int(self.cfg.get("eff_n_random_segments", 5))
        win_ast = int(self.cfg["ast_window_sec"])
        n_ast = int(self.cfg.get("ast_n_random_segments", 3))
        w_eff = float(self.cfg["w_eff"])
        w_ast = float(self.cfg["w_ast"])
        genres = self.cfg["genres"]

        seg_samples_eff = sr * seg_eff
        seg_samples_ast = sr_ast * win_ast

        self.eff_model.eval()
        probs_eff = np.zeros(len(genres), dtype=np.float64)
        y_eff = load_audio(audio_path, sr, dur)
        for _ in range(n_eff):
            start = random.randint(0, max(0, len(y_eff) - seg_samples_eff))
            seg = y_eff[start : start + seg_samples_eff]
            if random.random() < 0.5:
                seg = np.roll(seg, random.randint(0, sr * 2))
            mel = to_3ch_mel(seg, sr)
            x = torch.tensor(mel[np.newaxis]).to(self.device)
            with torch.no_grad():
                probs_eff += F.softmax(self.eff_model(x), dim=1).cpu().numpy()[0]
        probs_eff /= n_eff

        self.ast_model.eval()
        probs_ast = np.zeros(len(genres), dtype=np.float64)
        try:
            y_ast = load_audio(audio_path, sr_ast, dur)
            for _ in range(n_ast):
                start = random.randint(0, max(0, len(y_ast) - seg_samples_ast))
                seg = y_ast[start : start + seg_samples_ast]
                inputs = self.ast_extractor(
                    seg,
                    sampling_rate=sr_ast,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=1024,
                )
                x = inputs["input_values"].to(self.device)
                with torch.no_grad():
                    probs_ast += F.softmax(self.ast_model(input_values=x).logits, dim=1).cpu().numpy()[0]
            probs_ast /= n_ast
        except Exception:
            pass

        combined = w_eff * probs_eff + w_ast * probs_ast
        idx = int(np.argmax(combined))
        label = genres[idx]
        out = {g: float(combined[i]) for i, g in enumerate(genres)}
        return label, out


def get_predictor() -> EnsemblePredictor:
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
    return _predictor


def predict(audio_path: str | None) -> tuple[str, dict]:
    try:
        p = get_predictor()
        label, probs = p.predict_path(audio_path)
        if not probs:
            return label, {}
        return label, probs
    except FileNotFoundError as e:
        return f"Setup error: {e}", {}
    except Exception as e:
        return f"Error: {e}", {}


with gr.Blocks(title="Messy Mashup — Genre") as demo:
    gr.Markdown(
        "### Music genre (10-class)\n"
        "Upload a **WAV** clip (up to ~30s; longer files are trimmed). "
        "Uses EfficientNet + AST ensemble from training checkpoints."
    )
    inp = gr.Audio(type="filepath", label="Audio (WAV)", sources=["upload"])
    out_genre = gr.Textbox(label="Predicted genre")
    out_probs = gr.JSON(label="Class probabilities (ensemble)")

    def _wrap(path):
        label, probs = predict(path)
        return label, probs

    inp.change(fn=_wrap, inputs=inp, outputs=[out_genre, out_probs])

if __name__ == "__main__":
    demo.launch()
