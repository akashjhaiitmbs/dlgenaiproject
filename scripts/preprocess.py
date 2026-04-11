"""Build song index and disk caches for EfficientNet mel tensors and AST inputs."""

from __future__ import annotations

import glob
import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from transformers import ASTFeatureExtractor

from . import audio, config


def scan_genre_songs(stems_dir: str | None = None, genres: list[str] | None = None):
    stems_dir = stems_dir or config.STEMS_DIR
    genres = genres or config.GENRES
    genre_songs: dict[str, list[str]] = {g: [] for g in genres}
    for genre in genres:
        genre_path = os.path.join(stems_dir, genre)
        for song in sorted(os.listdir(genre_path)):
            sp = os.path.join(genre_path, song)
            if os.path.isdir(sp):
                genre_songs[genre].append(sp)
    return genre_songs


def build_efficientnet_mel_cache(
    genre_songs: dict[str, list[str]],
    label_encoder: LabelEncoder,
    esc_wav_glob: str | None = None,
    out_dir: str | None = None,
) -> pd.DataFrame:
    """Write .npy mel tensors and return a DataFrame with path / label / genre."""
    esc_glob = esc_wav_glob or os.path.join(config.ESC_DIR, "*.wav")
    out_dir = out_dir or config.EFF_CACHE
    os.makedirs(out_dir, exist_ok=True)
    esc_files = glob.glob(esc_glob)
    rows: list[dict] = []

    for genre in config.GENRES:
        label_enc = int(label_encoder.transform([genre])[0])
        for song_idx, sp in enumerate(tqdm(genre_songs[genre], desc=genre)):
            try:
                y_full = audio.mix_stems(sp, sr=config.SR)
                clean_segs = audio.get_segments(y_full, sr=config.SR)
                noisy = audio.add_noise(
                    y_full.copy(),
                    esc_files,
                    sr=config.SR,
                    snr_db=random.uniform(5, 12),
                )
                noisy_segs = audio.get_segments(noisy, sr=config.SR)
                for seg_idx, seg in enumerate(clean_segs + noisy_segs):
                    mel3 = audio.to_3ch_mel(seg, sr=config.SR)
                    cache_path = os.path.join(out_dir, f"{genre}_{song_idx}_s{seg_idx}.npy")
                    np.save(cache_path, mel3)
                    rows.append({"path": cache_path, "label": label_enc, "genre": genre})
            except Exception:
                pass

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "index.csv"), index=False)
    return df


def build_ast_feature_cache(
    genre_songs: dict[str, list[str]],
    label_encoder: LabelEncoder,
    extractor: ASTFeatureExtractor,
    esc_wav_glob: str | None = None,
    out_dir: str | None = None,
) -> pd.DataFrame:
    esc_glob = esc_wav_glob or os.path.join(config.ESC_DIR, "*.wav")
    out_dir = out_dir or config.AST_CACHE
    os.makedirs(out_dir, exist_ok=True)
    esc_files = glob.glob(esc_glob)
    rows: list[dict] = []

    for genre in config.GENRES:
        label_enc = int(label_encoder.transform([genre])[0])
        for song_idx, sp in enumerate(tqdm(genre_songs[genre], desc=genre)):
            try:
                y = audio.mix_stems(sp, sr=config.SR_AST)
                for win_idx in range(5):
                    seg_samples = config.SR_AST * 10
                    start = random.randint(0, max(0, len(y) - seg_samples))
                    seg = y[start : start + seg_samples].copy()
                    if win_idx >= 3:
                        seg = audio.add_noise(
                            seg,
                            esc_files,
                            sr=config.SR_AST,
                            snr_db=random.uniform(5, 12),
                        )
                    inputs = extractor(
                        seg,
                        sampling_rate=config.SR_AST,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=1024,
                    )
                    feat = inputs["input_values"].squeeze(0).numpy()
                    cache_path = os.path.join(out_dir, f"{genre}_{song_idx}_w{win_idx}.npy")
                    np.save(cache_path, feat)
                    rows.append({"path": cache_path, "label": label_enc, "genre": genre})
            except Exception:
                pass

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "index.csv"), index=False)
    return df
