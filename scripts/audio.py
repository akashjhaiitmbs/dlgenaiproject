"""Load stems, build spectrograms, handcrafted features (Milestone 2 / CNN front-end)."""

from __future__ import annotations

import os
import random

import librosa
import numpy as np

from . import config


def load_audio(fp: str, sr: int | None = None, duration: int | None = None) -> np.ndarray:
    sr = sr or config.SR
    duration = duration or config.DURATION
    y, _ = librosa.load(fp, sr=sr, duration=duration, mono=True)
    target = int(sr * duration)
    if len(y) < target:
        y = np.tile(y, int(np.ceil(target / len(y))))[:target]
    return y[:target].astype(np.float32)


def mix_stems(song_path: str, sr: int | None = None) -> np.ndarray:
    sr = sr or config.SR
    mixed = np.zeros(sr * config.DURATION, dtype=np.float32)
    for stem in ["drums.wav", "vocals.wav", "bass.wav", "others.wav"]:
        fp = os.path.join(song_path, stem)
        if os.path.exists(fp):
            try:
                mixed += load_audio(fp, sr=sr, duration=config.DURATION)
            except OSError:
                pass
    peak = np.abs(mixed).max()
    if peak > 0:
        mixed /= peak
    return mixed


def add_noise(
    y: np.ndarray,
    noise_files: list[str],
    sr: int | None = None,
    snr_db: float | None = None,
) -> np.ndarray:
    sr = sr or config.SR
    if not noise_files:
        return y
    if snr_db is None:
        snr_db = random.uniform(5, 15)
    try:
        noise = load_audio(random.choice(noise_files), sr=sr, duration=config.DURATION)
        signal_rms = np.sqrt(np.mean(y**2)) + 1e-9
        noise_rms = np.sqrt(np.mean(noise**2)) + 1e-9
        noise = noise * (signal_rms / ((10 ** (snr_db / 20)) * noise_rms))
        start = random.randint(0, max(0, len(y) - len(noise)))
        chunk_len = min(len(noise), len(y) - start)
        y = y.copy()
        y[start : start + chunk_len] += noise[:chunk_len]
    except OSError:
        pass
    return y


def get_segments(y: np.ndarray, sr: int | None = None) -> list[np.ndarray]:
    sr = sr or config.SR
    seg_samples = sr * config.SEG_LEN
    hop_samples = sr * config.SEG_HOP
    segs: list[np.ndarray] = []
    for start in range(0, len(y) - seg_samples + 1, hop_samples):
        segs.append(y[start : start + seg_samples].copy())
    if not segs:
        pad = np.zeros(seg_samples, dtype=np.float32)
        pad[: min(len(y), seg_samples)] = y[:seg_samples]
        segs.append(pad)
    return segs


def to_3ch_mel(y: np.ndarray, sr: int | None = None) -> np.ndarray:
    sr = sr or config.SR
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS, hop_length=config.HOP
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    d1 = librosa.feature.delta(mel_db)
    d2 = librosa.feature.delta(mel_db, order=2)
    out = np.stack([mel_db, d1, d2], axis=0)
    for i in range(3):
        out[i] = (out[i] - out[i].mean()) / (out[i].std() + 1e-9)
    return out.astype(np.float32)


def spec_augment(
    mel: np.ndarray, freq_mask: int = 20, time_mask: int = 50, n_masks: int = 2
) -> np.ndarray:
    mel = mel.copy()
    _, f, t = mel.shape
    for _ in range(n_masks):
        f_sz = random.randint(1, max(1, min(freq_mask, f - 1)))
        f0 = random.randint(0, max(0, f - f_sz - 1))
        mel[:, f0 : f0 + f_sz, :] = 0
        t_sz = random.randint(1, max(1, min(time_mask, t - 1)))
        t0 = random.randint(0, max(0, t - t_sz - 1))
        mel[:, :, t0 : t0 + t_sz] = 0
    return mel


def extract_handcrafted(y: np.ndarray) -> np.ndarray:
    sr = config.SR
    n_mfcc = config.N_MFCC
    feats: list[np.ndarray] = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    feats.extend(
        [
            mfcc.mean(1),
            mfcc.std(1),
            chroma.mean(1),
            chroma.std(1),
            contrast.mean(1),
            contrast.std(1),
            zcr.mean(1),
            rms.mean(1),
        ]
    )
    try:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        feats.extend([tonnetz.mean(1), tonnetz.std(1)])
    except Exception:
        feats.extend([np.zeros(6), np.zeros(6)])
    try:
        tempo = librosa.feature.tempo(y=y, sr=sr)
        feats.append([float(tempo[0])])
    except Exception:
        feats.append([0.0])
    return np.concatenate(feats)
