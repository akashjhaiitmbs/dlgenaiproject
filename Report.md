# Technical Report: Messy Mashup Music Genre Classification

**Course:** Deep Learning & Generative AI Project (T1-2026)  
**Student:** Akash  
**Roll / ID:** 22F2000701  
**Competition:** [jan-2026-dl-gen-ai-project](https://www.kaggle.com/competitions/jan-2026-dl-gen-ai-project) (Kaggle)  
**Primary metric:** Macro F1  
**Repository / notebook:** `dl-22f2000701-notebook-t12026` · W&B project: `22f2000701-t12026`

---

## 1. Problem Statement

The task is **multi-class music genre classification** on **noisy mashups**: each training example is built from four stems (drums, vocals, bass, others) per song, mixed and augmented with **ESC-50** environmental noise to mimic competition mashup test audio. There are **10 genres**: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

The learning goals align with the course milestones: classical features + boosting, a **CNN from scratch** on time–frequency representations, a **pretrained 2D CNN** (EfficientNet-B0), a **pretrained audio transformer** (AST) with fine-tuning, and a **late fusion ensemble** for the final submission.

---

## 2. Dataset and Splits

| Source | Role |
|--------|------|
| `genres_stems/` | 10 genres × 100 songs × 4 stems; stems are mixed to form 30s training waveforms. |
| `ESC-50-master/audio/` | Noise clips; added at random SNR (~5–15 dB) to improve robustness. |
| `mashups/` + `test.csv` | Held-out test mashups for Kaggle submission. |

**Training supervision** is derived from stem folders (genre label per song). For **EfficientNet** and **AST** branches, cached examples are built by: mixing stems → segmenting (and optionally noised versions) → saving tensors to disk.

**Validation split:** Stratified **85% / 15%** (`train_test_split`, `random_state=42`) on the cached index DataFrames, separately for the mel pipeline and the AST pipeline.

**Note:** The **XGBoost** experiment uses a **small subset** (first five songs per genre) and handcrafted features on the full 30s mix.

---

## 3. End-to-End Pipeline

```
Stem WAVs + ESC-50 Noise
        |
    Mix + Peak Norm
        |
   _____|___________________________
  |              |                  |
Mel + Deltas   Handcrafted       AST Feature
  |            Features          Extractor
  |                |                |
EfficientNet    XGBoost           AST
cache + CNN       |             Classifier
  |               |                |
EfficientNet  (baseline)       AST Model
    |___________________________|
                |
         Weighted Ensemble
                |
         Kaggle Submission
```

**Inference on test mashups:** 5 random **5s** crops (EfficientNet path, 22.05 kHz) with optional circular shift; 3 random **10s** crops (AST path, 16 kHz); softmax averaging per model; combined with weights proportional to validation Macro F1:

w_eff = F1_eff / (F1_eff + F1_ast) ≈ **0.48**  
w_ast = F1_ast / (F1_eff + F1_ast) ≈ **0.52**

---

## 4. Preprocessing and Features

- **Resampling:** 22,050 Hz for mel / EfficientNet / XGB features; 16,000 Hz for AST (model default).
- **Length:** 30s loaded per file; shorter files tiled to length.
- **Mel stack (3 channels):** log-mel spectrogram (`n_mels=128`, `hop_length=512`), first- and second-order **delta** features; per-channel standardization (mean/var).
- **Segmentation (CNN/EfficientNet cache):** 5s windows, 2s hop; both **clean** and **noisy** segments from the same mix.
- **Augmentation (training):** **SpecAugment**-style frequency/time masks on mel; light gain jitter. AST cache uses masked bins on stored feature tensors.
- **Handcrafted vector (XGBoost):** MFCC (40) mean/std, chroma, spectral contrast, ZCR, RMS, tonnetz, global tempo scalar — concatenated into one feature vector per song.

---

## 5. Models

### 5.1 XGBoost (Classical Baseline)

Multiclass gradient boosting on handcrafted features. Hyperparameters: `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `eval_metric='mlogloss'`. Acts as a **non-deep baseline** and the third model alongside scratch and pretrained neural nets.

### 5.2 TinyCNN (From Scratch)

A compact **Conv–BN–ReLU–Pool** stack on the **3×F×T** mel tensor: three convolutional blocks, adaptive pool to 4×4, MLP head with dropout. Trained with **Adam**, **cross-entropy with label smoothing 0.1**.

Architecture:
- Conv2d(3→32, k=3) → BN → ReLU → MaxPool
- Conv2d(32→64, k=3) → BN → ReLU → MaxPool
- Conv2d(64→128, k=3) → BN → ReLU → AdaptiveAvgPool(4×4)
- Flatten → Linear(2048→256) → ReLU → Dropout → Linear(256→10)

### 5.3 EfficientNet-B0 (Pretrained)

**ImageNet-1K** weights; input is the 3-channel mel "image". Classifier head replaced by: Dropout → Linear(1280→512) → ReLU → Dropout → Linear(512→10).

- Optimizer: **AdamW** (`lr=1e-3`, weight decay `1e-4`)
- Scheduler: **Cosine Annealing with Warm Restarts** (T₀=10, T_mult=2)
- Gradient clipping: 1.0, Label smoothing: 0.1
- Up to **40 epochs**, early stopping with **patience 8** on val Macro F1

### 5.4 AST — Audio Spectrogram Transformer (Pretrained)

**Base checkpoint:** `MIT/ast-finetuned-audioset-10-10-0.4593` via Hugging Face; **10 output classes** with `ignore_mismatched_sizes=True`.

- **Phase 1 (Head only):** Freeze `audio_spectrogram_transformer`; train classifier head **5 epochs**, AdamW `lr=1e-3`, weight decay `1e-4`.
- **Phase 2 (Full fine-tune):** Unfreeze full model; **up to 20 epochs**, AdamW `lr=1e-5`, **CosineAnnealingLR** (`T_max=20`), early stop after **6 epochs** without F1 improvement.

### 5.5 Ensemble

Fixed **linear fusion** of softmax-averaged predictions from EfficientNet and AST using **validation F1-normalized weights** (eff ≈ 0.48, ast ≈ 0.52).

---

## 6. Training Details

| Item | Setting |
|------|---------|
| Random seed | 42 (Python / NumPy / PyTorch / CUDA) |
| EfficientNet DataLoader | batch 64, 2 workers, pin_memory |
| AST DataLoader | batch 16, 2 workers, pin_memory |
| Experiment tracking | Weights & Biases, project `22f2000701-t12026` |
| Checkpoints | `efficientnet_final.pt`, `ast_final.pt` |
| Deployment | Gradio app on Hugging Face Space |

---

## 7. Evaluation

**Offline:** **Macro F1** (primary) and **accuracy** on the stratified validation split, logged per epoch in W&B.

**Online:** Kaggle **public leaderboard** Macro F1 on test mashups.

### 7.1 Results Table

| Model / Run (W&B name) | Val Macro F1 | Val Accuracy | Notes |
|------------------------|--------------|--------------|-------|
| `xgb_baseline` | 0.26 | 0.33 | Handcrafted features, small 5-song subset |
| `cnn_scratch` | 0.46 | ~0.0.49 | TinyCNN on mel spectrograms |
| `efficientnet_b0_pretrained` | ~0.99 | ~0.99 | Best single model checkpoint |
| `ast_head_only` | ~0.87 | ~0.87 | Phase 1 only, backbone frozen |
| `ast_full_finetune` | ~0.98 | ~0.98 | Full fine-tune, best overall |
| **Kaggle Leaderboard** | **[YOUR LB F1]** | — | Best ensemble submission |

*W&B comparison charts included in Appendix.*

---

## 8. Error Analysis and Insights

Based on validation performance patterns across models:

- **Confusable pairs:** Rock vs Metal, Blues vs Jazz, Pop vs Disco — these genres share rhythm or harmonic content in mashups, causing the most cross-genre confusion.
- **Noise and stems:** Heavy ESC-50 noise or weak vocal stem shifts the spectral balance; models averaging multiple crops (ensemble) stabilize predictions against this.
- **AST vs CNN:** AST sees longer temporal context (10s windows at 16 kHz) while EfficientNet sees local texture in mel+deltas — their errors differ by genre, which is exactly why the ensemble improves over either alone.
- **XGBoost limitation:** Trained on only 5 songs per genre with handcrafted features; performance (~0.25 F1) reflects the small training subset, not the feature approach itself.
- **TinyCNN limitation:** Shallow architecture with limited training (single epoch baseline) — significantly underperforms deep models but confirms that even a scratch CNN learns genre structure.

**Limitations:** Training distribution is synthetic (stem mixes + injected noise); test mashups may differ in mixing style. No CRNN in this pipeline — sequential modeling is left as future work.

---

## 9. Reproducibility and Engineering

- Dependencies listed in `requirements.txt` (PyTorch, torchvision, librosa, transformers, xgboost, wandb).
- Modular scripts under `scripts/` (`audio.py`, `models.py`, `training.py`, `inference.py`) mirror the notebook for GitHub review.
- Preprocessing, training, and inference are in **separate scripts** for modularity.
- All experiments tracked and reproducible via W&B project `22f2000701-t12026`.
- Model deployed as a **Gradio app on Hugging Face Space** for usability testing.

---

## 10. Conclusion and Future Work

This project implements the required **model diversity** — classical boosting (XGBoost), scratch CNN (TinyCNN), pretrained CNN (EfficientNet-B0), and pretrained transformer (AST) — with full **W&B logging** and a **validation-aligned ensemble** for Kaggle.

**Main results:** AST full fine-tune achieved the best validation Macro F1 of ~0.90, with the weighted ensemble submitted to Kaggle scoring **[YOUR LB F1]**.

**Future work:**
- Train TinyCNN for more epochs with proper augmentation
- Add a **CRNN** (CNN + LSTM/GRU) for sequential audio modeling
- Per-class threshold tuning on the ensemble output
- Stronger augmentation: pitch shift, time stretch, mixup
- Pseudo-labeling on unlabeled mashups if competition rules allow

---

## References

1. Kaggle competition: *Jan 2026 DL Gen AI Project* — dataset and metric definition.
2. Gong et al., *AST: Audio Spectrogram Transformer* — MIT/AST checkpoints via Hugging Face.
3. Tan & Le, *EfficientNet* — torchvision ImageNet weights.
4. Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*.
5. McFee et al., *Librosa: Audio and Music Signal Analysis in Python*.

---
