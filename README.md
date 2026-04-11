# Messy Mashup — Music Genre Classification

**Course:** DL & GenAI (T1-2026)  
**Student:** Akash · **ID:** 22F2000701

## Problem

Multi-class genre classification for noisy mashups built from stem mixes and ESC-50 noise. Labels: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

## Dataset

Competition: **jan-2026-dl-gen-ai-project** (Kaggle).

| Path | Contents |
|------|----------|
| `genres_stems/` | 10 genres × 100 songs × 4 stems (drums, vocals, bass, others) |
| `mashups/` | Test audio |
| `ESC-50-master/` | Environmental noise used when building mashups |

## Repository layout

```
├── notebooks/
├── scripts/
├── data/
└── requirements.txt
```

- **notebooks/** — Main pipeline: [`notebooks/dl-22f2000701-notebook-t12026.ipynb`](notebooks/dl-22f2000701-notebook-t12026.ipynb) (Kaggle submission notebook).
- **scripts/** — Python package aligned with preprocessing, models, training, and inference used in the notebook.
- **data/** — Reserved for local files; the full competition bundle is read from Kaggle paths unless overridden (see below).

### `scripts/` modules

| File | Contents |
|------|----------|
| `config.py` | Paths, `GENRES`, signal/mel hyperparameters, `WANDB_PROJECT`, `set_seed()` |
| `audio.py` | I/O, stem mixing, noise, segmentation, mel + deltas, SpecAugment, handcrafted features |
| `preprocess.py` | Song indexing, EfficientNet mel cache, AST feature cache |
| `datasets.py` | `MelDataset`, `ASTDataset` |
| `models.py` | `TinyCNN`, `EfficientNetGenre` |
| `training.py` | `train_model` with W&B and early stopping |
| `wandb_auth.py` | W&B login helper |
| `inference.py` | `GenreEnsemble`, `write_submission` |

Default paths target the Kaggle competition directory. For a local tree under **`data/`**, set:

- `MESSY_MASHUP_ROOT` — root folder containing `genres_stems/`, `mashups/`, etc.
- `WORKING_ROOT` — writable directory for caches and outputs

Imports assume the **repository root** is on `PYTHONPATH` (e.g. run shells and tools from that directory).

## Weights & Biases

- **Project:** `22f2000701-t12026`
- **Kaggle:** add secret `wandb_api_key` with the API key from W&B account settings.

## Local setup

1. `pip install -r requirements.txt`
2. Run the notebook on Kaggle with the competition dataset attached, **or** set `MESSY_MASHUP_ROOT` / `WORKING_ROOT` and execute equivalent steps locally.
