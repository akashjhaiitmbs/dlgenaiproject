Messy Mashup — Music Genre Classification

Problem Statement
Classify the music genre of noisy audio mashups constructed from mixed instrument stems and environmental noise. The dataset contains 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.
Dataset

genres_stems/ — 10 genres × 100 songs × 4 stems (drums, vocals, bass, others)
mashups/ — 3020 noisy test audio files
ESC-50-master/ — environmental noise dataset used during mashup creation

Dataset hosted on Kaggle: jan-2026-dl-gen-ai-project

Folder Structure
├── notebooks/          # Kaggle inference notebooks
├── scripts/            # Preprocessing and training scripts
├── data/               # Data lives on Kaggle (see link above)
└── requirements.txt    # Python dependencies
Experiment Tracking
All runs tracked on Weights & Biases:
Project: genre-classification
How to Run

Clone this repo
Install dependencies: pip install -r requirements.txt
Upload notebook to Kaggle and attach the competition dataset
Run all cells