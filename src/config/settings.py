"""
Project Configuration
"""

from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset Directories
DATASET_DIR = PROJECT_ROOT / "dataset"
DOWNLOAD_DIR = DATASET_DIR / "downloads"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
SPLITS_DIR = DATASET_DIR / "splits"

# Official INCLUDE Dataset
ZENODO_RECORD_ID = "4010759"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Download Settings
CHUNK_SIZE = 1024 * 1024      # 1 MB
TIMEOUT = 60                  # seconds