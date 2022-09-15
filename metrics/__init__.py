import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
METRIC_TYPES_DIR = DATA_DIR / "metric_datasets"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333))
)
