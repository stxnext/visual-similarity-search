import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from minio import Minio

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
METRIC_DATASETS_DIR = DATA_DIR / "metric_datasets"
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
MINIO_MAIN_PATH = os.getenv("MINIO_MAIN_PATH")
MINIO_DATA_DIR = os.path.join(MINIO_MAIN_PATH, "data")
MINIO_MODELS_DIR = os.path.join(MINIO_DATA_DIR, "models")
MINIO_METRIC_DATASETS_DIR = os.path.join(MINIO_DATA_DIR, "metric_datasets")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT"))
)

if os.getenv("TYPE") == "LOCAL":
    minio_client = None
else:
    minio_client = Minio(
        os.getenv("MINIO_HOST"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=True,
    )
