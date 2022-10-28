import os

from pathlib import Path
from dotenv import load_dotenv
from minio import Minio

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

QDRANT_PRELOAD_APP_NAME = os.getenv("QDRANT_PRELOAD_APP_NAME")
QDRANT_VOLUME_DIR = os.getenv("QDRANT_VOLUME_DIR")
DATA_DIR = "data"
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
MINIO_MAIN_PATH = os.getenv("MINIO_MAIN_PATH")
MINIO_DATA_DIR = os.path.join(MINIO_MAIN_PATH, "data")
METRIC_COLLECTION_NAMES = ["dogs", "shoes"]

minio_client = Minio(
    os.getenv("MINIO_HOST"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=True,
)
