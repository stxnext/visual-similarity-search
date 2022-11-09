import os

from pathlib import Path


PRELOAD_APP_NAME = Path(os.getenv("PRELOAD_APP_NAME"))
QDRANT_VOLUME_DIR = os.getenv("QDRANT_VOLUME_DIR")
DATA_DIR = "data"
MINIO_QDRANT_DATABASE_FILENAME = "qdrant_storage.zip"
METRIC_COLLECTION_NAMES = ["dogs", "shoes"]
REQUIRED_VOLUME_FOLDERS = ["aliases", "collections", "collections_meta_wal"]
