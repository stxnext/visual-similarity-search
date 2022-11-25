import os
from pathlib import Path

QDRANT_VOLUME_DIR = Path(os.getenv("QDRANT_VOLUME_DIR"))
DATA_DIR = "data"
MINIO_QDRANT_DATABASE_FILENAME = "qdrant_storage.zip"
REQUIRED_VOLUME_FOLDERS = ["aliases", "collections", "collections_meta_wal"]
