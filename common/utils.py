from pathlib import Path
from typing import Any

from common.consts import MINIO_MAIN_PATH, PROJECT_PATH
from metrics.consts import MetricCollections


def singleton(cls) -> Any:
    """Singleton implementation in form of decorator"""
    instances = {}

    def getinstance() -> Any:
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


class WeightsPathGenerator:
    """Class used for generating weight/model file paths based on the collection's name."""

    def __init__(self, collection_name: MetricCollections):
        self.collection_name_value = collection_name.value

    def create_weights_path(self, prefix: Path, suffix: str) -> Path:
        """
        Create collection/model specific paths to files containing weights and models.
        """
        return prefix / "data" / "models" / self.collection_name_value / suffix

    @property
    def trunk_minio(self):
        return self.create_weights_path(prefix=MINIO_MAIN_PATH, suffix="trunk.pth")

    @property
    def embedder_minio(self):
        return self.create_weights_path(prefix=MINIO_MAIN_PATH, suffix="embedder.pth")

    @property
    def trunk_local(self):
        return self.create_weights_path(prefix=PROJECT_PATH, suffix="trunk.pth")

    @property
    def embedder_local(self):
        return self.create_weights_path(prefix=PROJECT_PATH, suffix="embedder.pth")
