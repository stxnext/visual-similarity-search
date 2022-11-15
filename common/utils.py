from typing import Any
from dataclasses import dataclass
from pathlib import Path

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


def create_weights_path(prefix: Path, collection_name: MetricCollections, suffix: str) -> Path:
    return prefix / "data" / "models" / collection_name.value / suffix


@dataclass
class WeightsData:
    """Dto containing all dataset information"""
    collection_name: MetricCollections
    trunk_minio: Path = MINIO_MAIN_PATH
    embedder_minio: Path = MINIO_MAIN_PATH
    trunk_local: Path = PROJECT_PATH
    embedder_local: Path = PROJECT_PATH

    def __post_init__(self) -> None:
        self.trunk_minio = create_weights_path(
            prefix=self.trunk_minio,
            collection_name=self.collection_name,
            suffix="trunk.pth"
        )
        self.embedder_minio = create_weights_path(
            prefix=self.embedder_minio,
            collection_name=self.collection_name,
            suffix="embedder.pth"
        )
        self.trunk_local = create_weights_path(
            prefix=self.trunk_local,
            collection_name=self.collection_name,
            suffix="trunk.pth"
        )
        self.embedder_local = create_weights_path(
            prefix=self.embedder_local,
            collection_name=self.collection_name,
            suffix="embedder.pth"
        )
