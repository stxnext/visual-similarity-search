import os
import json
import random

from pathlib import Path, PosixPath
from minio import Minio
from PIL import Image
from typing import Union
from abc import ABC, abstractmethod

from qdrant_client.grpc import ScoredPoint


PROJECT_PATH = Path(__file__).parent.parent


def singleton(cls):
    """Singleton implementation in form of decorator"""
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


class EnvFunctionHandler(ABC):
    """
    Base class for listing shared methods.
    """
    local_data_dir = PROJECT_PATH / "data"
    local_models_dir = local_data_dir / "models"
    local_metric_datasets_dir = local_data_dir / "metric_datasets"

    @abstractmethod
    def get_best_score_imgs(self, results: list[ScoredPoint]) -> list[Image.Image]:
        """
        Handler for returning images with the highest similarity scores from env storage.
        Additionally, filenames are returned as future captions in fornt-end module.
        """
        pass

    @abstractmethod
    def get_random_images_from_collection(self, collection_name: str, k: int) -> tuple[list[str], list[Image.Image]]:
        """
        Pulls a random set of images from a selected collection in env storage.
        Used for image input suggestion in front-end component.
        Additionally, filenames are returned as captions.
        """
        pass

    @abstractmethod
    def get_meta_json(self, model_name: str) -> dict[Union[list[int], str]]:
        """
        Get meta.json dictionary created during model training from env storage.
        """
        pass

    @abstractmethod
    def get_weights_dict(self, model_name: str) -> dict[str]:
        """
        Get dictionary with directories to the embedder and trunk files for env storage.
        """
        pass

    @abstractmethod
    def get_weights_datasets(self, weights: dict[str]):
        """
        Pull embedder and trunk files to the container from cloud storage.
        """
        pass

@singleton
class LocalFunctionHandler(EnvFunctionHandler):
    """
    Managing class for local environment methods.
    """

    def get_best_score_imgs(self, results: list[ScoredPoint]) -> list[Image.Image]:
        """
        Handler for returning images with the highest similarity scores from local storage.
        Additionally, filenames are returned as future captions in fornt-end module.
        """
        object_list = [r.payload["file"] for r in results]
        return [Image.open(obj) for obj in object_list]

    def get_random_images_from_collection(self, collection_name: str, k: int) -> tuple[list[str], list[Image.Image]]:
        """
        Pulls a random set of images from a selected collection in local storage.
        Used for image input suggestion in front-end component.
        Additionally, filenames are returned as captions.
        """
        local_collection_dir = self.local_metric_datasets_dir / collection_name
        captions_local = random.choices(list(local_collection_dir.iterdir()), k=k)
        imgs_local = [
            Image.open(local_collection_dir / caption)
            for caption in captions_local
        ]
        captions_local_str = [str(c.stem) for c in captions_local]  # this result is loaded directly to the application state
        return captions_local_str, imgs_local

    def get_meta_json(self, model_name: str) -> dict[Union[list[int], str]]:
        """
        Get meta.json dictionary created during model training from local storage.
        """
        with open(self.local_models_dir / model_name / "meta.json") as f:
            return json.load(f)

    def get_weights_dict(self, model_name: str) -> dict[str]:
        """
        Get dictionary with directories to the embedder and trunk files for local storage.
        """
        return {
            "trunk_local": self.local_models_dir / model_name / "trunk.pth",
            "embedder_local": self.local_models_dir / model_name / "embedder.pth",
        }

    def get_weights_datasets(self, weights: dict[str]):
        """
        Pull embedder and trunk files to the container from cloud storage. Empty for local.
        """
        pass


@singleton
class CloudFunctionHandler(EnvFunctionHandler):
    """
    Managing class for local environment methods.
    """
    minio_bucket_name = os.getenv("MINIO_BUCKET_NAME")
    minio_main_path = Path(os.getenv("MINIO_MAIN_PATH"))
    minio_data_dir = minio_main_path / "data"
    minio_models_dir = minio_data_dir / "models"
    minio_metric_datasets_dir = minio_data_dir / "metric_datasets"

    @staticmethod
    def _get_minio_client() -> Minio:
        """
        Initializes Minio client based on .env-cloud parameters.
        """
        return Minio(
            endpoint=os.getenv("MINIO_HOST"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=True,
        )

    def get_qdrant_database_file(self, file_path: PosixPath, object_name: PosixPath):
        """
        Pulls zipped Qdrant database snapshot from cloud's object_name to container's file_path.
        """
        self._get_minio_client().fget_object(
            bucket_name=self.minio_bucket_name,
            object_name=str(object_name),
            file_path=str(file_path),
        )

    def get_best_score_imgs(self, results: list[ScoredPoint]) -> list[Image.Image]:
        """
        Handler for returning images with the highest similarity scores from cloud storage.
        Additionally, filenames are returned as future captions in fornt-end module.
        """
        object_list = [self.minio_main_path / r.payload["file"] for r in results]
        return [
            Image.open(
                self._get_minio_client().get_object(
                    bucket_name=self.minio_bucket_name, object_name=str(obj)
                )
            )
            for obj in object_list
        ]

    def get_random_images_from_collection(self, collection_name: str, k: int) -> tuple[list[str], list[Image.Image]]:
        """
        Pulls a random set of images from a selected collection in cloud storage.
        Used for image input suggestion in front-end component.
        Additionally, filenames are returned as captions.
        """
        objects = self._get_minio_client().list_objects(
            bucket_name=self.minio_bucket_name,
            prefix=f"{str(self.minio_metric_datasets_dir / collection_name)}/",  # lists objects in "SOME_PATH/"
        )
        object_list = [obj.object_name for obj in objects]
        object_sample_list = random.choices(object_list, k=k)
        captions_cloud = [obj.split("/")[-1] for obj in object_sample_list]
        imgs_cloud = [
            Image.open(
                self._get_minio_client().get_object(
                    bucket_name=self.minio_bucket_name, object_name=str(obj)
                )
            )
            for obj in object_sample_list
        ]
        return captions_cloud, imgs_cloud

    def get_meta_json(self, model_name: str) -> dict[Union[list[int], str]]:
        """
        Get meta.json dictionary created during model training from cloud storage.
        """
        return json.load(
            self._get_minio_client().get_object(
                bucket_name=self.minio_bucket_name,
                object_name=str(self.minio_models_dir / model_name / "meta.json"),
            )
        )

    def get_weights_dict(self, model_name: str) -> dict[str]:
        """
        Get dictionary with directories to the embedder and trunk files for cloud storage.
        """
        return {
            "trunk_minio": self.minio_models_dir / model_name / "trunk.pth",
            "embedder_minio": self.minio_models_dir / model_name / "embedder.pth",
            "trunk_local": self.local_models_dir / model_name / "trunk.pth",
            "embedder_local": self.local_models_dir / model_name / "embedder.pth",
        }

    def get_weights_datasets(self, weights: dict[str]):
        """
        Pull embedder and trunk files to the container from cloud storage. Empty for local.
        """
        if not os.path.isfile(str(weights["trunk_local"])):
            self._get_minio_client().fget_object(
                bucket_name=self.minio_bucket_name,
                object_name=str(weights["trunk_minio"]),
                file_path=str(weights["trunk_local"]),
            )
        if not os.path.isfile(str(weights["embedder_local"])):
            self._get_minio_client().fget_object(
                bucket_name=self.minio_bucket_name,
                object_name=str(weights["embedder_minio"]),
                file_path=str(weights["embedder_local"]),
            )
