import os
import json
import random

from pathlib import Path
from minio import Minio
from PIL import Image
from typing import List

from loguru import logger


class EnvFunctionHandler:
    """
    Class for managing environment-dependent functions.
    """

    def __init__(self):
        self.env = os.getenv("TYPE")
        self.local_data_dir = Path(__file__).parent.parent / "data"
        self.local_models_dir = self.local_data_dir / "models"
        self.local_metric_datasets_dir = self.local_data_dir / "metric_datasets"
        if self.env != "LOCAL":
            self.minio_client = Minio(
                endpoint=os.getenv("MINIO_HOST"),
                access_key=os.getenv("MINIO_ACCESS_KEY"),
                secret_key=os.getenv("MINIO_SECRET_KEY"),
                secure=True,
            )
            self.minio_bucket_name = os.getenv("MINIO_BUCKET_NAME")
            self.minio_main_path = Path(os.getenv("MINIO_MAIN_PATH"))
            self.minio_data_dir = self.minio_main_path / "data"
            self.minio_models_dir = self.minio_data_dir / "models"
            self.minio_metric_datasets_dir = self.minio_data_dir / "metric_datasets"

    def get_qdrant_database_file(self, file_path: str, object_name: str):
        """
        Provides instance of MinIO client.
        """
        self.minio_client.fget_object(
            bucket_name=self.minio_bucket_name,
            object_name=str(object_name),
            file_path=str(file_path),
        )

    def get_best_score_imgs(self, results: List) -> List:
        """
        Handler for returning images
        """
        if self.env != "LOCAL":
            object_list = [self.minio_main_path / r.payload["file"] for r in results]
            return [
                Image.open(
                    self.minio_client.get_object(
                        bucket_name=self.minio_bucket_name, object_name=str(obj)
                    )
                )
                for obj in object_list
            ]

        object_list = [r.payload["file"] for r in results]
        return [Image.open(obj) for obj in object_list]

    def get_random_images_from_collection(
        self, collection_name: str, k: int
    ) -> (list[str], list[Image.Image]):
        """
        Depends on environment.
        Pulls a random set of images from a selected collection. Used for search suggestion in front-end component.
        """
        if self.env != "LOCAL":
            objects = self.minio_client.list_objects(
                bucket_name=self.minio_bucket_name,
                prefix=f"{str(self.minio_metric_datasets_dir / collection_name)}/",  # lists objects in "SOME_PATH/"
            )
            object_list = [obj.object_name for obj in objects]
            logger.info(f"{object_list[:3]=}")
            object_sample_list = random.choices(object_list, k=k)
            captions_cloud = [obj.split("/")[-1] for obj in object_sample_list]
            imgs_cloud = [
                Image.open(
                    self.minio_client.get_object(
                        bucket_name=self.minio_bucket_name, object_name=str(obj)
                    )
                )
                for obj in object_sample_list
            ]
            return captions_cloud, imgs_cloud

        local_collection_dir = self.local_metric_datasets_dir / collection_name
        captions_local = random.choices(os.listdir(str(local_collection_dir)), k=k)
        imgs_local = [
            Image.open(str(local_collection_dir / caption))
            for caption in captions_local
        ]
        return captions_local, imgs_local

    def get_meta_json(self, model_name: str) -> dict:
        """
        Depends on environment.
        Get meta.json dataset based on configured environment.
        """
        if self.env != "LOCAL":
            return json.load(
                self.minio_client.get_object(
                    bucket_name=self.minio_bucket_name,
                    object_name=str(self.minio_models_dir / model_name / "meta.json"),
                )
            )

        with open(self.local_models_dir / model_name / "meta.json") as f:
            return json.load(f)

    def get_weights_dict(self, model_name: str) -> dict:
        """
        Returns path-dependent weights directories dictionary.
        """
        d = {
            "trunk_local": self.local_models_dir / model_name / "trunk.pth",
            "embedder_local": self.local_models_dir / model_name / "embedder.pth",
        }
        if self.env != "LOCAL":
            d["trunk_minio"] = self.minio_models_dir / model_name / "trunk.pth"
            d["embedder_minio"] = self.minio_models_dir / model_name / "embedder.pth"
        return d

    def get_weights_datasets(self, weights: dict):
        """
        Depends on environment.
        Get models based on directories in the weights dictionary.
        """
        if self.env != "LOCAL":
            if not os.path.exists(str(weights["trunk_local"])):
                self.minio_client.fget_object(
                    bucket_name=self.minio_bucket_name,
                    object_name=str(weights["trunk_minio"]),
                    file_path=str(weights["trunk_local"]),
                )
            if not os.path.exists(str(weights["embedder_local"])):
                self.minio_client.fget_object(
                    bucket_name=self.minio_bucket_name,
                    object_name=str(weights["embedder_minio"]),
                    file_path=str(weights["embedder_local"]),
                )
