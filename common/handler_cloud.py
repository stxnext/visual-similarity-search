import json
import os
import random
from pathlib import Path, PureWindowsPath

from minio import Minio
from PIL import Image
from qdrant_client.grpc import ScoredPoint

from common.consts import MINIO_BUCKET_NAME, MINIO_MAIN_PATH
from common.handler_env import EnvFunctionHandler
from common.utils import WeightsPathGenerator, singleton
from metrics.consts import MetricCollections


@singleton
class CloudFunctionHandler(EnvFunctionHandler):
    """
    Managing class for cloud environment methods.
    """

    minio_data_dir = MINIO_MAIN_PATH / "data"
    minio_models_dir = minio_data_dir / "models"
    minio_metric_datasets_dir = minio_data_dir / "metric_datasets"

    def __init__(self):
        self.minio_client = Minio(
            endpoint=os.getenv("MINIO_HOST"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=True,
        )

    def get_qdrant_database_file(self, file_path: Path, object_name: Path) -> None:
        """
        Pulls zipped Qdrant database snapshot from cloud's object_name to container's file_path.
        """
        self.minio_client.fget_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=str(object_name),
            file_path=str(file_path),
        )

    def get_best_score_imgs(self, results: list[ScoredPoint]) -> list[Image.Image]:
        """
        Handler for returning images with the highest similarity scores from cloud storage.
        Additionally, filenames are returned as future captions in front-end module.
        """
        object_list = [
            MINIO_MAIN_PATH / PureWindowsPath(r.payload["file"]).as_posix()
            for r in results
        ]
        return [
            Image.open(
                self.minio_client.get_object(
                    bucket_name=MINIO_BUCKET_NAME, object_name=str(obj)
                )
            )
            for obj in object_list
        ]

    def get_random_images_from_collection(
        self, collection_name: MetricCollections, k: int
    ) -> tuple[list[str], list[Image.Image]]:
        """
        Pulls a random set of images from a selected collection in cloud storage.
        Used for image input suggestion in front-end component.
        Additionally, filenames are returned as captions.
        """
        objects = self.minio_client.list_objects(
            bucket_name=MINIO_BUCKET_NAME,
            prefix=f"{str(self.minio_metric_datasets_dir / collection_name.value)}/",  # lists objects in "SOME_PATH/"
        )
        object_list = [obj.object_name for obj in objects]
        object_sample_list = random.choices(object_list, k=k)
        captions_cloud = [obj.split("/")[-1] for obj in object_sample_list]
        imgs_cloud = [
            Image.open(
                self.minio_client.get_object(
                    bucket_name=MINIO_BUCKET_NAME, object_name=str(obj)
                )
            )
            for obj in object_sample_list
        ]
        return captions_cloud, imgs_cloud

    def get_meta_json(
        self, collection_name: MetricCollections
    ) -> dict[str, list[int] | str]:
        """
        Get meta.json dictionary created during model training from cloud storage.
        """
        return json.load(
            self.minio_client.get_object(
                bucket_name=MINIO_BUCKET_NAME,
                object_name=str(
                    self.minio_models_dir / collection_name.value / "meta.json"
                ),
            )
        )

    def get_weights_datasets(self, weights: WeightsPathGenerator) -> None:
        """
        Pull embedder and trunk files to the container from cloud storage. Empty for local.
        """
        if not weights.trunk_local.is_file():
            self.minio_client.fget_object(
                bucket_name=MINIO_BUCKET_NAME,
                object_name=str(weights.trunk_minio),
                file_path=str(weights.trunk_local),
            )
        if not weights.embedder_local.is_file():
            self.minio_client.fget_object(
                bucket_name=MINIO_BUCKET_NAME,
                object_name=str(weights.embedder_minio),
                file_path=str(weights.embedder_local),
            )
