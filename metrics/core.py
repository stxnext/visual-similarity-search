import random

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import os
import torch
import torchvision
from loguru import logger
from PIL import Image
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.grpc import ScoredPoint
from torchvision.transforms.transforms import Compose
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from metrics import (
    DEVICE,
    MINIO_BUCKET_NAME,
    MINIO_MAIN_PATH,
    MINIO_METRIC_TYPES_DIR,
    qdrant_client,
    minio_client,
)
from metrics.consts import (
    INFER_TRANSFORM,
    METRIC_COLLECTION_NAMES,
    RESIZE_TRANSFORM,
    SEARCH_RESPONSE_LIMIT,
    MetricCollections,
)
from metrics.nets import MODEL_TYPE, get_full_pretrained_model
from metrics.utils import singleton


class InvalidCollectionName(Exception):
    """Exception raised when name of collection name is invalid"""


@dataclass
class MetricModel:
    model: MODEL_TYPE
    transformation: Compose  # TODO: check if we can define it inside model meta.json files (serialize Compose)

    def __post_init__(self):
        self.model.eval()


def init_all_metric_models() -> Dict[str, MetricModel]:
    """Load all metrics models into memory and return in form of dict"""
    logger.info(f"Loading metric models: {METRIC_COLLECTION_NAMES}")
    return {
        name: MetricModel(
            model=get_full_pretrained_model(name, data_parallel=False),
            transformation=INFER_TRANSFORM,
        )
        for name in tqdm(METRIC_COLLECTION_NAMES)
    }


@singleton
class MetricClient:
    """Main client written as a simple bridge between metric search and api"""

    def __init__(self, device_name: str = DEVICE) -> None:
        self.qdrant_client: QdrantClient = qdrant_client
        self.minio_client: Minio = minio_client
        self.models = init_all_metric_models()
        self.device = device_name

    def _single_img_infer(self, model: MetricModel, img: Image.Image) -> List[float]:
        """Perform single inference of image with proper model and return embeddings"""
        img = model.transformation(img)
        model = model.model
        with torch.no_grad():
            if self.device == torch.device("cuda"):
                raw_embedding = model(img.cuda().unsqueeze(0))
                embedding = list(raw_embedding.cpu().data.numpy())[0]
            else:
                raw_embedding = model(
                    img.cpu().unsqueeze(0)
                )  # check that on cpu to make sure its valid !!!
                embedding = list(raw_embedding.cpu().data.numpy())[0]
        return embedding

    def search(
        self,
        img: Union[str, Path, Image.Image],
        collection_name: Union[str, MetricCollections],
        limit: int = SEARCH_RESPONSE_LIMIT,
    ) -> List[ScoredPoint]:
        """Search for most similar images (vectors) using qdrant engine"""

        if isinstance(collection_name, MetricCollections):
            collection_name = collection_name.value
        if collection_name not in METRIC_COLLECTION_NAMES:
            raise InvalidCollectionName(collection_name)

        if isinstance(img, (str, Path)):
            img = Image.open(img)
        img = img.convert("RGB")

        model = self.models[collection_name]
        embedding = self._single_img_infer(model, img)

        search_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            query_filter=None,  # TODO: add filtering feature
            limit=limit,
        )
        return search_result

    def _get_best_choice_for_uploaded_image(
        self, img: Image.Image, collection_name: str, grid_nrow: int, k: int = 25
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Search for similar images of random image from given collection.
        Returns tuple of images [anchor_image, grid image of k most similar images (the biggest cosine similarity)
        """
        results = self.search(img, collection_name, limit=k)
        object_list = [
            os.path.join(MINIO_MAIN_PATH, r.payload["file"]) for r in results
        ]
        imgs = [
            RESIZE_TRANSFORM(
                Image.open(self.minio_client.get_object(MINIO_BUCKET_NAME, obj))
            )
            for obj in object_list
        ]
        grid = make_grid(imgs, nrow=grid_nrow)
        grid_img = torchvision.transforms.ToPILImage()(grid)
        return img, grid_img

    def _get_random_images_from_collection(
        self, collection_name: str, k: int = 5
    ) -> (List[str], List[Image.Image]):
        """
        Pulls a random set of images from a selected collection. Used for search suggestion in front-end component.
        """
        objects = self.minio_client.list_objects(
            "ml-demo", prefix=f"{MINIO_METRIC_TYPES_DIR}/{collection_name}/"
        )
        object_list = [obj.object_name for obj in objects]
        object_sample_list = random.choices(object_list, k=k)
        captions = [obj.split("/")[-1] for obj in object_sample_list]
        imgs = [
            Image.open(self.minio_client.get_object("ml-demo", object_name=obj))
            for obj in object_sample_list
        ]
        return captions, imgs
