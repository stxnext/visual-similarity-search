import torch
import torchvision

from dataclasses import dataclass
from pathlib import Path
from typing import Union
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from qdrant_client.grpc import ScoredPoint
from torchvision.transforms.transforms import Compose
from tqdm.auto import tqdm

from common import env_function_handler, qdrant_client

from metrics.consts import (
    DEVICE,
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


def init_all_metric_models() -> dict[str, MetricModel]:
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
        self.models = init_all_metric_models()
        self.device = device_name

    def _single_img_infer(self, model: MetricModel, img: Image.Image) -> list[float]:
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
    ) -> list[ScoredPoint]:
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

        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            query_filter=None,  # TODO: add filtering feature
            limit=limit,
        )
        return search_result

    def get_best_choice_for_uploaded_image(
        self, base_img: Image.Image, collection_name: str, benchmark: int, k: int = 25
    ) -> tuple[Image.Image, list[Image.Image]]:
        """
        Search for similar images of random image from given collection.
        Returns tuple of images [anchor_image, grid image of k most similar images (the biggest cosine similarity)]
        """
        results = self.search(base_img, collection_name, limit=k)
        results_bench = [r for r in results if round(r.score, 4) >= benchmark / 100]
        scores_bench = [100 * round(r.score, 4) for r in results_bench]
        if len(results_bench) > 0:
            imgs = [
                RESIZE_TRANSFORM(img)
                for img in env_function_handler.get_best_score_imgs(
                    results=results_bench
                )
            ]
            to_image = torchvision.transforms.ToPILImage()
            imgs_transformed = [to_image(img) for img in imgs]

            # Adding similarity scores to the images with best
            for i, img in enumerate(imgs_transformed):
                draw = ImageDraw.Draw(img)
                draw.text(
                    xy=(10, 10),
                    text="{0:.2f}%".format(scores_bench[i]),
                    font=ImageFont.truetype("DejaVuSans-Bold.ttf", 40),
                    fill=(0, 255, 0),
                )
        else:
            imgs_transformed = None
        return base_img, imgs_transformed
