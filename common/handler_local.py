import json
import random
from pathlib import PureWindowsPath

from PIL import Image
from qdrant_client.grpc import ScoredPoint

from common.handler_env import EnvFunctionHandler
from common.utils import singleton
from metrics.consts import MetricCollections


@singleton
class LocalFunctionHandler(EnvFunctionHandler):
    """
    Managing class for local environment methods.
    """

    def get_best_score_imgs(self, results: list[ScoredPoint]) -> list[Image.Image]:
        """
        Handler for returning images with the highest similarity scores from local storage.
        Additionally, filenames are returned as future captions in front-end module.
        """
        object_list = [PureWindowsPath(r.payload["file"]).as_posix() for r in results]
        return [Image.open(obj) for obj in object_list]

    def get_random_images_from_collection(
        self, collection_name: MetricCollections, k: int
    ) -> tuple[list[str], list[Image.Image]]:
        """
        Pulls a random set of images from a selected collection in local storage.
        Used for image input suggestion in front-end component.
        Additionally, filenames are returned as captions.
        """
        local_collection_dir = self.local_metric_datasets_dir / collection_name.value
        captions_local = random.choices(list(local_collection_dir.iterdir()), k=k)
        imgs_local = []
        captions_local_str = []
        for caption in captions_local:
            imgs_local.append(Image.open(local_collection_dir / caption))
            captions_local_str.append(
                caption.name
            )  # this result is loaded directly to the application state
        return captions_local_str, imgs_local

    def get_meta_json(
        self, collection_name: MetricCollections
    ) -> dict[str, list[int] | str]:
        """
        Get meta.json dictionary created during model training from local storage.
        """
        with open(self.local_models_dir / collection_name.value / "meta.json") as f:
            return json.load(f)
