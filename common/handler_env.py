from abc import ABC, abstractmethod

from PIL import Image
from qdrant_client.grpc import ScoredPoint

from common.consts import PROJECT_PATH
from metrics.consts import MetricCollections


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
        Additionally, filenames are returned as future captions in front-end module.
        """

    @abstractmethod
    def get_random_images_from_collection(
        self, collection_name: MetricCollections, k: int
    ) -> tuple[list[str], list[Image.Image]]:
        """
        Pulls a random set of images from a selected collection in env storage.
        Used for image input suggestion in front-end component.
        Additionally, filenames are returned as captions.
        """

    @abstractmethod
    def get_meta_json(
        self, collection_name: MetricCollections
    ) -> dict[str, list[int] | str]:
        """
        Get meta.json dictionary created during model training from env storage.
        """
