from typing import Optional

import torch
from loguru import logger
from pytorch_metric_learning.utils.common_functions import Identity
from torch import Tensor, nn
from torchvision import models as pretrained_models

from common import env_handler
from common.utils import WeightsPathGenerator
from metrics.consts import DEVICE, METRIC_COLLECTION_NAMES, MetricCollections
from metrics.utils import rgetattr, rsetattr

MODEL_TYPE = nn.DataParallel | nn.Module

# dictionary containing pretrained model names with name of last layer
supported_trunk_models = {
    "resnet50": "fc",
    "densnset121": "classifier",
}


class UnsupportedModel(Exception):
    """Exception raised when unsupported model is chosen"""


def get_trunk(
    trunk_model_name: str,
) -> tuple[nn.Module, int]:
    """Return pretrained model from torchvision with identity function on last layer and size of last layer"""
    if trunk_model_name not in supported_trunk_models:
        raise UnsupportedModel(trunk_model_name)
    trunk = getattr(pretrained_models, trunk_model_name)(pretrained=True)
    last_layer_name = supported_trunk_models[trunk_model_name]
    last_layer_size = rgetattr(trunk, f"{last_layer_name}.in_features")
    rsetattr(trunk, last_layer_name, Identity())
    trunk = trunk.to(DEVICE)
    return trunk, last_layer_size


class EmbeddingNN(nn.Module):
    """
    Embedding fully connected layer neural network.
    First layer size should be the same size as trunk output layer.
    Note: it's easier to wrap it as nn.Module instead of using just nn.Sequential, so it can be later modified.
    """

    def __init__(self, layer_sizes: list[int], final_relu: bool = False) -> None:
        super().__init__()
        self.net = self._create_net(layer_sizes, final_relu)

    def _create_net(self, layer_sizes: list[int], final_relu: bool) -> nn.Sequential:
        """Create sequential neural network from layer sizes"""
        layers = []
        num_layers = len(layer_sizes)
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Linear(input_size, curr_size))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def get_trunk_embedder(
    trunk_model_name: str,
    layer_sizes: list[int],
    data_parallel: bool = True,
    weights: Optional[WeightsPathGenerator] = None,
) -> tuple[MODEL_TYPE, MODEL_TYPE]:
    """
    Return trunk and embedder models.
    If you want to load checkpoints for models provide a dictionary with keys 'trunk' and 'embedder' with
    filepath as values
    """
    trunk, trunk_output_size = get_trunk(trunk_model_name)
    embedder = EmbeddingNN([trunk_output_size] + layer_sizes).to(DEVICE)
    if weights:
        try:
            env_handler.get_weights_datasets(weights=weights)
        except:
            logger.info(
                "Embedder and Trunk are pulled only for Cloud environment. Download models manually."
            )
        trunk.load_state_dict(torch.load(weights.trunk_local, map_location=DEVICE))
        embedder.load_state_dict(
            torch.load(weights.embedder_local, map_location=DEVICE)
        )
    if data_parallel:
        trunk = nn.DataParallel(trunk)
        embedder = nn.DataParallel(embedder)
    return trunk, embedder


def get_full_pretrained_model(
    collection_name: MetricCollections, data_parallel: bool = True
) -> MODEL_TYPE:
    """Get full pretrained model with loaded weights"""
    if collection_name.value not in METRIC_COLLECTION_NAMES:
        raise UnsupportedModel(collection_name.value)
    meta = env_handler.get_meta_json(collection_name=collection_name)
    weights = WeightsPathGenerator(collection_name=collection_name)
    trunk, embedder = get_trunk_embedder(
        meta["trunk"], meta["embedder_layers"], data_parallel=False, weights=weights
    )
    trunk.to(DEVICE)
    embedder.to(DEVICE)
    model = nn.Sequential(trunk, embedder)
    if data_parallel:
        model = nn.DataParallel(model)
    model.eval()
    return model
