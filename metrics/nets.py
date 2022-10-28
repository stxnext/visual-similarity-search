import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from metrics.consts import METRIC_COLLECTION_NAMES
from pytorch_metric_learning.utils.common_functions import Identity
from torch import Tensor, nn
from torchvision import models as pretrained_models

from metrics import (
    minio_client,
    MINIO_BUCKET_NAME,
    MINIO_MODELS_DIR,
    DEVICE,
    MODELS_DIR,
)
from metrics.utils import rgetattr, rsetattr

MODEL_TYPE = Union[nn.DataParallel, nn.Module]

# dictionary containing pretrained model names with name of last layer
supported_trunk_models = {
    "resnet50": "fc",
    "densnset121": "classifier",
}


class UnsupportedModel(Exception):
    """Exception raised when unsupported model is chosen"""


def get_trunk(
    trunk_model_name: str,
) -> Tuple[nn.Module, int]:
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

    def __init__(self, layer_sizes: List[int], final_relu: bool = False):
        super().__init__()
        self.net = self._create_net(layer_sizes, final_relu)

    def _create_net(self, layer_sizes: List[int], final_relu: bool) -> nn.Sequential:
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
    layer_sizes: List[int],
    data_parallel: bool = True,
    weights: Optional[Dict] = None,
) -> Tuple[MODEL_TYPE, MODEL_TYPE]:
    """
    Return trunk and embedder models.
    If you want to load checkpoints for models provide a dictionary with keys 'trunk' and 'embedder' with
    filepath as values
    """
    trunk, trunk_output_size = get_trunk(trunk_model_name)
    embedder = EmbeddingNN([trunk_output_size] + layer_sizes).to(DEVICE)
    if weights:
        if not os.path.exists(weights["trunk_local"]):
            if os.getenv("TYPE") != "LOCAL":
                minio_client.fget_object(
                    MINIO_BUCKET_NAME, weights["trunk_minio"], weights["trunk_local"]
                )
        if not os.path.exists(weights["embedder_local"]):
            if os.getenv("TYPE") != "LOCAL":
                minio_client.fget_object(
                    MINIO_BUCKET_NAME,
                    weights["embedder_minio"],
                    weights["embedder_local"],
                )
        trunk.load_state_dict(torch.load(weights["trunk_local"], map_location=DEVICE))
        embedder.load_state_dict(
            torch.load(weights["embedder_local"], map_location=DEVICE)
        )
    if data_parallel:
        trunk = nn.DataParallel(trunk)
        embedder = nn.DataParallel(embedder)
    return trunk, embedder


def get_full_pretrained_model(
    model_name: str, data_parallel: bool = True
) -> MODEL_TYPE:
    """Get full pretrained model with loaded weights"""
    if model_name not in METRIC_COLLECTION_NAMES:
        raise UnsupportedModel(model_name)
    if os.getenv("TYPE") == "LOCAL":
        with open(os.path.join(MODELS_DIR, model_name, "meta.json")) as f:
            meta = json.load(f)
    else:
        meta = json.load(
            minio_client.get_object(
                bucket_name=MINIO_BUCKET_NAME,
                object_name=os.path.join(MINIO_MODELS_DIR, model_name, "meta.json"),
            )
        )
    weights = {
        "trunk_minio": os.path.join(MINIO_MODELS_DIR, model_name, "trunk.pth"),
        "embedder_minio": os.path.join(MINIO_MODELS_DIR, model_name, "embedder.pth"),
        "trunk_local": os.path.join(MODELS_DIR, model_name, "trunk.pth"),
        "embedder_local": os.path.join(MODELS_DIR, model_name, "embedder.pth"),
    }
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
