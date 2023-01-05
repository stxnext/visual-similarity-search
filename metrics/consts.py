from enum import Enum

import torch
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEARCH_RESPONSE_LIMIT = 20

DEFAULT_SPLIT = 0.7
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.00001
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_SAMPLER_M = 2
DEFAULT_DATALOADER_NUM_WORKERS = 2
DATALOADER_WORKERS = 2
BASE_PAYLOAD_KEYS = ["file", "class", "label"]

DEFAULT_EMBEDDER_LAYERS = [1024, 1024, 1024]
SIZE = (300, 300)  # input size for neural net
MEAN = (0.485, 0.456, 0.406)  # mean and std for imagenet dataset
STD = (0.229, 0.224, 0.225)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(SIZE),
        transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=SIZE),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

TEST_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

RESIZE_TRANSFORM = transforms.Compose([transforms.Resize(SIZE), transforms.ToTensor()])

INFER_TRANSFORM = TEST_TRANSFORM


class MetricCollections(Enum):
    """
    Enum of available collections and pretrained models for similarity.
    """

    DOGS = "dogs"
    SHOES = "shoes"
    CELEBRITIES = "celebrities"
    LOGOS = "logos"
    WASTE = "waste"


METRIC_COLLECTION_NAMES = [x.value for x in iter(MetricCollections)]
