from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.transforms as T
from cycler import cycler
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from metrics.consts import MEAN, STD, TEST_TRANSFORM, TRAIN_TRANSFORM


class ImageDataset(Dataset):
    """Basic image dataset implementation"""

    def __init__(
        self,
        image_paths: list[str],
        image_labels: list[int],
        transformation: Optional[T.Compose] = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = image_labels
        self.transformation = transformation

    def __len__(self) -> Any:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Any:
        image = self.image_paths[index]
        image = Image.open(image).convert("RGB")
        if self.transformation:
            image = self.transformation(image)
        return image, self.labels[index]


@dataclass
class DatasetCombined:
    """Dto containing all dataset information"""

    df: pd.DataFrame
    train_dataset: Dataset
    test_dataset: Dataset
    x_train: np.array
    y_train: np.array
    x_test: np.array
    y_test: np.array

    @classmethod
    def get_dataset(
        cls,
        meta: Path | str,
        data_dir: Path | str,
        transformation: Optional[dict] = None,
        split: float = 0.7,
    ) -> DatasetCombined:
        """Load meta csv, splits dataframe into train/test and create torch Dataset instances"""
        df = pd.read_csv(meta, index_col=0)
        df = df.sample(
            frac=1
        )  # shuffle dataframe just in case, because split is not class balanced (not guaranteed)
        # add data_dir path to meta file, so filenames will contain absolute path for data
        df["file"] = df["file"].apply(lambda file: str(data_dir / file))
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(df["file"]), np.array(df["label"]), train_size=split
        )

        # load default or custom transformation for train/test dataset
        train_transform = TRAIN_TRANSFORM
        test_transform = TEST_TRANSFORM
        if transformation:
            train_transform = transformation.get("train", train_transform)
            test_transform = transformation.get("test", test_transform)

        # create torch datasets
        train_dataset = ImageDataset(x_train, y_train, transformation=train_transform)
        test_dataset = ImageDataset(x_test, y_test, transformation=test_transform)

        return cls(
            df=df,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )


def rgetattr(obj, attr, *args) -> Any:
    """Nested attribute getter"""

    def _getattr(obj, attr) -> Any:
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val) -> Any:
    """Nested attribute setter"""
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def visualizer_hook(_, umap_embeddings, labels, split_name, keyname, *args) -> Any:
    """Hook used for vectors visualisation using umap, args determined by pml"""
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    plt.show()


def get_transformation_with_size(size: int) -> T.Compose:
    """
    Create transformation pipeline of square resize,
    tensor cast and imagenet normalization
    """
    return T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


def save_training_meta(data: dict, path: Optional[Path] = None) -> None:
    """
    Save metadata json files into log dir to keep track of what args was passed to training cli,
    create meta.json for model loading.
    """
    path = path or Path(".")

    # convert pathlib.Path to absolute str path so it can be serialized into json
    for k, v in data.items():
        if isinstance(v, Path):
            data[k] = str(v.absolute())

    with open(path / "training_meta.json", "w") as f:
        json.dump(data, f)

    with open(path / "meta.json", "w") as f:
        json.dump(
            {"trunk": data["trunk_model"], "embedder_layers": data["embedder_layers"]},
            f,
        )
