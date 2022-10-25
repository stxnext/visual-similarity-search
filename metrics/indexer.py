from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from loguru import logger
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import Distance
from qdrant_client.http import models
from tqdm.auto import tqdm

from metrics.consts import INFER_TRANSFORM
from metrics.nets import get_full_pretrained_model
from metrics.utils import DatasetCombined

DISTANCES = Literal["Cosine", "Euclid", "Dot"]


def shoes_filter(meta: List[Dict]) -> List[Dict]:
    """Filter out most of the payload keys to prevent json decode error"""
    new_meta = []
    important_keys = {"file", "class", "label"}
    for d in meta:
        new_meta.append({k: d[k] for k in d.keys() & important_keys})
    return new_meta


# TODO: check if we need wrapper func for singleton client
def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: Union[Distance, DISTANCES],
):
    """Wrapper function for auto injecting qdrant client object and creating collection"""
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=distance),
    )


# TODO: add support for batch gpu inference to speed up index upload
def upload_indexes(
    client: QdrantClient,
    collection_name: str,
    meta_file: Union[Path, str],
    dataset_dir: Union[Path, str],
    qdrant_batch: int = 256,
    meta_filter: Optional[Callable] = None,
) -> None:
    """Helper function for creating embeddings and uploading them to qdrant"""
    logger.info(f"Loading model: {collection_name}")
    model = get_full_pretrained_model(collection_name, data_parallel=False)
    model.eval()
    dataset = DatasetCombined.get_dataset(meta_file, dataset_dir)
    embeddings = []
    meta_data = []
    df = dataset.df
    df = df.fillna("")  # JSON does not support np.nan and pd.NaN
    logger.info(f"Started indexing {len(df)} vectors for collection {collection_name}")
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        img = INFER_TRANSFORM(Image.open(row["file"]).convert("RGB"))
        with torch.no_grad():
            if torch.cuda.is_available():
                embedding = model(img.cuda().unsqueeze(0))[0, :]
            else:
                embedding = model(img.unsqueeze(0))[0, :]
        embeddings.append(embedding.cpu().data.numpy())
        meta_data.append(dict(row))
    embeddings = np.array(embeddings)

    if meta_filter:
        meta_data = meta_filter(meta_data)

    client.upload_collection(
        collection_name=collection_name,
        vectors=embeddings,
        payload=meta_data,
        ids=None,
        batch_size=qdrant_batch,
    )
