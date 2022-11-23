from typing import Any, Optional

from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image

from api.models import SimilarityResponse
from api.utils import create_named_temporary_files
from metrics.consts import SEARCH_RESPONSE_LIMIT, MetricCollections
from metrics.core import MetricClient

router = APIRouter()


@router.get("/ping")
async def ping_check() -> dict[str, str]:
    return {"msg": "pong"}


@router.post("/search/{collection_name}", response_model=list[SimilarityResponse])
async def upload_file(
    request: Request,
    collection_name: MetricCollections,
    file: UploadFile = File(...),
    limit: Optional[int] = SEARCH_RESPONSE_LIMIT,
) -> list[Any]:
    """Upload file with collection name and return most similar objects"""
    metric_client: MetricClient = request.app.state.metric_client
    with create_named_temporary_files(file=file) as files:
        img = Image.open(files["file"].name)
        raw_results = metric_client.search(img, collection_name, limit=limit)
        return list(map(SimilarityResponse.get_from_raw_data, raw_results))
