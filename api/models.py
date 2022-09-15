from __future__ import annotations

from pydantic import BaseModel
from qdrant_client.grpc import ScoredPoint


class Payload(BaseModel):
    """Basic payload stored within vector in qdrant"""

    class_: str
    file: str
    label: int

    class Config:
        schema_extra = {
            "example": {"class_": "border_collie", "file": "167841.png", "label": 7}
        }


class SimilarityResponse(BaseModel):
    """
    Response for similarity endpoint containing most similar vectors
    sorted by distance along with payload
    """

    id: int
    version: int
    score: float
    payload: Payload

    @classmethod
    def get_from_raw_data(cls, data: ScoredPoint) -> SimilarityResponse:
        return cls(
            id=data.id,
            version=data.version,
            score=data.score,
            payload=Payload(
                class_=data.payload["class"],
                file=data.payload["file"],
                label=data.payload["label"],
            ),
        )

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "version": 1,
                "score": 0.981,
                "payload": {
                    "class_": "border_collie",
                    "file": "167841.png",
                    "label": 7,
                },
            }
        }
