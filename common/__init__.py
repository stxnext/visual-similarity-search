import os

from qdrant_client import QdrantClient
from global_utils import CloudFunctionHandler, LocalFunctionHandler

env_handler = LocalFunctionHandler() if os.getenv("TYPE") == "LOCAL" else CloudFunctionHandler()
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT"))
)
