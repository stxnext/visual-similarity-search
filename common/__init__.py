import os

from qdrant_client import QdrantClient
from common.global_utils import EnvFunctionHandler


env_function_handler = EnvFunctionHandler()
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT"))
)
