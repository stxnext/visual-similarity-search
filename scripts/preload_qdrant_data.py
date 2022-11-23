import os
import shutil
import zipfile
from distutils.dir_util import copy_tree

from loguru import logger

from common import env_handler
from common.consts import MINIO_MAIN_PATH
from metrics.consts import METRIC_COLLECTION_NAMES
from scripts import (
    MINIO_QDRANT_DATABASE_FILENAME,
    QDRANT_VOLUME_DIR,
    REQUIRED_VOLUME_FOLDERS,
)


def preload_and_unzip_qdrant_database() -> None:
    """
    Pulls a pre-indexed Qdrant database from MinIO.
    """
    logger.info("Process Started")
    file_path = QDRANT_VOLUME_DIR / MINIO_QDRANT_DATABASE_FILENAME
    object_name = MINIO_MAIN_PATH / "data" / MINIO_QDRANT_DATABASE_FILENAME
    required_paths = [
        QDRANT_VOLUME_DIR / "collections" / name for name in METRIC_COLLECTION_NAMES
    ]
    required_paths = required_paths + [
        QDRANT_VOLUME_DIR / folder for folder in REQUIRED_VOLUME_FOLDERS
    ]
    required_paths_exist = [target_path.is_dir() for target_path in required_paths]
    logger.info(f"File path: {file_path}")
    logger.info(f"Object name: {object_name}")
    logger.info(f"Path list: {required_paths}")
    logger.info(f"Paths exist: {required_paths_exist}")
    if not all(required_paths_exist):
        logger.info("Start pulling data from MinIO")
        env_handler.get_qdrant_database_file(
            object_name=object_name, file_path=file_path
        )
        logger.info(f"Data pulled to the path: {file_path}")

        logger.info("Start unzipping data")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(QDRANT_VOLUME_DIR)

        logger.info(f"Data unzipped to: {QDRANT_VOLUME_DIR}")

        logger.info("Start replacing data")
        storage_dir = QDRANT_VOLUME_DIR / MINIO_QDRANT_DATABASE_FILENAME.split(".")[0]

        logger.info(f"{QDRANT_VOLUME_DIR=}")
        logger.info(f"{storage_dir=}")
        copy_tree(str(storage_dir.absolute()), str(QDRANT_VOLUME_DIR.absolute()))

        logger.info("Data replaced")

        logger.info(f"Remove folder: {storage_dir}")
        shutil.rmtree(str(storage_dir.absolute()))
        os.remove(str((QDRANT_VOLUME_DIR / MINIO_QDRANT_DATABASE_FILENAME).absolute()))

    logger.info("Process Ended")


if __name__ == "__main__":
    preload_and_unzip_qdrant_database()
