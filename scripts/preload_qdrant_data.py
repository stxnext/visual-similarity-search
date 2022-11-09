import os
import zipfile
import shutil

from pathlib import Path
from loguru import logger
from pathlib import PosixPath
from distutils.dir_util import copy_tree

from common import env_function_handler
from scripts import (
    PRELOAD_APP_NAME,
    QDRANT_VOLUME_DIR,
    METRIC_COLLECTION_NAMES,
    MINIO_QDRANT_DATABASE_FILENAME,
    REQUIRED_VOLUME_FOLDERS,
)


def preload_and_unzip_qdrant_database(
    local_path: PosixPath, file_path: PosixPath, object_name: PosixPath
):
    """
    Pulls a preindexed Qdrant database from MinIO.
    """
    required_paths = [
        local_path / "collections" / name for name in METRIC_COLLECTION_NAMES
    ]
    required_paths = required_paths + [
        local_path / folder for folder in REQUIRED_VOLUME_FOLDERS
    ]
    required_paths_exist = [
        os.path.exists(target_path) for target_path in required_paths
    ]
    logger.info(f"Path list: {required_paths}")
    logger.info(f"Paths exist: {required_paths_exist}")
    if not all(required_paths_exist):
        logger.info("Start pulling data from MinIO")
        env_function_handler.get_qdrant_database_file(
            object_name=object_name, file_path=file_path
        )
        logger.info(f"Data pulled to the path: {file_path}")

        logger.info("Start unzipping data")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(local_path)

        logger.info(f"Data unzipped to: {local_path}")

        logger.info("Start replacing data")
        storage_dir = local_path / MINIO_QDRANT_DATABASE_FILENAME.split(".")[0]

        logger.info(f"{local_path=}")
        logger.info(f"{storage_dir=}")
        copy_tree(str(storage_dir), str(local_path))

        logger.info("Data replaced")

        logger.info(f"Remove folder: {storage_dir}")
        shutil.rmtree(str(storage_dir))
        os.remove(str(local_path / MINIO_QDRANT_DATABASE_FILENAME))


if __name__ == "__main__":
    logger.info("Process Started")
    local_path = PRELOAD_APP_NAME / QDRANT_VOLUME_DIR
    file_path = local_path / MINIO_QDRANT_DATABASE_FILENAME
    object_name = (
        Path(os.getenv("MINIO_MAIN_PATH")) / "data" / MINIO_QDRANT_DATABASE_FILENAME
    )

    preload_and_unzip_qdrant_database(local_path, file_path, object_name)
    logger.info("Process Ended")
