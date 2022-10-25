import os
import zipfile
import shutil

from loguru import logger
from distutils.dir_util import copy_tree

from preload import (
    minio_client,
    QDRANT_PRELOAD_APP_NAME,
    SHARED_VOLUME_DIR,
    MINIO_BUCKET_NAME,
    MINIO_DATA_DIR,
    METRIC_COLLECTION_NAMES,
)


STORAGE_NAME = "qdrant_storage.zip"


def preload_and_unzip_qdrant_database(
    local_path: str, file_path: str, object_name: str
):
    """
    Pulls a preindexed Qdrant database from MinIO.
    """
    replace_files = ["raft_state"]
    replace_folders = ["aliases", "collections", "collections_meta_wal"]
    target_paths = [
        f"{local_path}/collections/{name}" for name in METRIC_COLLECTION_NAMES
    ]
    target_paths = target_paths + [
        f"{local_path}/{folder}" for folder in replace_folders
    ]
    target_paths_exist = [os.path.exists(target_path) for target_path in target_paths]
    logger.info(f"Path list: {target_paths}")
    logger.info(f"Paths exist: {target_paths_exist}")
    if not all(target_paths_exist):
        logger.info("Start pulling data from MinIO")
        minio_client.fget_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=object_name,
            file_path=file_path,
        )
        logger.info(f"Data pulled to the path: {file_path}")

        logger.info("Start unzipping data")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(local_path)

        logger.info(f"Data unzipped to: {local_path}")

        logger.info("Start replacing data")
        storage_dir = os.path.join(local_path, STORAGE_NAME.split(".")[0])

        for file in replace_files:
            shutil.copy(f"{storage_dir}/{file}", f"{local_path}/{file}")

        for folder in replace_folders:
            copy_tree(f"{storage_dir}/{folder}", f"{local_path}/{folder}")

        logger.info("Data replaced")

        logger.info(f"Remove folder: {storage_dir}")
        shutil.rmtree(storage_dir)
        os.remove(os.path.join(local_path, STORAGE_NAME))


if __name__ == "__main__":
    logger.info("Process Started")
    local_path = os.path.join(QDRANT_PRELOAD_APP_NAME, SHARED_VOLUME_DIR)
    file_path = os.path.join(local_path, STORAGE_NAME)
    object_name = os.path.join(MINIO_DATA_DIR, STORAGE_NAME)

    preload_and_unzip_qdrant_database(local_path, file_path, object_name)
    logger.info("Process Ended")
