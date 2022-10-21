import os
import zipfile
import shutil

from distutils.dir_util import copy_tree

from preload import minio_client, QDRANT_PRELOAD_APP_NAME, SHARED_VOLUME_DIR, MINIO_BUCKET_NAME, MINIO_DATA_DIR


STORAGE_NAME = "qdrant_storage.zip"


def preload_and_unzip_qdrant_database(local_path: str, file_path: str, object_name: str):
    """
    Pulls a preindexed Qdrant database from MinIO.
    """
    if not os.path.exists(file_path):
        print("Start pulling data from MinIO")
        minio_client.fget_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=object_name,
            file_path=file_path,
        )
        print(f"Data pulled to the path: {file_path}")

        print("Start unzipping data")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(local_path)

        print("Data unzipped")

        print("Start replacing data")
        storage_dir = os.path.join(local_path, STORAGE_NAME.split(".")[0])
        replace_files = ["raft_state"]
        replace_folders = ["aliases", "collections", "collections_meta_wal"]
        print(storage_dir, "\n", local_path)

        for file in replace_files:
            shutil.copy(f"{storage_dir}/{file}", f"{local_path}/{file}")

        for folder in replace_folders:
            copy_tree(f"{storage_dir}/{folder}", f"{local_path}/{folder}")

        print("Data replaced")

        print(f"Remove folder {storage_dir}")
        shutil.rmtree(storage_dir)
        os.remove(os.path.join(local_path, STORAGE_NAME))


if __name__ == "__main__":
    print("Start Process")
    local_path = os.path.join(QDRANT_PRELOAD_APP_NAME, SHARED_VOLUME_DIR)
    file_path = os.path.join(local_path, STORAGE_NAME)
    object_name = os.path.join(MINIO_DATA_DIR, STORAGE_NAME)

    print(f"pwd: {os.getcwd()}")
    print(f"local_path: {local_path}")
    print(f"file_path: {file_path}")
    print(f"object_name: {object_name}")

    preload_and_unzip_qdrant_database(local_path, file_path, object_name)
    print("Process Ended")
