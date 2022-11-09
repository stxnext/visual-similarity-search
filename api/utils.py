import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Iterator

from fastapi import UploadFile


@contextmanager
def create_named_temporary_files(
    **kwargs: UploadFile,
) -> Iterator[dict[str, IO[bytes]]]:
    """
    Create dictionary containing reference to temporary files.
    Creating temporary files are slower than using byte-streams directly, but some
    python libraries are based on loading object by providing filenames. It can be later
    improved, but right now performance on those operations are not critical.
    """
    files_dict = {key: _create_temp_file(value) for key, value in kwargs.items()}
    try:
        yield files_dict
    finally:
        for file in files_dict.values():
            file.close()
            os.remove(file.name)


def _create_temp_file(upload_file: UploadFile) -> IO[bytes]:
    """
    Private method used by create_named_temporary_files
    for creating single file and closing uploaded one.
    """
    try:
        suffix = Path(upload_file.filename).suffix
        data = upload_file.file.read()
        return _create_temp_file_from_bytes(data, suffix)
    finally:
        upload_file.file.close()


def _create_temp_file_from_bytes(data: bytes, suffix: str) -> IO[bytes]:
    """
    Private method used by create_named_temporary_files
    for creating single file and closing uploaded one.
    """
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(data)
    return temp_file
