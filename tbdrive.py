'''
Source adapted from:
https://raw.githubusercontent.com/Lightning-Universe/Training-Studio-app/master/lightning_training_studio/loggers/tensorboard.py
'''
import concurrent.futures
import os
from pathlib import Path
from time import time
from typing import Any, Dict, Optional

import lightning as L

from lightning.pytorch.loggers import TensorBoardLogger

from lightning.app.storage.path import _filesystem
from lightning.app.storage import Drive

from fsspec.implementations.local import LocalFileSystem

'''
Manage uploads of log and checkpoint files. Log files are uploaded during execution,
while checkpoints are uploaded when execution terminates.
'''
class DriveTensorBoardLogger(TensorBoardLogger):
    def __init__(self, *args, drive: Drive, refresh_time: int = 5, **kwargs):
        super().__init__(*args, flush_secs=refresh_time, max_queue=1, **kwargs)
        self.timestamp = None
        self.drive = drive
        self.refresh_time = refresh_time
        self.ckpt_epoch = 0

    def log_metrics(self, metrics, step) -> None:
        super().log_metrics(metrics, step)

        # Upload logging files
        if self.timestamp is None:
            self._upload_to_storage(logs_only=True)
            self.timestamp = time()
        elif (time() - self.timestamp) > self.refresh_time:
            self._upload_to_storage(logs_only=True)
            self.timestamp = time()

    def finalize(self, status: str) -> None:
        super().finalize(status)

    def _upload_to_storage(self, logs_only=True):
        fs = _filesystem()
        fs.invalidate_cache()

        source_path = Path(self.log_dir).resolve()
        destination_path = self.drive._to_shared_path(self.log_dir, component_name=self.drive.component_name)

        def _copy(from_path: Path, to_path: Path) -> Optional[Exception]:

            try:
                # NOTE: S3 does not have a concept of directories, so we do not need to create one.
                if isinstance(fs, LocalFileSystem):
                    fs.makedirs(str(to_path.parent), exist_ok=True)

                # Only copy log files if flagged
                if logs_only:
                    if "events.out.tfevents" in str(from_path):
                        fs.put(str(from_path), str(to_path), recursive=False)
                else:
                    fs.put(str(from_path), str(to_path), recursive=False)

            except Exception as e:
                # Return the exception so that it can be handled in the main thread
                return e

        src = [file for file in source_path.rglob("*") if file.is_file()]
        dst = [destination_path / file.relative_to(source_path) for file in src]

        with concurrent.futures.ThreadPoolExecutor(4) as executor:
            results = executor.map(_copy, src, dst)

        # Raise the first exception found
        exception = next((e for e in results if isinstance(e, Exception)), None)
        if exception:
            raise exception
