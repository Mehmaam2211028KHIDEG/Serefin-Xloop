import os
import boto3
from pathlib import Path
from threading import Thread, Lock
from queue import Queue
import logging
import time

class S3Downloader:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.download_queue = Queue(maxsize=2)  # Queue to hold paths of files to be downloaded
        self.download_lock = Lock()
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup detailed logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of log messages
        if not logger.handlers:
            fh = logging.FileHandler('logs/s3_downloader.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        return logger

    def list_webm_files(self, prefix=''):
        """List all .webm files in the bucket under the given prefix."""
        self.logger.info(f"Listing .webm files in bucket {self.bucket_name} with prefix '{prefix}'")
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.webm'):
                    self.logger.debug(f"Found .webm file: {obj['Key']}")
                    yield obj['Key']

    def download_file(self, key, local_dir):
        """Download a single file from S3."""
        local_path = Path(local_dir) / Path(key).name
        try:
            self.s3_client.download_file(self.bucket_name, key, str(local_path))
            self.logger.info(f"Downloaded {key} to {local_path}")
        except Exception as e:
            self.logger.error(f"Failed to download {key}: {str(e)}")
        return local_path

    def manage_downloads(self, local_dir):
        """Manage file downloads ensuring only two are done at a time."""
        while True:
            key = self.download_queue.get()
            if key is None:
                self.logger.info("No more files to download, stopping thread.")
                break  # If None is fetched from the queue, stop the thread
            self.download_file(key, local_dir)

    def monitor_file(self, file_path):
        """Monitor the downloaded file and download a new one if it's deleted."""
        while True:
            if not file_path.exists():
                self.logger.warning(f"{file_path} has been removed.")
                with self.download_lock:
                    next_key = next(self.webm_files, None)
                    if next_key:
                        self.logger.info(f"Queueing next file for download: {next_key}")
                        self.download_queue.put(next_key)
                break

    def start_downloading(self, local_dir):
        """Start the download process with daily checks for new files."""
        while True:
            self.webm_files = self.list_webm_files()  # Generator to list files
            self.logger.info("Checking for new files to download.")
            has_files = False
            with self.download_lock:
                for _ in range(2):  # Try to enqueue two files
                    next_key = next(self.webm_files, None)
                    if next_key:
                        self.download_queue.put(next_key)
                        has_files = True
            if not has_files:
                self.logger.info("No new files found. Checking again in 24 hours.")
                time.sleep(86400)  # Sleep for 24 hours before checking again
            else:
                self.logger.info("Starting download threads.")
                # Start threads to manage downloads
                threads = []
                for _ in range(2):  # Start two threads to download files
                    thread = Thread(target=self.manage_downloads, args=(local_dir,))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()  # Wait for all threads to finish
                    
