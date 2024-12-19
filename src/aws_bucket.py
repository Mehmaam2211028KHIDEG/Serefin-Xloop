import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Optional
import time

import boto3
from botocore.exceptions import ClientError


class S3ProcessingPipeline:
    def __init__(
        self, bucket_name: str, download_dir: str = "downloads", max_concurrent: int = 2
    ):
        self.bucket_name = bucket_name
        self.download_dir = Path(download_dir)
        self.max_concurrent = max_concurrent
        self.s3_client = boto3.client("s3")
        self.processing_queue = Queue()
        self.stop_event = Event()
        self.active_downloads = 0  # Track number of active downloads
        self.download_complete = Event()  # Signal when download is complete

        # Create directories
        self.download_dir.mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        # Set up logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the pipeline"""
        # Main processing log
        self.logger = logging.getLogger("processing")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler("logs/processing.log")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(fh)

        # Error log
        self.error_logger = logging.getLogger("errors")
        self.error_logger.setLevel(logging.ERROR)
        eh = logging.FileHandler("logs/errors.log")
        eh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.error_logger.addHandler(eh)

    def verify_bucket_access(self) -> bool:
        """Verify access to S3 bucket"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "403":
                self.error_logger.error(
                    f"No permission to access bucket '{self.bucket_name}'"
                )
            elif error_code == "404":
                self.error_logger.error(f"Bucket '{self.bucket_name}' does not exist")
            return False

    def download_file(self, s3_key: str) -> Optional[Path]:
        """Download a single file from S3 maintaining directory structure"""
        try:
            # Create full local path maintaining S3 structure
            local_path = self.download_dir / s3_key

            # Create parent directories if they don't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Downloading {s3_key} to {local_path}")

            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            return local_path

        except Exception as e:
            self.error_logger.error(f"Error downloading {s3_key}: {str(e)}")
            if local_path.exists():
                local_path.unlink()
            return None

    def process_file(self, local_path: Path):
        """Process a single file through transcription"""
        try:
            self.logger.info(f"Starting transcription for {local_path}")
            # Let the main process handle transcription
            self.active_downloads -= 1  # Decrease active downloads count
            return True

        except Exception as e:
            self.error_logger.error(f"Error processing {local_path}: {str(e)}")
            self.active_downloads -= 1  # Ensure we decrease count even on error
            return False

    def start_processing(self):
        """Start the download pipeline with strictly controlled downloads"""
        try:
            if not self.verify_bucket_access():
                self.error_logger.error("Failed to verify bucket access")
                return

            self.logger.info(f"Starting download process for bucket: {self.bucket_name}")

            # Get list of .webm files
            paginator = self.s3_client.get_paginator("list_objects_v2")
            s3_files = []
            for page in paginator.paginate(Bucket=self.bucket_name):
                for obj in page.get("Contents", []):
                    if obj["Key"].lower().endswith(".webm"):
                        s3_files.append(obj["Key"])
                        # Only get 2 files at a time
                        if len(s3_files) >= 2:
                            break
                if len(s3_files) >= 2:
                    break

            # Download and process up to 2 files
            for s3_key in s3_files:
                local_path = self.download_file(s3_key)
                if local_path:
                    self.active_downloads += 1
                    self.process_file(local_path)

        except Exception as e:
            self.error_logger.error(f"Pipeline error: {str(e)}")
            self.stop_event.set()


if __name__ == "__main__":
    # Example usage
    pipeline = S3ProcessingPipeline(
        bucket_name="access-oap-prod-twilio-bucket",
        download_dir="downloads",
        max_concurrent=2,
    )

    pipeline.start_processing()

