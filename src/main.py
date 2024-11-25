import logging
import time
from pathlib import Path

from aws_bucket import S3ProcessingPipeline
from transcription import AudioTranscriber


def setup_logging():
    """Setup logging with directory creation"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("main_execution")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fh = logging.FileHandler("logs/main_execution.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def ensure_directories():
    """Ensure all required directories exist"""
    directories = ["output", "transcriptions", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def process_files(
    downloader: S3ProcessingPipeline,
    transcriber: AudioTranscriber,
    local_dir: str,
    logger: logging.Logger,
):
    """Process files with download and transcription"""
    while True:
        try:
            # Start download manager
            download_thread = downloader.start_downloading(local_dir)

            # Monitor the output directory for files
            output_dir = Path(local_dir)
            while True:
                # Process any existing files
                for file_path in output_dir.glob("*.webm"):
                    logger.info(f"Processing file: {file_path}")

                    # Attempt transcription
                    result = transcriber.transcribe_audio(file_path)
                    if result:
                        # Save transcription
                        transcriber._save_transcription(file_path, result)
                        logger.info(f"Successfully transcribed: {file_path}")

                        # Remove the original file
                        file_path.unlink()
                        logger.info(f"Removed processed file: {file_path}")
                    else:
                        logger.error(f"Failed to transcribe: {file_path}")

                # Brief pause before next check
                time.sleep(5)

        except Exception as e:
            logger.error(f"Error in processing loop: {str(e)}")
            time.sleep(60)  # Wait before retrying


def main():
    logger = setup_logging()
    logger.info("Starting the application")

    try:
        # Ensure directories exist
        ensure_directories()

        # Configuration
        bucket_name = "access-oap-prod-twilio-bucket"
        local_dir = "./output"

        # Initialize and start pipeline
        pipeline = S3ProcessingPipeline(bucket_name, download_dir=local_dir)
        pipeline.start_processing()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
