from aws_bucket import S3Downloader
from transcription import AudioTranscriber
from config import TranscriptionConfig
import logging

def setup_main_logging():
    """Setup logging for the main execution script."""
    logger = logging.getLogger('main_execution')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler('logs/main_execution.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def main():
    logger = setup_main_logging()
    logger.info("Starting the application")

    try:
        # Configuration for both downloader and transcriber
        local_dir = "/home/mehmaam/Desktop/serefin_bot/Serefin-Xloop-main/audio_files"  # Ensure this path is accessible and writable
        bucket_name = "your-s3-bucket-name"

        # Create custom config for transcriber
        config = TranscriptionConfig(
            model_name="openai/whisper-large-v3-turbo",
            chunk_length=30,
            batch_size=16,
            output_dir=local_dir,  # Use the same directory for output
        )

        # Initialize downloader and transcriber
        # downloader = S3Downloader(bucket_name)
        transcriber = AudioTranscriber(config)

        # Start the download process in a separate thread
        # from threading import Thread
        # download_thread = Thread(target=downloader.start_downloading, args=(local_dir,))
        # download_thread.start()

        # # Wait for some files to be downloaded before starting transcription
        # download_thread.join()

        # Process the directory with downloaded files
        transcriber.process_directory(local_dir)

        logger.info("Application finished successfully")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
