import logging
import time
from pathlib import Path
from aws_bucket import S3ProcessingPipeline
from transcription import AudioTranscriber
from config import TranscriptionConfig
import threading


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
    transcriber: AudioTranscriber,
    local_dir: str,
    logger: logging.Logger,
    pipeline: S3ProcessingPipeline,
    stats: dict  # Pass stats as a parameter
):
    """Process files with download and transcription"""
    print(f"\n🔍 Monitoring directory: {Path(local_dir).resolve()}\n")
    
    stats_logger = setup_stats_logger()
    stats_logger.info("=== Starting New Processing Session ===")
    
    try:
        output_dir = Path(local_dir).resolve()
        files = list(output_dir.rglob("*.webm"))
        
        if files:
            print(f"\n📁 Found {len(files)} files to process")
            
            for file_path in files:
                stats['total_processed'] += 1
                s3_key = str(file_path.relative_to(output_dir))
                
                result = transcriber.transcribe_audio(file_path)
                if result:
                    transcriber._save_transcription(file_path, result)
                    stats['successful'] += 1
                    
                  
                    
                    # # Local cleanup
                    # file_path.unlink()
                    
                    # Log success with stats
                    stats_msg = (
                        f"SUCCESS - Total: {stats['total_processed']}, "
                        f"Success: {stats['successful']}, "
                        f"Failed: {stats['failed']}"
                    )
                    stats_logger.info(stats_msg)
                    print(f"\n📊 {stats_msg}")
                    print(f"✅ Processed and removed: {s3_key}")
                else:
                    stats['failed'] += 1
                    # Log failure with stats
                    stats_msg = (
                        f"FAILED - Total: {stats['total_processed']}, "
                        f"Success: {stats['successful']}, "
                        f"Failed: {stats['failed']}"
                    )
                    stats_logger.info(stats_msg)
                    print(f"\n📊 {stats_msg}")
                    print(f"❌ Failed to process: {s3_key}")
                
    except Exception as e:
        logger.error(f"Error in processing loop: {str(e)}")


def setup_stats_logger():
    """Setup statistics logger"""
    stats_logger = logging.getLogger("transcription_stats")
    stats_logger.setLevel(logging.INFO)
    
    if not stats_logger.handlers:
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        # File handler for stats
        fh = logging.FileHandler("logs/statistics.log")
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        fh.setFormatter(formatter)
        stats_logger.addHandler(fh)
    
    return stats_logger


def main():
    logger = setup_logging()
    logger.info("Starting the application")

    try:
        ensure_directories()
        bucket_name = "access-oap-prod-twilio-bucket"
        local_dir = "./output"

        config = TranscriptionConfig()
        transcriber = AudioTranscriber(config)
        pipeline = S3ProcessingPipeline(bucket_name, download_dir=local_dir)
        
        # Initialize stats outside the loop to maintain counts
        stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0
        }
        
        last_processing_time = 0
        processing_interval = 30  # seconds between processing batches
        
        while True:
            current_time = time.time()
            
            # Check if enough time has passed since last processing
            if current_time - last_processing_time >= processing_interval:
                files_in_output = list(Path(local_dir).rglob("*.webm"))
                files_count = len(files_in_output)
                
                if files_count < 5:  # Only download new files if we're running low
                    logger.info("Initiating new downloads")
                    pipeline.start_processing()
                
                if files_in_output:
                    logger.info(f"Processing batch of {files_count} files")
                    process_files(transcriber, local_dir, logger, pipeline, stats)
                    last_processing_time = current_time
                
            # Short sleep to prevent CPU spinning
            time.sleep(1)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()

