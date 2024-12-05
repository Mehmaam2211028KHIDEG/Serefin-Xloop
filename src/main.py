import logging
import sys
from pathlib import Path

from aws_bucket import S3ProcessingPipeline

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
        path = Path(directory).resolve()
        path.mkdir(exist_ok=True)

def main():
    logger = setup_logging()
    logger.info("=== Application Starting ===")
    
    try:
        # Ensure directories exist
        logger.info("Creating required directories...")
        ensure_directories()
        logger.info("Directories created successfully")

        # Configuration
        local_dir = "./output"
        bucket_name = "access-oap-prod-twilio-bucket"
        logger.info(f"Configuration - Bucket: {bucket_name}, Local dir: {local_dir}")
        
        # Initialize and start the pipeline
        logger.info("Initializing S3 Processing Pipeline...")
        pipeline = S3ProcessingPipeline(
            bucket_name=bucket_name,
            download_dir=local_dir,
            max_concurrent=2
        )
        
        # Start processing
        logger.info("Starting pipeline processing...")
        pipeline.start_processing()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("=== Application shutdown complete ===")

if __name__ == "__main__":
    main()

