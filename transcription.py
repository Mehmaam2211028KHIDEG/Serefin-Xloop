import logging
import time
from pathlib import Path
from typing import Optional, Union

from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

from config import TranscriptionConfig


class AudioTranscriber:
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self.logger = self._setup_logging()
        self.model = self._initialize_model()

    def _setup_logging(self) -> logging.Logger:
        """Setup detailed logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of log messages
        if not logger.handlers:
            fh = logging.FileHandler('logs/transcription.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        return logger

    def _initialize_model(self):
        start_time = time.time()
        try:
            model_kwargs = {"torch_dtype": self.config.torch_dtype}
            if is_flash_attn_2_available():
                model_kwargs["use_flash_attention2"] = True

            model = pipeline(
                "automatic-speech-recognition",
                model=self.config.model_name,
                device=self.config.device,
                **model_kwargs,
            )
            loading_time = time.time() - start_time
            self.logger.info(f"Model loaded in {loading_time:.2f} seconds")
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def transcribe_audio(self, file_path: Union[str, Path]) -> Optional[str]:
        """Transcribe audio file to text."""
        file_path = Path(file_path)
        if not self._validate_file(file_path):
            return None

        start_time = time.time()
        try:
            result = self.model(
                str(file_path),
                chunk_length_s=self.config.chunk_length,
                batch_size=self.config.batch_size,
            )
            transcription_time = time.time() - start_time
            self.logger.info(
                f"Transcribed {file_path.name} in {transcription_time:.2f} seconds"
            )
            return result["text"].strip()
        except Exception as e:
            self.logger.error(f"Error transcribing {file_path}: {str(e)}")
            return None

    def _validate_file(self, file_path: Path) -> bool:
        """Validate audio file."""
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False

        if file_path.suffix.lower() not in self.config.supported_formats:
            formats = ", ".join(self.config.supported_formats)
            self.logger.error(
                f"Unsupported format: {file_path.suffix}. " f"Supported: {formats}"
            )
            return False
        return True

    def process_directory(self, directory_path: Union[str, Path]) -> None:
        """Process all audio files in a directory, check daily if no files are found."""
        directory_path = Path(directory_path)
        while True:
            if not directory_path.is_dir():
                self.logger.error(f"Directory not found: {directory_path}")
                return

            audio_files_found = False
            for file_path in directory_path.glob("*.webm"):  # Ensure it only looks for .webm files
                if file_path.suffix.lower() in self.config.supported_formats:
                    result = self.transcribe_audio(file_path)
                    if result:
                        self._save_transcription(file_path, result)
                        audio_files_found = True

            if not audio_files_found:
                self.logger.info("No audio files found. Checking again in 24 hours.")
                time.sleep(86400)  # Sleep for 24 hours before checking again
            else:
                self.logger.info("Audio files processed. Continuing to monitor the directory.")

    def _save_transcription(self, audio_path: Path, text: str) -> None:
        """Save transcription to file in a separate directory outside of src."""
        try:
            output_dir = Path("/home/mehmaam/Desktop/serefin_bot/Serefin-Xloop-main/transcriptions")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{audio_path.stem}.txt"
            self.logger.info(f"Saving transcription to {output_path}")
            output_path.write_text(text)
            
            # Uncomment the following lines to enable deletion of the original .webm file after transcription
            # if audio_path.exists():
            #     audio_path.unlink()
            #     self.logger.info(f"Deleted original audio file: {audio_path}")

        except Exception as e:
            self.logger.error(f"Error saving transcription: {str(e)}")
