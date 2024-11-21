import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available


@dataclass
class TranscriptionConfig:
    """Configuration settings for transcription."""

    model_name: str = "openai/whisper-large-v3-turbo"
    chunk_length: int = 30
    batch_size: int = 16
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16
    supported_formats: tuple = (".webm", ".mp3", ".wav", ".m4a", ".ogg")


class AudioTranscriber:
    def __init__(self, config: TranscriptionConfig = None):
        """Initialize the transcriber with given configuration."""
        self.config = config or TranscriptionConfig()
        self.logger = self._setup_logging()
        self.model = self._initialize_model()

    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            filename="transcription.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return logging.getLogger(__name__)

    def _initialize_model(self) -> pipeline:
        """Initialize the transcription model."""
        try:
            start_time = time.time()
            model = pipeline(
                "automatic-speech-recognition",
                model=self.config.model_name,
                torch_dtype=self.config.torch_dtype,
                device=self.config.device,
                model_kwargs={
                    "attn_implementation": (~
                        "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
                    )
                },
            )
            loading_time = time.time() - start_time
            self.logger.info(f"Model loaded in {loading_time:.2f} seconds")
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def transcribe_audio(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Transcribe audio file and return the text.

        Args:
            file_path: Path to the audio file

        Returns:
            Transcribed text or None if transcription fails
        """
        file_path = Path(file_path)
        if not self._validate_file(file_path):
            return None

        try:
            start_time = time.time()
            result = self.model(
                str(file_path),
                chunk_length_s=self.config.chunk_length,
                batch_size=self.config.batch_size,
                return_timestamps=True,
            )
            transcription_time = time.time() - start_time

            self.logger.info(
                f"Transcribed {file_path.name} in {transcription_time:.2f} seconds"
            )

            return result["text"] if isinstance(result, dict) else result

        except Exception as e:
            self.logger.error(f"Transcription failed for {file_path}: {str(e)}")
            return None

    def _validate_file(self, file_path: Path) -> bool:
        """Validate if the file exists and has supported format."""
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False

        if file_path.suffix.lower() not in self.config.supported_formats:
            self.logger.error(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {self.config.supported_formats}"
            )
            return False

        return True

    def process_directory(self, directory_path: Union[str, Path]) -> None:
        """Process all supported audio files in a directory."""
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            self.logger.error(f"Invalid directory path: {directory_path}")
            return

        for file_path in directory_path.glob("*"):
            if file_path.suffix.lower() in self.config.supported_formats:
                text = self.transcribe_audio(file_path)
                if text:
                    output_path = file_path.with_suffix(".txt")
                    self._save_transcription(text, output_path)

    def _save_transcription(self, text: str, output_path: Path) -> None:
        """Save transcription to a file."""
        try:
            output_path.write_text(text, encoding="utf-8")
            self.logger.info(f"Transcription saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save transcription: {str(e)}")


def main():
    # Example usage
    config = TranscriptionConfig(
        chunk_length=30,
        batch_size=16,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    transcriber = AudioTranscriber(config)

    # Process single file
    result = transcriber.transcribe_audio("/content/composed_final.mp3")
    if result:
        print(f"Transcription: {result}")

    # Or process entire directory
    # transcriber.process_directory("./")


if __name__ == "__main__":
    main()
