import logging
import time
from pathlib import Path
from typing import Optional, Union
import json

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
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            # File handler
            fh = logging.FileHandler("logs/transcription.log")
            fh_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            fh.setFormatter(fh_formatter)
            logger.addHandler(fh)
            
            # Console handler
            ch = logging.StreamHandler()
            ch_formatter = logging.Formatter("%(levelname)s - %(message)s")
            ch.setFormatter(ch_formatter)
            logger.addHandler(ch)

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

        try:
            # Get just the folder structure for logging
            try:
                # Find 'output' in the path parts and get everything after it
                path_parts = file_path.parts
                output_index = path_parts.index('output')
                relative_parts = path_parts[output_index + 1:]
                folder_structure = "/".join(relative_parts[:-1])
            except ValueError:
                # If 'output' not found in path, use the last two directory names
                folder_structure = "/".join(file_path.parts[-3:-1])
            
            self.logger.info(f"� Processing: {folder_structure}")
            
            result = self.model(
                str(file_path),
                chunk_length_s=self.config.chunk_length,
                batch_size=self.config.batch_size,
                return_timestamps=True,
 
            )
            
            self.logger.info(f"✅ Completed: {folder_structure}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error: {str(e)}")
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
        """Process all audio files in a directory."""
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            self.logger.error(f"Directory not found: {directory_path}")
            return

        for file_path in directory_path.glob("*"):
            if file_path.suffix.lower() in self.config.supported_formats:
                result = self.transcribe_audio(file_path)
                if result:
                    self._save_transcription(file_path, result)
                    self.logger.info(f"Successfully processed: {file_path.name}")

    def _save_transcription(self, audio_path: Path, result: dict) -> None:
        """Save complete transcription output to JSON file."""
        try:
            base_output_dir = Path(self.config.output_dir)
            
            # Get the parts of the path after 'output'
            path_parts = audio_path.parts
            output_index = path_parts.index('output')
            relative_parts = path_parts[output_index + 1:]
            
            # Create the full output path maintaining directory structure
            output_dir = base_output_dir.joinpath(*relative_parts[:-1])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create two output files: one for JSON (full output) and one for text (just transcription)
            json_path = output_dir / f"{audio_path.stem}.json"
            text_path = output_dir / f"{audio_path.stem}.txt"
            
            # Save complete output as JSON
            with json_path.open('w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            # Also save just the text for convenience
            text_path.write_text(result.get('text', '').strip())
            
            self.logger.info(f"Saved full transcription to {json_path}")
            self.logger.info(f"Saved text transcription to {text_path}")

        except Exception as e:
            self.logger.error(f"Error saving transcription: {str(e)}")
