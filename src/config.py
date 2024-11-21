from dataclasses import dataclass

import torch


@dataclass
class TranscriptionConfig:
    """Configuration settings for transcription."""

    model_name: str = "openai/whisper-large-v3-turbo"
    chunk_length: int = 30
    batch_size: int = 16
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16
    supported_formats: tuple = ".webm"
    output_dir: str = "transcriptions"
    log_level: str = "INFO"
