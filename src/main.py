from config import TranscriptionConfig
from transcription import AudioTranscriber


def main():
    # Create custom config if needed
    config = TranscriptionConfig(
        model_name="openai/whisper-large-v3-turbo",
        chunk_length=30,
        batch_size=16,
        output_dir="transcriptions",
    )

    # Initialize transcriber
    transcriber = AudioTranscriber(config)

    # Example: Transcribe single file
    result = transcriber.transcribe_audio("path/to/audio.mp3")
    if result:
        print(f"Transcription: {result}")

    # Example: Process entire directory
    transcriber.process_directory("path/to/audio/files")


if __name__ == "__main__":
    main()
