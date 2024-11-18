import logging
import os
import whisper
from pydub import AudioSegment

# Setup logging
def setup_logging():
    logging.basicConfig(filename='app.log', level=logging.DEBUG, filemode='a', 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

# Load the model once to optimize performance
model = whisper.load_model("turbo")

def check_files_in_directory(base_path):
    """Check for .webm files in the directory and process them."""
    logger.warning("Starting Directory Check")
    processed_files = set()
    for root, _, files in os.walk(base_path):
        for file_name in files:
            if file_name.endswith('.webm'):
                process_file(file_name, os.path.join(root, file_name), processed_files)

def process_file(file_name, file_path, processed_files):
    """Process each .webm file: convert to .wav, transcribe, and save output."""
    logger.warning("Starting File Processing")
    try:
        txt_filename = create_unique_filename(file_name)
        wav_file = webm_to_wav(file_path)
        whisper_text = transcribe_audio(wav_file)
        if whisper_text:
            save_transcription(whisper_text, txt_filename)
        cleanup_files(file_path, wav_file)
        processed_files.add(file_name)
    except Exception as e:
        logger.critical(f"Error processing file {file_path}: {e}")

def create_unique_filename(file_name):
    """Generate a unique filename for the transcription text file."""
    base_txt_filename = os.path.splitext(file_name)[0]
    txt_filename = f"{base_txt_filename}.txt"
    counter = 1
    while os.path.exists(txt_filename):
        txt_filename = f"{base_txt_filename}_{counter}.txt"
        counter += 1
    return txt_filename

def webm_to_wav(file_path):
    """Convert .webm file to .wav format."""
    logger.warning(f"Converting {file_path} to WAV format")
    try:
        audio = AudioSegment.from_file(file_path, format="webm")
        wav_file_path = file_path.replace(".webm", ".wav")
        audio.export(wav_file_path, format="wav")
        logger.info(f"Conversion to WAV complete: {wav_file_path}")
        return wav_file_path
    except Exception as e:
        logger.error(f"Error converting {file_path} to WAV: {e}")
        raise

def transcribe_audio(file_path):
    """Transcribe audio using Whisper model."""
    logger.warning("Starting Transcription")
    if os.path.exists(file_path):
        result = model.transcribe(file_path)
        logger.info("Transcription successful")
        return result['text']
    else:
        logger.error("WAV file for transcription not found.")
        return ""

def save_transcription(text, filename):
    """Save the transcription text to a file."""
    with open(filename, 'w') as file:
        file.write(text)
    logger.info(f"Transcription saved to {filename}")

def cleanup_files(*files):
    """Remove processed files after use."""
    for file in files:
        os.remove(file)
        logger.info(f"File deleted: {file}")

base_path = '''Add File Path'''
check_files_in_directory(base_path)
