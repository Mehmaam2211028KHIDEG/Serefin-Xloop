# import logging
# import os
# import whisper
# from pydub import AudioSegment
# import numpy as np
# from scipy.io import wavfile

# logging.basicConfig(filename='app.log', level=logging.DEBUG, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def check_files_in_directory(base_path):
#     logging.warning("=============check_files_in_directory_function=================")
#     for root, dirs, files in os.walk(base_path):
#         logger.debug(f"Checking folder: {root}")
#         for file_name_webm in files:
#             if file_name_webm.endswith('.webm'):
#                 logger.debug(".webm file found")
#                 file_path = os.path.join(root, file_name_webm)
#                 process_file(file_name_webm, file_path)

# def process_file(file_name_webm, file_path):
#     logging.warning("=============process_file_function=================")
#     try:
#         txt_filename = os.path.splitext(file_name_webm)[0] + ".txt"
#         logger.debug(f"Start Processing file: {file_path}")
        
#         # Convert .webm to .wav
#         conversion_file = webm_to_wav(file_name_webm, file_path)
        
#         # Transcribe with Whisper
#         whisper_text = whisper_model(conversion_file)
        
#         # Write transcribed text to a file
#         with open(txt_filename, 'w') as file:  # changed 'rb+' to 'w'
#             file.write(whisper_text)
#         logging.info(f"Result saved to {txt_filename}")

#         # Delete original file after processing
#         os.remove(file_path)
#         logger.warning(f"Deleted file: {file_path}")

#     except Exception as e:
#         logger.critical(f"Error processing file {file_path}: {e}")

# def whisper_model(conversion_file):
#     logging.warning("=============whisper_model_function=================")
#     model_turbo = whisper.load_model("turbo")  # Changed model to "tiny" for faster processing, adjust if needed
    
#     if os.path.exists(conversion_file):
#         result_turbo = model_turbo.transcribe(conversion_file)
#         logging.info("Return Turbo!")
#         return result_turbo['text']
#     else:
#         logger.error("Conversion file not found.")
#         return ""

# def webm_to_wav(file_name_webm, file_path):
#     logging.warning("=============webm_to_wav_function=================")
#     logging.warning(f"Entering whisper model conversion, {file_path}")
    
#     # Load the WebM file
#     audio = AudioSegment.from_file(file_path, format="webm")
#     logging.warning(f"Audio loaded successfully, {audio}")

#     # Export the audio to a WAV file and save path
#     wav_file_path = "output_file.wav"
#     audio.export(wav_file_path, format="wav")
#     logging.info("Conversion to WAV complete!")
    
#     return wav_file_path  # Return the path of the WAV file

# # Specify the base path you want to check
# base_path = '/home/mehmaam/Desktop/Gen_AI_Serefin/finalize_code/code/003Am000004isUXIAY (copy)'
# check_files_in_directory(base_path)



import logging
import os
import whisper
from pydub import AudioSegment

import warnings

# # Suppress specific warnings
# warnings.filterwarnings("ignore", category=UserWarning)   # Ignores UserWarnings (NumPy version & FP16 warnings)
# warnings.filterwarnings("ignore", category=FutureWarning) # Ignores FutureWarnings (pickle warnings)
# warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")



# Initialize logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, filemode='a', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the model once to avoid reloading it for every file
model = whisper.load_model("turbo")  # Adjust to "turbo" or other as needed for better performance

def check_files_in_directory(base_path):
    """Check for .webm files in the directory and process them."""
    logging.warning("=============Starting Directory Check=================")
    for root, _, files in os.walk(base_path):
        logger.debug(f"Checking folder: {root}")
        for file_name in files:
            if file_name.endswith('.webm'):
                logger.debug(".webm file found")
                file_path = os.path.join(root, file_name)
                process_file(file_name, file_path)

def process_file(file_name, file_path):
    """Process each .webm file: convert to .wav, transcribe, and save output."""
    logging.warning("=============Starting File Processing=================")
    try:
        txt_filename = os.path.splitext(file_name)[0] + ".txt"
        logger.debug(f"Processing file: {file_path}")
        
        # Convert .webm to .wav
        wav_file = webm_to_wav(file_path)
        
        # Transcribe audio to text
        whisper_text = transcribe_audio(wav_file)
        
        # Write transcribed text to a file
        if whisper_text:
            with open(txt_filename, 'w') as file:
                file.write(whisper_text)
            logging.info(f"Transcription saved to {txt_filename}")
        
        # Clean up files after processing
        os.remove(file_path)
        os.remove(wav_file)
        logger.info(f"Cleaned up files: {file_path} and {wav_file}")

    except Exception as e:
        logger.critical(f"Error processing file {file_path}: {e}")

def transcribe_audio(file_path):
    """Transcribe audio using Whisper model."""
    logging.warning("=============Starting Transcription=================")
    if os.path.exists(file_path):
        result = model.transcribe(file_path)
        logging.info("Transcription successful")
        return result['text']
    else:
        logger.error("WAV file for transcription not found.")
        return ""

def webm_to_wav(file_path):
    """Convert .webm file to .wav format."""
    logging.warning(f"Converting {file_path} to WAV format")
    try:
        audio = AudioSegment.from_file(file_path, format="webm")
        wav_file_path = file_path.replace(".webm", ".wav")
        audio.export(wav_file_path, format="wav")
        logging.info(f"Conversion to WAV complete: {wav_file_path}")
        return wav_file_path
    except Exception as e:
        logger.error(f"Error converting {file_path} to WAV: {e}")
        raise

# Specify the base path to check for files
base_path = '/home/mehmaam/Desktop/Gen_AI_Serefin/finalize_code/code/003Am000004isUXIAY (copy)'
check_files_in_directory(base_path)
