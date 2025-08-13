import argparse
from pyannote.audio import Pipeline
import torch
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from tqdm import tqdm
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diarization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_gpu_available(device_id):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if device_id >= torch.cuda.device_count():
        raise ValueError(f"Device {device_id} is not available")

parser = argparse.ArgumentParser(description="Diarization script")
parser.add_argument("--data_path", type=str, default='/home4/tuanlha/DataTest', help="Path to the data directory")
parser.add_argument("--save_path", type=str, default='/home4/tuanlha/DataProcessStep/01_diary', help="Path to the save directory")
#TODO: Change the default value of playlist_name to '00_diary'
parser.add_argument("--playlist_name", type=str, default='Temp', help="Name of the playlist")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use")
parser.add_argument("--device", type=str, default='3', help="GPU device ID")
args = parser.parse_args()

if args.playlist_name is not None:
    args.save_path = os.path.join(args.save_path, args.playlist_name)
    args.data_path = os.path.join(args.data_path, args.playlist_name)

# Check GPU
check_gpu_available(int(args.device))
logger.info(f"Using GPU {args.device}: {torch.cuda.get_device_name(int(args.device))}")

# Load model
logger.info("Loading diarization model...")
secret = "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"
diarization = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=secret).to(torch.device(f"cuda:{args.device}"))

def get_audio_files(directory):
    """Get valid audio files"""
    valid_extensions = ('.wav', '.mp3', '.flac', '.ogg')
    return [
        f for f in os.listdir(directory) 
        if f.lower().endswith(valid_extensions)
    ]

def get_diarized_episodes(folder_path):
    """Get already processed episodes"""
    return [file_name.split(' ')[0] for file_name in os.listdir(folder_path)]

def process_file(file_name):
    """Process single audio file"""
    episode = os.path.splitext(file_name)[0]
    
    if episode in diarized_episodes:
        logger.debug(f"Skipping already processed episode: {episode}")
        return

    logger.info(f"Processing {episode}")
    file_path = os.path.join(args.data_path, file_name)
    
    try:
        # Process audio
        diary = diarization(file_path)
        
        # Save with temp file to avoid corruption
        output_path = os.path.join(args.save_path, f"{episode}.pkl")
        temp_path = output_path + '.tmp'
        
        with open(temp_path, 'wb') as f:
            pickle.dump(diary, f)
        
        os.rename(temp_path, output_path)
        logger.info(f"Completed {episode}")
        
    except Exception as e:
        logger.error(f"Failed to process {file_name}: {str(e)}", exc_info=True)
    finally:
        torch.cuda.empty_cache()

def create_diarization():
    """Main processing function"""
    os.makedirs(args.save_path, exist_ok=True)
    global diarized_episodes
    diarized_episodes = get_diarized_episodes(args.save_path)
    
    files = get_audio_files(args.data_path)
    logger.info(f"Found {len(files)} audio files to process")
    
    # Process files with progress bar
    for file in files:
        process_file(file)
    # with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    #     list(tqdm(executor.map(process_file, files), total=len(files)))

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Starting diarization at {start_time}")
    
    try:
        create_diarization()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        raise
    finally:
        duration = datetime.now() - start_time
        logger.info(f"Completed in {duration}")