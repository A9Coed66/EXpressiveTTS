import os
import pickle
import torch
import json
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('remove_collision.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_gpu_available(device_id):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if device_id >= torch.cuda.device_count():
        raise ValueError(f"Device {device_id} is not available")

parser = argparse.ArgumentParser(description="Remove collision script")
parser.add_argument("--data_path", type=str, default='/home4/tuanlha/DataProcessStep/01_diary', 
                    help="Path to the data directory containing diary files")
parser.add_argument("--save_path", type=str, default='/home4/tuanlha/DataProcessStep/02_no_collision',
                    help="Path to the save directory for processed logs")
parser.add_argument("--device", type=str, default='0',
                    help="GPU device ID")
parser.add_argument("--playlist_name", type=str, default='Temp',
                    help="Name of the playlist")
parser.add_argument("--num_workers", type=int, default=4,
                    help="Number of workers to use")
parser.add_argument("--min_segment_length", type=float, default=0.3,
                    help="Minimum segment length to keep")
args = parser.parse_args()

# Check GPU
check_gpu_available(int(args.device))
logger.info(f"Using GPU {args.device}: {torch.cuda.get_device_name(int(args.device))}")

def get_processed_episodes(folder_path):
    """Get set of already processed episodes from plalist folder"""
    return [f.split(' ')[0] for f in os.listdir(folder_path) if f.endswith('.json')]

def create_origin_logs(diarization):
    """Convert diarization object to simplified log format"""
    return [
        [round(turn.start, 2), round(turn.end, 2), speaker]
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

def process_episode(file_name, data_path, save_path, playlist_name):
    """Process a single episode file"""
    episode = os.path.splitext(file_name)[0]
    file_path = os.path.join(data_path, playlist_name, file_name)
    
    try:
        # Load diary data
        with open(file_path, 'rb') as f:
            diary = pickle.load(f)
        
        origin_log = create_origin_logs(diary)
        processed_log = []
        
        # Skip if empty log
        if not origin_log:
            return episode, processed_log
        
        # Process log segments
        last_end = 0
        current_segment = []
        
        for i in range(len(origin_log)-1):
            start, end, speaker = origin_log[i]
            
            if start > last_end:
                if current_segment:  # Finalize current segment
                    processed_log.append(current_segment)
                    current_segment = []
                #NOTE: Only keep segments longer than 0.3 second
                if end - start > args.min_segment_length:  # Only keep segments longer than 0.3 second
                    current_segment = [[start, end], speaker]
                
                last_end = end
            else:
                last_end = max(last_end, end)
                current_segment = []  # Discard overlapping segments
        
        # Process last segment
        start, end, speaker = origin_log[-1]
        if start > last_end  and end - start > args.min_segment_length:
            processed_log.append([[start, last_end], speaker])
        
        return file_name, processed_log
    
    except Exception as e:
        logger.error(f"Error processing {file_name}: {str(e)}", exc_info=True)
        raise

def save_processed_log(episode, log, save_path):
    """Save processed log to JSON file"""
    output_path = os.path.join(save_path, f"{os.path.splitext(episode)[0]}.json")
    
    with open(output_path, 'w') as f:
        json.dump(log, f)

def remove_collisions():
    """Main function to process all episodes"""
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.playlist_name), exist_ok=True)

    processed_episode_path = os.path.join(args.save_path, args.playlist_name)
    
    processed_episodes = get_processed_episodes(processed_episode_path)
    print(f"Processed episodes: {processed_episodes}")
    diary_files = [f for f in os.listdir(os.path.join(args.data_path, args.playlist_name)) 
                 if f.endswith('.pkl')]
    
    logger.info(f"Found {len(diary_files)} diary files to process")
    logger.info(f"{len(processed_episodes)} episodes already processed")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for file_name in diary_files:
            episode = file_name.split(' ')[0]
            
            if episode in processed_episodes:
                logger.debug(f"Skipping already processed episode: {episode}")
                continue
            
            futures.append(executor.submit(process_episode, file_name, args.data_path, args.save_path, args.playlist_name))
        
        for future in tqdm(futures, desc="Processing episodes"):
            try:
                episode, processed_log = future.result()
                save_processed_log(episode, processed_log, os.path.join(args.save_path, args.playlist_name))
                logger.info(f"Completed {episode}")
            except Exception as e:
                logger.error(f"Failed to process episode: {str(e)}")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Starting collision removal at {start_time}")
    
    try:
        remove_collisions()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        raise
    finally:
        duration = datetime.now() - start_time
        logger.info(f"Completed in {duration}")