import json
import pandas as pd
import os
import librosa
import soundfile as sf
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('concat_audio.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Remove collision script")
parser.add_argument("--source_audio_path", type=str, default='/home4/tuanlha/DataTest', help="Path to the source audio directory")
parser.add_argument("--data_path", type=str, default='/home4/tuanlha/DataProcessStep/02_no_collision', help="Path to the data directory")
parser.add_argument("--save_path", type=str, default=f'/home4/tuanlha/DataProcessStep/03_concat_audio', help="Path to the save directory")
parser.add_argument("--min_segment_length", type=float, default=0.5, help="Length of segment to keep")
parser.add_argument("--min_length", type=float, default=0.5, help="Length of segment to keep")
parser.add_argument("--playlist_name", type=str, default='Temp', help="Name of the playlist")
args = parser.parse_args()

def get_processed_episodes(folder_path):
    """Get set of already processed episodes from plalist folder"""
    return [f.split(' ')[0] for f in os.listdir(folder_path)]

def check_exist(episode, folder):
    '''Check if diary logs already exist
    exist:      return True
    not exist:  return False'''
    return episode in [dir for dir in os.listdir(folder)]

def create_audio(audio_path, episode, speaker, q):
    save_path = os.path.join(args.save_path, args.playlist_name, os.path.splitext(episode)[0])
    os.makedirs(save_path, exist_ok=True)
    start, end = q[0], q[-1]
    if end - start < args.min_length:
        return
    y, sr = librosa.load(audio_path, offset=start, duration=end - start)
    sf.write(os.path.join(save_path, f'{speaker} - {round(start,2)} - {round(end,2)}.wav'), y, sr)

def process_episode(dir_n, processed_episodes):
    episode = dir_n.split(' ')[0]
    if episode in processed_episodes:
        logger.info(f"Episode {episode} already processed")
        return

    logger.info(f"Processing episode {episode}")
    print(f"Bat dau: {episode}")

    file_path = os.path.join(args.data_path, args.playlist_name, dir_n)
    with open(file_path, 'rb') as f:
        data = json.load(f)

    audio_path = os.path.join(args.source_audio_path, f'{os.path.splitext(dir_n)[0]}.mp3')
    if not os.path.exists(audio_path):
        print(f"Thieu {audio_path}")
        return

    q = []
    current_speaker = None
    for item in data:
        start, end, speaker = item[0][0], item[0][1], item[1]
        if not q:
            q = [start, end]
            current_speaker = speaker
        elif speaker == current_speaker:
            if start - q[-1] < args.min_segment_length:
                q[-1] = end
            else:
                create_audio(audio_path, dir_n, current_speaker, q)
                q = [start, end]
        else:
            create_audio(audio_path, dir_n, current_speaker, q)
            q = [start, end]
            current_speaker = speaker
    if q:
        create_audio(audio_path, dir_n, current_speaker, q)

def concat_audio_unverified():
    data_path = os.path.join(args.data_path, args.playlist_name)
    save_folder_path = os.path.join(args.save_path, args.playlist_name)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(save_folder_path, exist_ok=True)

    processed_episodes = get_processed_episodes(save_folder_path)
    no_collision_log = [f for f in os.listdir(data_path) if f.endswith('.json')]
    logger.info(f"Found {len(no_collision_log)} diary files to process")
    logger.info(f"{len(processed_episodes)} episodes already processed")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_episode, dir_n, processed_episodes) for dir_n in no_collision_log]
        for future in as_completed(futures):
            future.result()

concat_audio_unverified()
