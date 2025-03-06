import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="Remove collision script")
parser.add_argument("--data_path", type=str, default='/home2/tuannd/tuanlha/EXpressiveTTSDataset/data_unverified/03_transcription', help="Path to the data directory")
args = parser.parse_args()

def delete_file(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
    else:
        print(f"No such file: {file_path}")
    
def delete_speaker(speakers, episode_path):
    for wav_file in os.listdir(episode_path):
        if wav_file.split(' ')[0] in speakers:
            delete_file(os.path.join(episode_path, wav_file))
    return 0

def episode_filter(episode_path):
    speaker_count = {}
    for wav_file in os.listdir(episode_path):
        speaker = wav_file.split(' ')[0]
        if speaker in speaker_count:
            speaker_count[speaker] += 1
        else:
            speaker_count[speaker] = 1
    
    total_files = len(os.listdir(episode_path))
    threshold = total_files * 0.05
    
    speaker_del = []

    for speaker, count in speaker_count.items():
        if count < threshold:
            speaker_del.append(speaker)
            print(f'Remove {speaker} with {count} files {episode_path}')
    delete_speaker(speaker_del, episode_path)

    return 0

def speaker_filter():
    data_path = args.data_path
    for episode in os.listdir(data_path):
        episode_path = os.path.join(data_path, episode)
        episode_filter(episode_path)
    return 0

speaker_filter()