import argparse
from pyannote.audio import Pipeline
import torch
import os
import pickle


parser = argparse.ArgumentParser(description="Diarization script")
parser.add_argument("--data_path", type=str, default='/home2/tuannd/tuanlha/HaveASipDataset', help="Path to the data directory")
parser.add_argument("--save_path", type=str, default='/home2/tuannd/tuanlha/EXpressiveTTSDataset/data_unverified/00_diary', help="Path to the save directory")
args = parser.parse_args()

secret = "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"

print("Loading diarization model...")
diarization = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=secret)
    

diarization.to(torch.device("cuda"))

def get_dirized_episode(folder_path):
    return [file_name.split(' ')[0] for file_name in os.listdir(folder_path)]

def create_diarization():
    # apply pretrained pipeline
    data_path = args.data_path
    save_path = args.save_path


    os.makedirs(save_path, exist_ok=True)
    diarizated_episodes = get_dirized_episode(save_path)
    print(f'Diarized episodes: {diarizated_episodes}')

    for file_name in os.listdir(f'{data_path}'):
        episode = file_name.split(' ')[0]

        # Skip episode nếu đã duyệt 
        if episode in diarizated_episodes:    
            print(f"Skip {episode}")
            continue
        print(f"Start {episode}")

        file_path = os.path.join(f'{data_path}', file_name)
        diary = diarization(file_path)
        with open(f'{save_path}/{file_name[:-4]}.pkl', 'wb') as f:
            pickle.dump(diary, f)
        print(f"Finish {episode}")
create_diarization()         # Step 3