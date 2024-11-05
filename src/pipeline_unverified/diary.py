import argparse
from pyannote.audio import Pipeline
import torch
import os
import pickle

secret = "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"

print("Loading diarization model...")
diarization = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=secret)


diarization.to(torch.device("cuda"))

def check_exist(episode, folder):
    '''Check if diary logs already exist
    exist:      return True
    not exist:  return False'''
    current_episode = [file_name for file_name in os.listdir(folder)]
    for ep in current_episode:
        if episode in ep:
            return True
    return False

def create_diarization():
    # apply pretrained pipeline
    data_path = '/home/tuannd/tuanlha/Dataset/HaveASip'
    save_path = f'/home/tuannd/tuanlha/EXpressiveTTS/data_unverified/diary'


    os.makedirs(save_path, exist_ok=True)

    for file_name in os.listdir(f'{data_path}'):
        episode = file_name.split(' ')[0]

        if check_exist(episode, save_path):    # Skip if diary logs already exist
            print(f"Skip {episode}")
            continue

        file_path = os.path.join(f'{data_path}', file_name)
        diary = diarization(file_path)
        with open(f'{save_path}/logs_{file_name[:-4]}.pkl', 'wb') as f:
            pickle.dump(diary, f)
        print(f"Finish {episode}")
create_diarization()         # Step 3