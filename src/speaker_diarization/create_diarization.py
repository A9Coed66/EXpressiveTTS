import argparse
from pyannote.audio import Pipeline
import torch
import os
import json

secret = "mo di cu"

print("Loading diarization model...")
diarization = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=secret)


diarization.to(torch.device("cuda"))

def create_diarization(folder_name):
    # apply pretrained pipeline
    data_path = '../../data'
    save_path = f'{data_path}/denoised/{folder_name}'


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file_name in os.listdir(f'{data_path}/denoised/{folder_name}'):
        file_path = os.path.join(f'{data_path}/denoised/{folder_name}', file_name)
        diary = diarization(file_path)
        with open(f'{save_path}/logs_{file_name[:-4]}.json', 'w') as f:
            json.dump(diary, f)
        return 

