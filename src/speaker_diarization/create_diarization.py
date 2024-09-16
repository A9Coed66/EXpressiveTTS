import argparse
from remove_collision import remove_collision
from pyannote.audio import Pipeline
import torch
import os
import json

print("Loading diarization model...")
diarization = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token='khong cho a nha')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path",
	help="audio file path")
args = vars(ap.parse_args())

diarization.to(torch.device("cuda"))

def create_diarization(folder_name):
    # apply pretrained pipeline
    for file_name in os.listdir(f'denoised/{folder_name}'):
        file_path = os.path.join(f'denoised/{folder_name}', file_name)
        diary = diarization(file_path)
        with open(f'diary/{folder_name}/logs_{file_name[:-4]}.json', 'w') as f:
            json.dump(diary, f)
        return 

