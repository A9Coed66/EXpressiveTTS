import argparse
from remove_collision import remove_collision
from pyannote.audio import Pipeline
import torch
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

def create_diarization():
    # apply pretrained pipeline
    diarization = diarization(args["path"])
    return remove_collision(diarization)

