import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
import resampy
import soundfile as sf


def prepare_align(config):
    print("Start preparing align...")
    in_dir  = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    with open (os.path.join(in_dir, "text.txt")) as f:
        lines = f.readlines()
        for line in lines:
            base_name, text = line.strip().split("|")
            wav_path = os.path.join(in_dir, 'wav_segment', base_name+".wav")
            wav, fs = sf.read(wav_path)
            if fs != sampling_rate:
                wav = resampy.resample(x=wav, sr_orig=fs, sr_new=sampling_rate, axis=0)
        
            # Check if audio duration is less than 1 second
            duration = len(wav) / sampling_rate
            if duration < 1.0:
                # print(f"Skipping {wav_path} because duration is less than 1 second.")
                continue

            wav = wav / np.max(np.abs(wav))
            sf.write(os.path.join(out_dir, "{}.wav".format(base_name)), wav, sampling_rate)        
                
            with open(
                os.path.join(out_dir, "{}.lab".format(base_name)),
                "w",
            ) as f1:
                f1.write(text)