import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
import resampy
import soundfile as sf
import contextlib
import wave


def prepare_align(config):
    print("Start preparing align...")
    in_dir  = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    for speaker in tqdm(os.listdir(in_dir)):
        for file_name in os.listdir(os.path.join(in_dir, speaker)):
            if file_name[-4:] != ".wav":
                continue

            file_path = os.path.join(in_dir, speaker, file_name)
            with contextlib.closing(wave.open(file_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                if duration < 1.0 or duration > 20.0:
                    continue
            
            base_name = file_name[:-4]
            text_path = os.path.join(in_dir, speaker, "{}.lab".format(base_name))
            wav_path  = os.path.join(in_dir, speaker, "{}.wav".format(base_name))

            with open(text_path) as f:
                text = f.readline().strip("\n")
            text = _clean_text(text, cleaners)

            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            wav, fs = sf.read(wav_path)
            if fs != sampling_rate:
                wav = resampy.resample(x=wav, sr_orig=fs, sr_new=sampling_rate, axis=0)
            
            # Check if audio duration is less than 1 second
            duration = len(wav) / sampling_rate
            if duration < 1.0:
                # print(f"Skipping {wav_path} because duration is less than 1 second.")
                continue

            wav = wav / np.max(np.abs(wav))
            sf.write(os.path.join(out_dir, speaker, "{}.wav".format(base_name)), wav, sampling_rate)        
                
            with open(
                os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                "w",
            ) as f1:
                f1.write(text)