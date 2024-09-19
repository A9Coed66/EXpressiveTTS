"""
Dataframe:
	audio_path	    diary_label	    model_label	    score	    verified	        start
30	/kaggle/...	    SPEAKER_00	    DucAnh	        0.220918	[tensor(False)]	    7.0     """

"""
Concat near audio files to one file
Condition:
    1. Must have same speaker
    2. Must have a distance less than 0.5"""

import os
import pandas as pd
import librosa
import soundfile as sf
# import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-df", "--data_frame", type=str, default=0,
# 	help="path of data frame")
# ap.add_argument("-ap", "--audio_path", type=str, default=0,
#     help="path of audio file")
# args = vars(ap.parse_args())


def create_audio(audio_path, folder_name, q, current_speaker):
    start, end = q[0], q[-1]
    y, sr = librosa.load(audio_path)
    segment = y[int(start*sr):int(end*sr)]
    sf.write(f'concat_audio/{folder_name}/{round(start,1)} to {round(end,1) }.wav', segment, sr)

def concat_audio(folder_name):
    df = pd.DataFrame(columns=['audio_path', 'model_label', 'score', 'start', 'end'])
    for dir in os.listdir(f'verified_speaker/{folder_name}'):    #chunk_1, chunk_2,...
        dir_path = os.path.join(f'verified_speaker/{folder_name}', dir)
        audio_path = os.path.join(f'denoised/{folder_name}', dir)
        df = pd.read_csv(dir_path)
        q = []
        current_speaker = None
        for _, row in df.iterrows():
            if row['verified']=='True':
                if not q:
                    q = [float(row['start']), float(row['end'])]
                    current_speaker = row['speaker']

                if row['speaker'] == current_speaker:   
                    if row['start'] - q[-1][1] < 0.5: # neu khoang cach 2 wav nho va cung nguoi nois
                        q[-1] = row['end']
                    else:
                        create_audio(audio_path, folder_name, q)
                        q = [float(row['start']), float(row['end'])]
                else:                               # neu la nguoi khac
                    create_audio(audio_path, folder_name, q)
                    q = [[row['start'], row['end']]]
                    current_speaker = row['speaker']
            else:
                if q:
                    create_audio(audio_path, folder_name, q, current_speaker)
                    q = []
                    current_speaker = None
    return