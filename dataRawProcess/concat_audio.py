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
import pickle
import sys
# import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-df", "--data_frame", type=str, default=0,
# 	help="path of data frame")
# ap.add_argument("-ap", "--audio_path", type=str, default=0,
#     help="path of audio file")
# args = vars(ap.parse_args())


def create_audio(audio_path, folder_name, q):
    data_path = '../data'
    save_path = f'{data_path}/concat_audio/{folder_name}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    start, end = q[0], q[-1]
    print(start, end)
    y, sr = librosa.load(audio_path)
    segment = y[int(start*sr):int(end*sr)]
    sf.write(f'{save_path}/{round(start,1)} to {round(end,1) }.mp3', segment, sr)

def concat_audio_verified(folder_name):
    data_path = '../data'
    
    for dir_n in os.listdir(f'{data_path}/verified_speaker/{folder_name}'):    #chunk_1, chunk_2,...
        # dataframe from verified speaker
        dir_path = os.path.join(f'{data_path}/verified_speaker/{folder_name}', dir_n)
        
        # denoised audio
        audio_path = os.path.join(f'{data_path}/denoised/{folder_name}', f'{dir_n[:-4]}.mp3')
        
        # new df to get data audio after concat
        new_df = pd.DataFrame(columns=['audio_path', 'model_label', 'start', 'end'])
        df = pd.read_csv(dir_path)
        
        q = []
        current_speaker = None
        for _, row in df.iterrows():
            if row['verified']:
                if not q:
                    q = [float(row['start']), float(row['end'])]
                    current_speaker = row['model_label']

                elif row['model_label'] == current_speaker:   
                    if row['start'] - q[-1] < 1: # neu khoang cach 2 wav nho va cung nguoi nois
                        q[-1] = float(row['end'])
                    else:
                        new_row = {'audio_path': f'{data_path}/concat_audio/{folder_name}/{dir_n[:-4]}/{round(q[0],1)} to {round(q[-1],1) }.mp3', 'model_label': current_speaker, 'start': round(q[0],1), 'end': round(q[-1],1)}
                        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                        create_audio(audio_path, f'{folder_name}/{dir_n[:-4]}' , q)
                        q = [float(row['start']), float(row['end'])]
                else:                               # neu la nguoi khac
                    new_row = {'audio_path': f'{data_path}/concat_audio/{folder_name}/{dir_n[:-4]}/{round(q[0],1)} to {round(q[-1],1) }.mp3', 'model_label': current_speaker, 'start': round(q[0],1), 'end': round(q[-1],1)}
                    new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                    create_audio(audio_path, f'{folder_name}/{dir_n[:-4]}', q)
                    q = [float(row['start']), float(row['end'])]
                    current_speaker = row['model_label']
            else:
                if q:
                    new_row = {'audio_path': f'{data_path}/concat_audio/{folder_name}/{dir_n[:-4]}/{round(q[0],1)} to {round(q[-1],1) }.mp3', 'model_label': current_speaker, 'start': round(q[0],1), 'end': round(q[-1],1)}
                    new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                    create_audio(audio_path, f'{folder_name}/{dir_n[:-4]}', q)
                q = []
                current_speaker = None
                    
        new_df.to_csv(f'../data/logs_concat/{dir_n}', index=False)
    return

def concat_audio_unverified(folder_name):
    data_path = '../data'
    
    for dir_n in os.listdir(f'{data_path}/logs_no_col/{folder_name}'):    #logs_chunk_0.json
        # dataframe from verified speaker
        data = pickle.load(os.path.join(f'{data_path}/logs_no_col/{folder_name}', dir_n))
        print(data)
        sys.exit()
        # denoised audio
        audio_path = os.path.join(f'{data_path}/denoised/{folder_name}', f'{dir_n[5:-4]}.mp3')
        
        # new df to get data audio after concat
        new_df = pd.DataFrame(columns=['audio_path', 'model_label', 'start', 'end'])
        df = pd.read_csv(dir_path)
        
        q = []
        current_speaker = None
        for _, row in df.iterrows():
            if row['verified']:
                if not q:
                    q = [float(row['start']), float(row['end'])]
                    current_speaker = row['model_label']

                elif row['model_label'] == current_speaker:   
                    if row['start'] - q[-1] < 1: # neu khoang cach 2 wav nho va cung nguoi nois
                        q[-1] = float(row['end'])
                    else:
                        new_row = {'audio_path': f'{data_path}/concat_audio/{folder_name}/{dir_n[:-4]}/{round(q[0],1)} to {round(q[-1],1) }.mp3', 'model_label': current_speaker, 'start': round(q[0],1), 'end': round(q[-1],1)}
                        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                        create_audio(audio_path, f'{folder_name}/{dir_n[:-4]}' , q)
                        q = [float(row['start']), float(row['end'])]
                else:                               # neu la nguoi khac
                    new_row = {'audio_path': f'{data_path}/concat_audio/{folder_name}/{dir_n[:-4]}/{round(q[0],1)} to {round(q[-1],1) }.mp3', 'model_label': current_speaker, 'start': round(q[0],1), 'end': round(q[-1],1)}
                    new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                    create_audio(audio_path, f'{folder_name}/{dir_n[:-4]}', q)
                    q = [float(row['start']), float(row['end'])]
                    current_speaker = row['model_label']
            else:
                if q:
                    new_row = {'audio_path': f'{data_path}/concat_audio/{folder_name}/{dir_n[:-4]}/{round(q[0],1)} to {round(q[-1],1) }.mp3', 'model_label': current_speaker, 'start': round(q[0],1), 'end': round(q[-1],1)}
                    new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                    create_audio(audio_path, f'{folder_name}/{dir_n[:-4]}', q)
                q = []
                current_speaker = None
                    
        new_df.to_csv(f'../data/logs_concat/{dir_n}', index=False)
    return