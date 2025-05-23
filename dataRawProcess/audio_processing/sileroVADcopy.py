SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)
from IPython.display import Audio
from pprint import pprint
import glob
import os
import librosa
import soundfile as sf
import argparse
import csv
import json
import pandas as pd
from audio_processing.audio_utils import standardize_audio, save_sub_audio
from utils.logger import Logger
from utils.tool import check_exists
 
logger = Logger.get_logger()
#NOTE not clear model
# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
#                               model='silero_vad',
#                               force_reload=True)


# def vad(episode_name, playlist_name):
#     #NOTE: đưa vào trong để tiết kiệm bộ nhớ
#     model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
#                               model='silero_vad',
#                               force_reload=True)
#     (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

#     step_path = './04_vad_extract'
#     os.makedirs(os.path.join(step_path, playlist_name), exist_ok=True)
#     episode_name = check_exists(step_path, playlist_name, episode_name, type='file')
    
    
#     for episode in episode_name:
#         path_folder_file_wav = f'./04_denoise/{playlist_name}/{episode}'
#         # save_dir = f'./04_denoise/{playlist_name}/{episode}'
#         path_file_csv = f'./04_vad_extract/{playlist_name}/{episode}.csv'
#         with open(path_file_csv, 'w', encoding = 'UTF8', newline='') as f:
#             writer = csv.writer(f)

#             writer.writerow(['path', 'timestamps'])

#             for name in glob.glob(path_folder_file_wav + '/*.wav'):
#                 j = 0
#                 h = 0
#                 wav = read_audio(name, sampling_rate = SAMPLING_RATE)
#                 speech_timestamps = get_speech_timestamps(wav, model, threshold=0.5, sampling_rate=SAMPLING_RATE)

#                 sum = 0
#                 k = 0   
#                 speech_timestamps_mini = {}
#                 speech_timestamps_full = []

#                 try:
#                     while(True):
#                         sum =  sum + speech_timestamps[k]['end'] - speech_timestamps[k]['start'] 
#                         if 'start' not in speech_timestamps_mini:
#                             speech_timestamps_mini['start'] = speech_timestamps[k]['start'] 
#                         speech_timestamps_mini['end'] = speech_timestamps[k]['end']

#                         if k < len(speech_timestamps)-1:
#                             k = k + 1
#                             if sum >= 16000: #NOTE: các sub audio phải có độ dài hơn 1s
#                                 speech_timestamps_full.append(speech_timestamps_mini.copy())
#                                 speech_timestamps_mini.clear()
#                                 sum = 0
#                                 continue
#                             else:
#                                 continue
#                         else:
#                             speech_timestamps_full.append(speech_timestamps_mini.copy())
#                             speech_timestamps_mini.clear()
#                             sum = 0
#                             break
#                     writer.writerow([path_folder_file_wav + '/' + os.path.basename(name), speech_timestamps_full])
#                 except:
#                     logger.error(f'Error {name}')
#                     print(f'Error {name}')
#                     pass
#     # torch.cuda.empty_cache()

def vad(args, cfg):
    #NOTE: đưa vào trong để tiết kiệm bộ nhớ
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    step_path = './04_vad_extract'
    playlist_name = args.playlist_name
    os.makedirs(os.path.join(step_path, playlist_name), exist_ok=True)
    
    episode_list = sorted(os.listdir(os.path.join(args.data_path, args.playlist_name)))
    episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in episode_list]
    episode_name = check_exists(step_path, playlist_name, episode_name, type='file')
    
    for episode in episode_name:
        path_folder_file_wav = f'./04_denoise/{playlist_name}/{episode}'
        # save_dir = f'./04_denoise/{playlist_name}/{episode}'
        path_file_csv = f'./04_vad_extract/{playlist_name}/{episode}.csv'
        with open(path_file_csv, 'w', encoding = 'UTF8', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['path', 'timestamps'])

            for name in glob.glob(path_folder_file_wav + '/*.wav'):
                j = 0
                h = 0
                wav = read_audio(name, sampling_rate = SAMPLING_RATE)
                speech_timestamps = get_speech_timestamps(wav, model, threshold=0.5, sampling_rate=SAMPLING_RATE)

                sum = 0
                k = 0   
                speech_timestamps_mini = {}
                speech_timestamps_full = []

                try:
                    while(True):
                        sum =  sum + speech_timestamps[k]['end'] - speech_timestamps[k]['start'] 
                        if 'start' not in speech_timestamps_mini:
                            speech_timestamps_mini['start'] = speech_timestamps[k]['start'] 
                        speech_timestamps_mini['end'] = speech_timestamps[k]['end']

                        if k < len(speech_timestamps)-1:
                            k = k + 1
                            if sum >= 16000: #NOTE: các sub audio phải có độ dài hơn 1s
                                speech_timestamps_full.append(speech_timestamps_mini.copy())
                                speech_timestamps_mini.clear()
                                sum = 0
                                continue
                            else:
                                continue
                        else:
                            speech_timestamps_full.append(speech_timestamps_mini.copy())
                            speech_timestamps_mini.clear()
                            sum = 0
                            break
                    writer.writerow([path_folder_file_wav + '/' + os.path.basename(name), speech_timestamps_full])
                except:
                    logger.error(f'Error {name}')
                    print(f'Error {name}')
                    pass
    # torch.cuda.empty_cache()

           
def remove_and_rename(save_dir):
    for i in glob.glob(save_dir + '/*' ):
        k = 0
        for j in glob.glob(i + '/*'):
            if librosa.get_duration(filename = j) < 3 or librosa.get_duration(filename = j) > 15:
                os.remove(j)
            else:
                os.rename(j, i + '/audio_' + str(k) + '.wav')
                k= k+1

# vad()
# remove_and_rename()
        
def sileroVAD_pipelien(episode_name, playlist_name):
    denoise_path = f'./04_denoise/{playlist_name}'
    os.makedirs(f'./04_vad/{playlist_name}', exist_ok=True)
    for episode in episode_name:
        episode_path = os.path.join(denoise_path, episode)
        os.makedirs(episode_path, exist_ok=True)
        


def create_sub_vad(args, cfg):
    # lam viec voi tung episode

    playlist_name = args.playlist_name
    episode_list = sorted(os.listdir(os.path.join(args.data_path, args.playlist_name)))
    episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in episode_list]

    folder_path = f'./04_vad_extract/{playlist_name}'
    save_folder_path = f'./04_vad/{playlist_name}'
    os.makedirs(save_folder_path, exist_ok=True)
    for episode in episode_name:
        os.makedirs(os.path.join(save_folder_path, episode), exist_ok=True)
        dataframe_path = os.path.join(folder_path, episode + '.csv')
        df = pd.read_csv(dataframe_path)
        for index, row in df.iterrows():
            file_path = row['path']
            file_name = os.path.basename(file_path).rsplit('.', 1)[0]
            y, sr = sf.read(file_path)
            assert sr == 48000, f"Expected sample rate of 48000, but got {sr}"
            timestamps = row['timestamps']
            timestamps = timestamps.replace("'", '"')
            timestamps = json.loads(timestamps)
            h = 0
            for subtime in timestamps:
                start = subtime['start']*3
                end = subtime['end']*3
                sub_audio = y[int(start):int(end)]
                sf.write(os.path.join(save_folder_path, episode, f'{file_name}_{h}.wav'), sub_audio, sr)
                h+=1

