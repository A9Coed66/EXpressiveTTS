SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)
from IPython.display import Audio
from pprint import pprint
import glob
import os
import librosa
import soundfile as sf
import torchaudio
import argparse
import csv
import json
import pandas as pd
from audio_processing.audio_utils import standardize_audio, save_sub_audio
from utils.logger import Logger
from utils.tool import check_exists
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
 
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

def read_audio_vad(path: str,
               sampling_rate: int = 16000,
               normalize=False):

    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sampling_rate:
        if sr != sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                       new_freq=sampling_rate)
            wav = transform(wav)
            sr = sampling_rate

    if normalize and wav.abs().max() != 0:
        wav = wav / wav.abs().max()

    return wav.squeeze(0)

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

# NOTE CSV vertion
def vad(args, cfg):
    #NOTE: đưa vào trong để tiết kiệm bộ nhớ
    logger.info('Start VAD process')
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    step_path = './02_vad_extract_pkl'
    playlist_name = args.playlist_name
    os.makedirs(step_path, exist_ok=True)
    os.makedirs(os.path.join(step_path, playlist_name), exist_ok=True)
    
    episode_list = sorted(os.listdir(os.path.join('./00_standardization', args.playlist_name)))
    episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in episode_list]
    episode_name = check_exists(step_path, playlist_name, episode_name, type='file')
    
    for episode in episode_name:
        # Audio information
        logger.info(f'Start processing episode: {episode}')
        path_folder_file_wav = f'./00_standardization/{playlist_name}/{episode}.wav'
        waveform = read_audio(path_folder_file_wav)

        # Segments information
        path_clean_diary = f'./01_clean_diarization/{playlist_name}/{episode}.pkl'
        with open(path_clean_diary, 'rb') as f:
            data_clean = pickle.load(f)

        path_file_csv = f'./02_vad_extract_pkl/{playlist_name}/{episode}.pkl'

        clean_vad = []

        for data_point in data_clean:
            j = 0
            h = 0
            wav = waveform[int(data_point[0]*SAMPLING_RATE):int(data_point[1]*SAMPLING_RATE)]
            speech_timestamps = get_speech_timestamps(wav, model, threshold=0.5, sampling_rate=SAMPLING_RATE)

            before_clean = speech_timestamps.copy()

            if not data_point[3]:
                if speech_timestamps:
                    speech_timestamps.pop(0)
            if not data_point[4]:
                if speech_timestamps:
                    speech_timestamps.pop(-1)

            sum = 0
            k = 0   
            speech_timestamps_mini = {}
            speech_timestamps_full = []

            if not speech_timestamps:
                continue

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
                clean_vad.append([data_point, speech_timestamps_full, before_clean])
            except Exception as e:
                print(e)
                logger.error(f'Error {data_point}')
                print(f'Error {data_point}')
                pass
        with open(path_file_csv, 'wb') as f:
            pickle.dump(clean_vad, f)
        logger.info(f'VAD process for {episode} completed')
#     # torch.cuda.empty_cache()

           
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

    step_path = './03_vad'
    playlist_name = args.playlist_name
    save_folder_path = f'./03_vad/{playlist_name}'
    os.makedirs(step_path, exist_ok=True)
    os.makedirs(save_folder_path, exist_ok=True)


    episode_list = sorted(os.listdir(os.path.join('./00_standardization', args.playlist_name)))
    episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in episode_list]
    episode_name = check_exists(step_path, playlist_name, episode_name, type='file')

    folder_path = f'./02_vad_extract_pkl/{playlist_name}'
    for episode in episode_name:
        os.makedirs(os.path.join(save_folder_path, episode), exist_ok=True)
        episode_path = os.path.join(folder_path, episode + '.pkl')
        with open(episode_path, 'rb') as f:
            # Load the data from the pickle file
            data_clean = pickle.load(f)
        y, sr = sf.read(os.path.join('./00_standardization', playlist_name, episode + '.wav'))
        
        queue = []
        for data_point in data_clean:
            queue = []
            
            for segment in data_point[1]:
                if not queue:
                    queue.append(segment)
                    continue

                if (segment['end'] - queue[-1]['start']) / 16000 > 20:
                    queue.append(segment)
                elif (segment['start'] - queue[-1]['end']) / 16000 < 0.25:
                    queue[-1]['end'] = segment['end']
                else:
                    queue.append(segment)

            start, end = data_point[0][0], data_point[0][1]
            assert sr == 24000, f"Expected sample rate of 48000, but got {sr}"
            current_audio = y[int(start*sr):int(end*sr)]

            cnt = 0
            for segment in queue:
                speaker = data_point[0][2]
                seg_start = segment['start'] * 3 / 2
                seg_end = segment['end'] * 3 / 2
                sub_audio = current_audio[int(seg_start):int(seg_end)]
                file_name = f"{speaker}_{round(start,2)}_{round(end,2)}_{cnt}.wav"
                cnt+=1
                sf.write(os.path.join(save_folder_path, episode, file_name), sub_audio, sr)