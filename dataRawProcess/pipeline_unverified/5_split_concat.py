import os, librosa
import soundfile as sf

def check_exist(episode, folder):
    '''Check if diary logs already exist
    exist:      return True
    not exist:  return False'''
    current_episode = [file_name for file_name in os.listdir(folder)]
    for ep in current_episode:
        if episode in ep:
            return True
    return False


def create_audio_ver2(audio_path, episode, speaker, text, q, begin):
    audio_path = audio_path.replace('mp3', 'wav')
    data_path = '../../data_unverified'
    save_path = f'{data_path}/04_split_concat_audio/{episode}'

    os.makedirs(save_path, exist_ok=True)

    start, end = round(q[0],2), round(q[-1],2)
    # print(start, end, audio_path)
    y, sr = librosa.load(audio_path, offset=start, duration=end - start)
    sf.write(f'{save_path}/{speaker} - {round(begin+start,2)} - {round(begin+end,2) }.wav', y, sr)
    # Create lab file:
    with open(f'{save_path}/{speaker} - {round(begin+start,2)} - {round(begin+end,2) }.lab', 'w') as f:
        f.write(text)

import json
def split_audio():
    data_path = '../../data_unverified'
    transcription_folder = "../../data_unverified/03_transcription"

    for episode in os.listdir(transcription_folder):

        # Check if file is already processed
        if check_exist(episode, f'../../data_unverified/04_split_concat_audio'):
            print(f"Skip {episode}")
            continue

        logs_folder = os.path.join(transcription_folder, episode)    
        for log in os.listdir(logs_folder):     # log in logs folder for episode
            with open(os.path.join(logs_folder, log), 'r') as f:
                data = json.load(f)
            
            #get start time, speaker in name, segments time in data
            start = float(log.split(' ')[2])
            speaker = log.split(' ')[0]
            segments = data['segments']

            for seg in segments:
                sub_start, sub_end = float(seg['start']), float(seg['end'])
                if sub_end - sub_start < 1 or sub_end - sub_start>10:
                    continue
                text = seg['text']
                audio_path = transcription_folder.replace('03_transcription', '02_concat_audio')
                create_audio_ver2(os.path.join(audio_path,episode,log[:-5]), episode, speaker, text, [sub_start, sub_end], start)

split_audio()