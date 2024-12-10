import json
import pandas as pd
import os
import librosa
import soundfile as sf
folder_name = 'HAS'

def check_exist(episode, folder):
    '''Check if diary logs already exist
    exist:      return True
    not exist:  return False'''
    current_episode = [file_name for file_name in os.listdir(folder)]
    for ep in current_episode:
        if episode in ep:
            return True
    return False

def create_audio(audio_path, folder_name, episode, speaker, q):
    data_path = '../../data_unverified'
    save_path = f'{data_path}/concat_audio/{folder_name}/{episode}'

    os.makedirs(save_path, exist_ok=True)

    start, end = q[0], q[-1]
    # print(start, end)
    y, sr = librosa.load(audio_path, offset=start, duration=end - start)
    sf.write(f'{save_path}/{speaker} - {round(start,2)} - {round(end,2) }.wav', y, sr)

def concat_audio_unverified(folder_name):
    data_path = '../../data_unverified'
    
    for dir_n in os.listdir(f'{data_path}/logs_no_col/{folder_name}'):    #logs_chunk_0.json
        # dataframe from verified speaker

        # Check if file is already processed
        episode = dir_n.split(' ')[0]
        if check_exist(episode, f'{data_path}/logs_concat/{folder_name}'):
            print(f"Skip {episode}")
            continue

        file_path = os.path.join(f'{data_path}/logs_no_col/{folder_name}', dir_n)
        episode = dir_n.split(' ')[0]
        with open(file_path, 'rb') as f:
            data = json.load(f)

        # denoised audio
        audio_path = os.path.join(f'/home2/tuannd/tuanlha/Dataset/HaveASip', f'{dir_n[:-5]}.mp3')
        
        # new df to get data audio after concat
        new_df = pd.DataFrame(columns=['episode', 'audio_path', 'model_label', 'start', 'end'])
        
        q = []
        current_speaker = None
        for item in data:
            if not q:
                q = [item[0][0], item[0][1]]
                current_speaker = item[1]

            elif item[1] == current_speaker:   
                if item[0][0] - q[-1] < 1: # neu khoang cach 2 wav nho va cung nguoi nois
                    q[-1] = item[0][1]
                else:
                    new_row = {'episode': episode, 'audio_path': f'{data_path}/concat_audio/{folder_name}/{episode}/{current_speaker} - {round(q[0],2)} - {round(q[-1],2) }.mp3', 'model_label': current_speaker, 'start': round(q[0],2), 'end': round(q[-1],2)}
                    new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                    create_audio(audio_path, f'{folder_name}', episode, current_speaker, q)
                    q = [item[0][0], item[0][1]]
                    current_speaker = current_speaker
            else:                               # neu la nguoi khac
                new_row = {'episode': episode, 'audio_path': f'{data_path}/concat_audio/{folder_name}/{episode}/{current_speaker} - {round(q[0],2)} - {round(q[-1],2) }.mp3', 'model_label': current_speaker, 'start': round(q[0],2), 'end': round(q[-1],2)}
                new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                create_audio(audio_path, f'{folder_name}', episode, current_speaker, q)
                q = [item[0][0], item[0][1]]
                current_speaker = item[1]      
        # TODO: hậu xử lí nếu q vẫn còn
        if q:
            new_row = {'episode': episode, 'audio_path': f'{data_path}/concat_audio/{folder_name}/{episode}/{current_speaker} - {round(q[0],2)} - {round(q[-1],2) }.mp3', 'model_label': current_speaker, 'start': round(q[0],2), 'end': round(q[-1],2)}
            new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
            create_audio(audio_path, f'{folder_name}', episode, current_speaker, q)
        save_csv_path = f'../../data_unverified/logs_concat/{folder_name}'
        os.makedirs(save_csv_path, exist_ok=True)              
        new_df.to_csv(f'{save_csv_path}/{dir_n[:-5]}.csv', index=False)
    return

concat_audio_unverified(folder_name)