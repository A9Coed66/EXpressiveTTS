import json
import pandas as pd
import os
import librosa
import soundfile as sf
import argparse

parser = argparse.ArgumentParser(description="Remove collision script")
parser.add_argument("--data_path", type=str, default='/home2/tuannd/tuanlha/EXpressiveTTSDataset/data_unverified', help="Path to the data directory")
parser.add_argument("--save_path", type=str, default=f'/home2/tuannd/tuanlha/EXpressiveTTSDataset/data_unverified/02_concat_audio', help="Path to the save directory")
args = parser.parse_args()


def check_exist(episode, folder):
    '''Check if diary logs already exist
    exist:      return True
    not exist:  return False'''
    return episode in [dir for dir in os.listdir(folder)]
    # current_episode = [file_name for file_name in os.listdir(folder)]
    # if episode in current_episode:
    #     return True
    # return False

def create_audio(audio_path, episode, speaker, q):
    save_path = args.save_path

    os.makedirs(save_path, exist_ok=True)

    start, end = q[0], q[-1]
    # print(start, end)
    y, sr = librosa.load(audio_path, offset=start, duration=end - start)
    sf.write(f'{save_path}/{speaker} - {round(start,2)} - {round(end,2) }.wav', y, sr)

def concat_audio_unverified():
    data_path = args.data_path
    
    for dir_n in os.listdir(f'{data_path}/01_logs_no_col'):    #logs_chunk_0.json
        # dataframe from verified speaker

        # Check if file is already processed
        episode = dir_n.split('-')[0]
        #Kiem tra xem tap da dich chua
        if check_exist(episode, f'{data_path}/02_concat_audio'):
            print(f"Skip {episode}")
            continue
        print(f"Bat dau: {episode}")

        file_path = os.path.join(f'{data_path}/01_logs_no_col', dir_n)
        with open(file_path, 'rb') as f:
            data = json.load(f)

        # Kiem tra xem co audio path chua
        audio_path = os.path.join(f'/home2/tuannd/tuanlha/HaveASipDataset', f'{dir_n[:-5]}.mp3')
        if not os.path.exists(audio_path):
            print(f"Thieu {audio_path}")
            continue
        # new df to get data audio after concat
        # new_df = pd.DataFrame(columns=['episode', 'audio_path', 'model_label', 'start', 'end'])
        
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
                    # new_row = {'episode': episode, 'audio_path': f'{data_path}/02_concat_audio/{episode}/{current_speaker}-{round(q[0],2)}-{round(q[-1],2) }.mp3', 'model_label': current_speaker, 'start': round(q[0],2), 'end': round(q[-1],2)}
                    # new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                    create_audio(audio_path, episode, current_speaker, q)
                    q = [item[0][0], item[0][1]]
            else:                               # neu la nguoi khac
                # new_row = {'episode': episode, 'audio_path': f'{data_path}/02_concat_audio/{episode}/{current_speaker}-{round(q[0],2)}-{round(q[-1],2) }.mp3', 'model_label': current_speaker, 'start': round(q[0],2), 'end': round(q[-1],2)}
                # new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                create_audio(audio_path, episode, current_speaker, q)
                q = [item[0][0], item[0][1]]
                current_speaker = item[1]      
        # TODO: hậu xử lí nếu q vẫn còn
        if q and q[-1]-q[0]>1:
            # new_row = {'episode': episode, 'audio_path': f'{data_path}/02_concat_audio/{episode}/{current_speaker} - {round(q[0],2)} - {round(q[-1],2) }.mp3', 'model_label': current_speaker, 'start': round(q[0],2), 'end': round(q[-1],2)}
            # new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
            create_audio(audio_path, episode, current_speaker, q)
        # save_csv_path = f'../../data_unverified/02_logs_concat'
        # os.makedirs(save_csv_path, exist_ok=True)              
        # new_df.to_csv(f'{save_csv_path}/{dir_n[:-5]}.csv', index=False)
    return

concat_audio_unverified()