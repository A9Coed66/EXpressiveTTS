import soundfile as sf
import librosa
import os
import pickle
import json
import argparse

parser = argparse.ArgumentParser(description="Remove collision script")
parser.add_argument("--data_path", type=str, default='/home2/tuannd/tuanlha/EXpressiveTTSDataset/data_unverified', help="Path to the data directory")
parser.add_argument("--save_path", type=str, default=f'/home2/tuannd/tuanlha/EXpressiveTTSDataset/data_unverified/01_logs_no_col', help="Path to the save directory")
args = parser.parse_args()

def get_episode(folder_path):
    episode = []
    for file_name in os.listdir(folder_path):
        episode.append(file_name.split(' ')[0])
    return episode

def create_origin_logs(diarization):
    logs = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        logs.append([round(turn.start,2), round(turn.end,2), speaker])
    return logs

def remove_collision():
    '''
    Loại đi các đoạn audio đè nhau giữa người nói'''
    data_path = args.data_path
    save_path = args.save_path

    os.makedirs(save_path, exist_ok=True)
    rmed_episodes = get_episode(save_path)
    
    for file_name in os.listdir(f'{data_path}/00_diary'):
        episode = file_name.split(' ')[0]
        
        if episode in rmed_episodes:
            print(f"Skip {episode}")
            continue

        print(f'{file_name} - {episode}')
        file_path = os.path.join(f'{data_path}/00_diary', file_name)
        with open(file_path, 'rb') as f:
            diary = pickle.load(f)
        origin_log = create_origin_logs(diary)
        log = []
        # Pre
        if not origin_log:
            with open(f'{save_path}/{file_name[:-4]}.json', 'w') as f:   
                json.dump(log, f)
            continue
        
        # Chọn ra các mốc cần lưu ý
        queue = []
        last_end = 0
        
        # In
        for i in range(0, len(origin_log)-1):
            start, end, speaker = origin_log[i][0], origin_log[i][1], origin_log[i][2]
            if start > last_end:
                if queue:   # Add queue and reset queue
                    log.append(queue)
                    queue = []
                if end - start > 1:
                    queue = [[start, end], speaker]
                last_end = end

            else:
                last_end = max(last_end, end)
                queue = []
                

        
        # End
        if origin_log[-1][0] > last_end+1:
            if origin_log[-1][1] - origin_log[-1][0]>1:
                log.append([[origin_log[-1][0], origin_log[-1][1]], origin_log[-1][2]])

        # Save
        with open(f'{save_path}/{file_name[:-4]}.json', 'w') as f:   
            # /log_no_col/Yen nhau di/logs_chunk_1.json
            json.dump(log, f)
    return
remove_collision()

# def split_diarization(folder_name):
#     """
#     Use:
#         logs_no_collision
#         audio_path
#     Output:
#         split_diary
#             Nguoi phan xu
#                 chunk_1
#                     output 0.0 to 1.0.wav
#                     output 1.0 to 2.0.wav
#                 chunk_2...
#     """
#     print("Start Split base on diarization")
#     data_path = '../../data_unverified'
#     save_path = f'{data_path}/split_diary/{folder_name}'

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
    

#     for file_name in os.listdir(f'{data_path}/logs_no_col/{folder_name}'):
#         print(file_name)
#         with open(f'{data_path}/logs_no_col/{folder_name}/{file_name}', 'r') as f:
#             logs = json.load(f)
#         audio_path = f'{data_path}/denoised/{folder_name}/{file_name[5:-5]}.mp3'
#         # print(audio_path)
#         y, sr = librosa.load(audio_path)
#         for log in logs:
#             if not os.path.exists(f'{save_path}/{file_name[5:-5]}'):
#                 os.makedirs(f'{save_path}/{file_name[5:-5]}')
            
#             start, end, speaker = log[0][0], log[0][1], log[1]
#             # print(start, end, speaker)
#             segment = y[int(start*sr):int(end*sr)]
#             sf.write(f'{save_path}/{file_name[5:-5]}/{round(start,1)} {round(end,1)} {speaker}.wav', segment, sr)

