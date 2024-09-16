import soundfile as sf
import librosa
import os
import json

def create_origin_logs(diarization):
    logs = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        logs.append([round(turn.start,2), round(turn.end,2), speaker])
    return logs

def remove_collision(folder_name):
    for file_name in os.listdir(f'diary/{folder_name}'):
        file_path = os.path.join(f'diary/{folder_name}', file_name)
        with open(file_path, 'r') as f:
            diary = json.load(f)
        origin_log = create_origin_logs(diary)
        log = []
        # Pre
        preivous = [origin_log[0][0], origin_log[0][1]]
        last_end = origin_log[0][1]
        if last_end < origin_log[1][0]:
            log.append([preivous, origin_log[0][2]])
        
        # In
        for i in range(1, len(origin_log)-1):
            start, end = origin_log[i][0], origin_log[i][1]
            if start < last_end:
                last_end = max(last_end, end)
            else:
                if end > origin_log[i+1][0]:
                    preivous = [start, end]
                else:
                    preivous = [start, last_end]
                    if end-start < 1:
                        continue
                    log.append([[start, end], origin_log[i][2]])
        
        # End
        if origin_log[-1][0] > last_end:
            log.append([[origin_log[-1][0], origin_log[-1][1]], origin_log[-1][2]])

        # Save
        with open(f'log_no_col/{folder_name}/{file_name}', 'w') as f:   
            # /log_no_col/Yen nhau di/logs_chunk_1.json
            json.dump(log, f)
    return

def split_diarization(folder_name):
    """
    Use:
        logs_no_collision
        audio_path
    Output:
        split_diary
            Nguoi phan xu
                chunk_1
                    output 0.0 to 1.0.wav
                    output 1.0 to 2.0.wav
                chunk_2...
    """
    for file_name in os.listdir(f'./log_no_col/{folder_name}'):
        with open(f'./log_no_col/{folder_name}/{file_name}', 'r') as f:
            logs = json.load(f)
        audio_path = f'./denoised/{folder_name}/{file_name[5:-5]}.wav'
        y, sr = librosa.load(audio_path)
        for log in logs:
            start, end = log
            segment = y[int(start*sr):int(end*sr)]
            sf.write(f'split_diary/{folder_name}/{file_name[5:-5]}/output {round(start,1)} to {round(end,1)}.wav', segment, sr)