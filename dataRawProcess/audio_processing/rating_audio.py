import os
import subprocess
import psutil
from concurrent.futures import ThreadPoolExecutor
from wvmos import get_wvmos
from utils.logger import Logger
import pandas as pd

logger = Logger()
model = get_wvmos()

def run_predict(audio_folder_path, playlist_name, episode):
    os.makedirs(f'/home4/tuanlha/EXpressiveTTS/dataRawProcess/05_data_extract/{playlist_name}', exist_ok=True)
    # os.makedirs(f'/home4/tuanlha/EXpressiveTTS/dataRawProcess/05_data_extract/{playlist_name}/{episode}', exist_ok=True)
    command = f"python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar --data_dir '{audio_folder_path}' --num_workers 8 --bs 30 --output_dir /home4/tuanlha/EXpressiveTTS/dataRawProcess/05_data_extract/{playlist_name}"
    print(command)
    subprocess.run(command, shell=True, cwd="/home4/tuanlha/NISQA")
def limit_cpu_for_diarization():
    # return
    """Giới hạn process và các subprocess chỉ chạy trên core 0 và 1"""
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity([i for i in range(36,40)])  # Chỉ dùng core 0 và 1
        print(f"Process {os.getpid()} bị giới hạn trên core: {p.cpu_affinity()}")
    except Exception as e:
        print(f"Không thể thiết lập cpu_affinity: {e}")

def rating_audio(args, cfg):

    playlist_name = args.playlist_name
    episode_list = sorted(os.listdir(os.path.join(args.data_path, args.playlist_name)))
    episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in episode_list]

    audio_folder_path = f'/home4/tuanlha/EXpressiveTTS/dataRawProcess/04_vad/{playlist_name}'
    os.makedirs(audio_folder_path, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2, initializer=limit_cpu_for_diarization) as executor:
        limit_cpu_for_diarization()
        for cnt, episode in enumerate(episode_name):
            executor.submit(run_predict, os.path.join(audio_folder_path, episode), playlist_name, episode)   
    # logger.info("Done rating with NISQA")
    # dct = {}
    for episode in episode_name:

        mos = model.calculate_dir(os.path.join(audio_folder_path, episode), mean=False)
        file_list = sorted(os.listdir(os.path.join(audio_folder_path, episode)))
        # file_list = [os.path.join(audio_folder_path, episode, file) for file in file_list]
        # print(file_list)
        df = pd.DataFrame({
            'audio_path': file_list,  # Cột 1: đường dẫn file
            'score': mos      # Cột 2: điểm số tương ứng
        })
        df.to_csv(f'/home4/tuanlha/EXpressiveTTS/dataRawProcess/05_data_extract/{playlist_name}/{episode}_wvmos.csv', index=False)


def process_filter_by_rating(args, episode):
    csv_path = f'./05_data_extract/{args.playlist_name}/'
    nisqa_path = os.path.join(csv_path, f'{episode}_NISQA_results.csv')
    wvmos_path = os.path.join(csv_path, f'{episode}_wvmos.csv')
    df_1 = pd.read_csv(nisqa_path)
    df_2 = pd.read_csv(wvmos_path)
    
    # From df_1, remove audio which MOS < 3.5
    df_1 = df_1[df_1['mos_pred'] < 3.5]['deg']
    df_1 = df_1.tolist()
    df_2 = df_2[df_2['score']<3]['audio_path']
    df_2 = df_2.tolist()

    folder_path = os.path.join('./04_vad', args.playlist_name, episode)

    for file in df_1:
        file = os.path.join(folder_path, file)
        if os.path.exists(file):
            os.remove(file)
        else:
            print(f"File {file} does not exist")
    for file in df_2:
        file = os.path.join(folder_path, file)
        if os.path.exists(file):
            os.remove(file)
        else:
            print(f"File {file} does not exist")


def filter_by_rating(args, cfg):
    playlist_name = args.playlist_name
    episode_list = sorted(os.listdir(os.path.join(args.data_path, args.playlist_name)))
    episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in episode_list]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # limit_cpu_for_diarization()
        for cnt, episode in enumerate(episode_name):
            # print(f"Processing {cnt+1}/{len(episode_name)}: {episode}")
            executor.submit(process_filter_by_rating, args, episode)