import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_predict(audio_folder_path):
    command = f"python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar --data_dir '{audio_folder_path}' --num_workers 8 --bs 30 --output_dir /home2/tuannd/tuanlha/EXpressiveTTSDataset/data_unverified/05_data_extract"
    print(command)
    subprocess.run(command, shell=True, cwd="/home2/tuannd/tuanlha/NISQA")



folder_path = '/home2/tuannd/tuanlha/EXpressiveTTSDataset/data_unverified/04_split_concat_audio'
episodes = [dir for dir in os.listdir(folder_path)]
with ThreadPoolExecutor(max_workers=3) as executor:
    for cnt, audio_name in enumerate(os.listdir(folder_path)):
        # print(os.path.join(folder_path, audio_name))
        executor.submit(run_predict, os.path.join(folder_path, audio_name))

