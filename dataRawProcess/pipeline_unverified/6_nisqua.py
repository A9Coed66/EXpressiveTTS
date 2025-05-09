import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import psutil

def run_predict(audio_folder_path):
    command = f"taskset -c 36,37,38,39 python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar --data_dir '{audio_folder_path}' --num_workers 8 --bs 30 --output_dir /home4/tuanlha/EXpressiveTTS/dataRawProcess/05_data_extract"
    print(command)
    subprocess.run(command, shell=True, cwd="/home4/tuanlha/NISQA")

def limit_cpu_for_diarization():
    # return
    """Giới hạn process và các subprocess chỉ chạy trên core 0 và 1"""
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity([i for i in range(32,40)])  # Chỉ dùng core 0 và 1
        print(f"Process {os.getpid()} bị giới hạn trên core: {p.cpu_affinity()}")
    except Exception as e:
        print(f"Không thể thiết lập cpu_affinity: {e}")

folder_path = '/home4/tuanlha/EXpressiveTTS/dataRawProcess/05_transcript/HaveASip'
episodes = [dir for dir in os.listdir(folder_path)]
with ThreadPoolExecutor(max_workers=2, initializer=limit_cpu_for_diarization) as executor:
    for cnt, audio_name in enumerate(os.listdir(folder_path)):
        # print(os.path.join(folder_path, audio_name))
        executor.submit(run_predict, os.path.join(folder_path, audio_name))

