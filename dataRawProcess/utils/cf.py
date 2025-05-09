import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial


def get_mp3_duration_ffprobe(file_path):
    """
    Args:
        file_path (Đường dẫn đến tệp mp3):
    Returns:
        float: Thời gian của tệp mp3 (đơn vị giây) hoặc None nếu có lỗi.
    """
    cmd = [
        'ffprobe', 
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    try:
        output = subprocess.check_output(cmd).decode('utf-8').strip()
        return float(output)
    except Exception as e:
        print(f"Lỗi: {e}")
        return None

def process_episode(episode_path):
    """Xử lý 1 episode và trả về tổng duration"""
    total = 0
    for file_name in os.listdir(episode_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(episode_path, file_name)
            total += get_mp3_duration_ffprobe(file_path)
    return total

def get_playlist_duration(folder_path):
    """Tính tổng duration của playlist với multiprocessing"""
    # Lấy danh sách các episode
    episodes = [os.path.join(folder_path, ep) for ep in os.listdir(folder_path)]
    
    # Sử dụng multiprocessing ở level này (không nested)
    with Pool(processes=min(cpu_count(), 4)) as pool:
        durations = pool.map(process_episode, episodes)
    
    total_time = sum(durations)
    print(f'Total time: {total_time}s (~{total_time/3600:.2f} hours)')
    return total_time

bins = [i for i in range (15)]
def create_bins(data, file_name):
    # Đếm số phần tử trong mỗi khoảng
    counts, bin_edges = np.histogram(data, bins=bins)

    # Tạo nhãn cho các khoảng (ví dụ: 0.0-0.5, 0.5-1.0,...)
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 5))
    bars = plt.bar(bin_labels, counts, color='skyblue', edgecolor='black')

    # Thêm số lượng lên từng cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom')

    plt.title('Phân phối giá trị theo khoảng', fontweight='bold')
    plt.xlabel(f'{file_name}')
    plt.ylabel('Số lượng phần tử')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

