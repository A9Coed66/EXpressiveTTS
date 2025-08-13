import os
import re
import subprocess

def reverse_filter():
    '''Loại chỉ số đánh dấu trong tên file'''
    # Lấy đường dẫn thư mục hiện tại
    directory = '/home4/tuanlha/DataTest/HaveASip'
    
    # Duyệt qua tất cả các file trong thư mục
    for filename in os.listdir(directory):
        # Kiểm tra xem tên file có khớp với mẫu "số + khoảng trắng + tên" không
        match = re.match(r'^\d+\s+(.*)', filename)
        
        if match:
            # Tên mới sẽ là phần sau số và khoảng trắng
            new_name = match.group(1)
            
            # Đường dẫn đầy đủ của file cũ và mới
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            # Đổi tên file
            os.rename(old_path, new_path)
            print(f'Đã đổi tên: "{filename}" → "{new_name}"')

def get_mp3_duration_ffprobe(file_path):
    """Lấy thời lượng file MP3 bằng ffprobe (nhanh nhất)"""
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

def sort_files_by_audio_length(folder_path):
    """
    Sắp xếp tên file theo thứ tự tăng dần của độ dài audio.

    Args:
        folder_path (str): Đường dẫn thư mục chứa các file audio.

    Returns:
        list: Danh sách tên file được sắp xếp theo độ dài audio tăng dần.
    """
    audio_files = []
    file_durations = {}

    # Lấy danh sách file audio trong thư mục
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.mp3', '.wav', '.flac', '.aac')):
            file_path = os.path.join(folder_path, file_name)
            try:
                duration = get_mp3_duration_ffprobe(file_path)
                file_durations[file_name] = duration
            except Exception as e:
                print(f"Lỗi khi xử lý {file_name}: {str(e)}")

    # Sắp xếp file theo độ dài audio
    sorted_files = sorted(file_durations.items(), key=lambda x: x[1])

    return [file[0] for file in sorted_files]

# using example
# sorted_files = sort_files_by_audio_length(folder_path)
# print("Danh sách file sắp xếp theo độ dài audio tăng dần:")
# cnt = 1
# for file in sorted_files:
#     print(file)
#     os.rename(os.path.join(folder_path, file), os.path.join(folder_path, f'{cnt:02d} {file}'))
#     cnt+=1
def rename_by_length(folder_path):
    """
    Đổi tên file theo thứ tự tăng dần của độ dài audio.

    Args:
        folder_path (str): Đường dẫn thư mục chứa các file audio.
    """
    pattern = re.compile(r'\[.*?\]')
    sorted_files = sort_files_by_audio_length(folder_path)
    print("Danh sách file sắp xếp theo độ dài audio tăng dần:")
    cnt = 1
    for file in sorted_files:
        new_name = re.sub(pattern, '', file).strip()

        # os.rename(os.path.join(folder_path, file), os.path.join(folder_path, f'{cnt:02d} {new_name}'))
        print(f"Đã đổi tên: {file} → {cnt:02d} {new_name}")
        cnt+=1