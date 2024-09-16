import os
import subprocess
import numpy as np
import soundfile as sf

from utils.time_converter import second_to_time, second_to_samples, time_to_name_str

# Cut audiofrom file
def lst_to_segment_dir(lst, out_dir, audio_full, sr, prefix='segment'):
    for item in lst:
        start, end = item['start'], item['end']
        start_samples = second_to_samples(start, sr)
        end_samples = second_to_samples(end, sr)
        start_str = time_to_name_str(second_to_time(start))
        end_str = time_to_name_str(second_to_time(end))
        fpath = os.path.join(out_dir, f'{prefix}-{start_str}-{end_str}.wav')
        sf.write(fpath, audio_full[start_samples:end_samples], sr)
    return

def run_ffmpeg(input_file, start_time, end_time, output_file):
    """
    Run FFmpeg command save a segment of a video start_time to end_time.
    Use when not change codec."""
    ffmpeg_cmd = [
        'ffmpeg',
        '-ss', str(start_time),
        '-to', str(end_time), # NOTE: PLACE SS, TO BEFORE -I
        '-i', input_file,
        '-c:v', 'copy',
        '-c:a', 'copy',
        output_file
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("FFmpeg command executed successfully.")
    except subprocess.CalledProcessError as e:
        print("FFmpeg command failed with error:", e)
    return

def run_ffmpeg_webm(input_file, start_time, end_time, output_file):
    """
    Run FFmpeg to save segment of video from start_time to end_time. 
    Change codec to libx265 and flac."""
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'libx265', '-crf', '18', '-c:a', 'flac',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-strict', '-2',
        output_file
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("FFmpeg command executed successfully.")
    except subprocess.CalledProcessError as e:
        print("FFmpeg command failed with error:", e)
