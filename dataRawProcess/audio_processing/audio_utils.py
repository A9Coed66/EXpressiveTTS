import os
import numpy as np
from pydub import AudioSegment
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logger import Logger
from utils.manage_memory import set_cpu_affinity
import pickle
from utils.tool import check_exists

logger = Logger.get_logger()

MAX_THREADS = 4
NORMALIZE_VOLUME = 4

def standardize(audio, cfg):
    """Preprocess single audio file"""
    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    else:
        raise ValueError("Invalid audio type")

    # Convert to WAV format
    audio = (audio.set_frame_rate(cfg["save_step"]["standardization"]["sample_rate"])
                  .set_sample_width(2)
                  .set_channels(1))

    # Normalize volume
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    normalized_audio = audio.apply_gain(min(max(gain, -NORMALIZE_VOLUME), NORMALIZE_VOLUME))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    waveform /= np.max(np.abs(waveform))  # Normalize
    
    return {"waveform": waveform, "name": name}

def standardize_audio(audio_files, cfg, playlist_name, is_save=False):
    """Process multiple audio files in parallel
    Args:
        audio_files (list): List of audio file paths
        cfg (dict): Configuration dictionary
        playlist_name (str): Name of the playlist
        is_save (bool): Whether to save the standardized files
    Returns:
        list: List of standardized audio files"""
    set_cpu_affinity([34+i for i in range(0, 6)])

    step_path = './00_standardization'
    os.makedirs(os.path.join(step_path, playlist_name), exist_ok=True)
    audio_files = check_exists(step_path, playlist_name, audio_files, type='file')



    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        results = list(executor.map(lambda x: standardize(x, cfg), audio_files))

    if is_save:
        logger.info("Saving standardized audio files...")
        os.makedirs('./00_standardization', exist_ok=True)
        os.makedirs(f'./00_standardization/{playlist_name}', exist_ok=True)
        for result in results:
            name = result["name"].rsplit('.', 1)[0]
            waveform = result["waveform"]
            sr = cfg["save_step"]["standardization"]["sample_rate"]
            sf.write(os.path.join(f'./00_standardization/{playlist_name}', f'{name}.wav'), waveform, sr)
    return results

def save_sub_audio(results, clean_diary, playlist_name, cfg):
    """
    Save sub-audio segments based on the cleaned diarization results.

    Args:
        results (list): List of processed audio results.
        clean_diary (list): List of cleaned diarization results.
    """
    sr = cfg["save_step"]["standardization"]["sample_rate"]
    def process_segment(result, segment, sr):
        """
        Process and save a single audio segment.

        Args:
            result (dict): Processed audio result containing waveform and name.
            segment (list): A segment containing start time, end time, and speaker.
            sr (int): Sample rate of the audio.
        """
        start, end = segment[0]
        start = round(start, 2)
        end = round(end, 2)
        speaker = segment[1]
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        sub_audio = result["waveform"][start_sample:end_sample]
        sub_audio_path = f"./03_concat_audio/{playlist_name}/{os.path.basename(result['name']).rsplit('.', 1)[0]}/{speaker}_{start}_{end}.wav"
        sf.write(sub_audio_path, sub_audio, sr)

    logger.info("Saving sub-audio segments based on cleaned diarization results...")


    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        os.makedirs('./03_concat_audio', exist_ok=True)
        os.makedirs(f'./03_concat_audio/{playlist_name}', exist_ok=True)

        for result, diary in zip(results, clean_diary):
            print(os.path.join(f'./03_concat_audio/{playlist_name}', os.path.basename(result["name"]).rsplit('.', 1)[0]))
            if os.path.exists(os.path.join(f'./03_concat_audio/{playlist_name}', os.path.basename(result["name"]).rsplit('.', 1)[0])):
                print("Already have it, skipping")
                continue

            # Create directory for each audio file
            os.makedirs(os.path.join(f'./03_concat_audio/{playlist_name}', os.path.basename(result["name"]).rsplit('.', 1)[0]), exist_ok=True)
            for segment in diary:
                futures.append(executor.submit(process_segment, result, segment, sr))
        for future in as_completed(futures):
            future.result()  # Ensure all tasks are completed
    logger.info("Saved sub-audio segments based on cleaned diarization results")

def save_sub_audio(args, cfg):
    """
    Save sub-audio segments based on the cleaned diarization results.

    Args:
        results (list): List of processed audio results.
        clean_diary (list): List of cleaned diarization results.
    """
    sr = cfg["save_step"]["standardization"]["sample_rate"]
    def process_segment(waveform, episode, segment, sr):
        """
        Process and save a single audio segment.

        Args:
            result (dict): Processed audio result containing waveform and name.
            segment (list): A segment containing start time, end time, and speaker.
            sr (int): Sample rate of the audio.
        """
        start, end = segment[0]
        start = round(start, 2)
        end = round(end, 2)
        speaker = segment[1]
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        sub_audio = waveform[start_sample:end_sample]
        sub_audio_path = f"./03_concat_audio/{args.playlist_name}/{episode}/{speaker}_{start}_{end}.wav"
        sf.write(sub_audio_path, sub_audio, sr)

    logger.info("Saving sub-audio segments based on cleaned diarization results...")


    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        os.makedirs('./03_concat_audio', exist_ok=True)
        os.makedirs(f'./03_concat_audio/{args.playlist_name}', exist_ok=True)
        
        episode_list = sorted(os.listdir(os.path.join(args.data_path, args.playlist_name)))
        episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in episode_list]
        for episode in episode_name:
            waveform , sr = sf.read(os.path.join('./00_standardization', args.playlist_name, f"{episode}.wav"))
            print(os.path.join(f'./03_concat_audio/{args.playlist_name}', episode))
            if os.path.exists(os.path.join(f'./03_concat_audio/{args.playlist_name}', episode)):
                print("Already have it, skipping")
                continue
            os.makedirs(os.path.join(f'./03_concat_audio/{args.playlist_name}', episode), exist_ok=True)

            # Create directory for each audio file
            diary_path = os.path.join('./01_clean_diarization', args.playlist_name, f"{episode}.pkl")
            with open(diary_path, 'rb') as f:
                diary = pickle.load(f)
            for segment in diary:
                futures.append(executor.submit(process_segment, waveform, episode, segment, sr))
        for future in as_completed(futures):
            future.result()  # Ensure all tasks are completed
    logger.info("Saved sub-audio segments based on cleaned diarization results")