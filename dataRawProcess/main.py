import argparse
import json
from pprint import pprint
import multiprocessing
import re
from utils.tool import (
    export_to_mp3,
    load_cfg,
    get_audio_files,
    detect_gpu,
    check_env,
    calculate_audio_stats,
)
import json
import librosa
import subprocess
from pyannote.audio import Pipeline
import time
from utils.logger import Logger, time_logger
import torch, os
from pydub import AudioSegment
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import pickle
import soundfile as sf
import json

from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
from concurrent.futures import ThreadPoolExecutor
import argparse
import os

from tqdm import tqdm
import whisperx
import psutil

MAX_THREADS = 4
NORMALIZE_VOLUME = 4


def standarlization_audio(audio_files, is_save=False):
    """Xử lý song song nhiều file audio cùng lúc
    
    Args:
        audio_files (list): List of audio ful file path 
    Returns:
        list: list of standardization audio (dict: waveform, name)   
    """
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        results = list(executor.map(standardize, audio_files))

    if is_save:
        logger.info("Saving standardized audio files...")
        os.makedirs('./00_standardization', exist_ok=True)
        os.makedirs(f'./00_standardization/{args.playlist_name}', exist_ok=True)
        for result in results:
            name = result["name"].rsplit('.', 1)[0]
            waveform = result["waveform"]
            sr = cfg["save_step"]["standardization"]["sample_rate"]
            sf.write(os.path.join(f'./00_standardization/{args.playlist_name}', f'{name}.wav'), waveform, sr)
    logger.info("Saved standardized audio files")
    return results

def standardize(audio):
    """
    Preprocess audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object, the audio to be preprocessed.

    Returns:
        dict: A dictionary containing the preprocessed audio waveform, audio file name, and sample rate, formatted as:
              {
                  "waveform": np.ndarray, the preprocessed audio waveform, dtype is np.float32, shape is (num_samples,)
                  "name": str, the audio base name
              }

    Raises:
        ValueError: If the audio parameter is neither a str nor an AudioSegment.
    """
    start_time = time.time()

    # global audio_count
    name = "audio"

    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    else:
        raise ValueError("Invalid audio type")

    logger.debug("Entering the preprocessing of audio")

    # Convert the audio file to WAV format
    audio = (audio.set_frame_rate(cfg["save_step"]["standardization"]["sample_rate"])).set_sample_width(2).set_channels(1)
    logger.debug("Audio file converted to WAV format")

    # Calculate the gain to be applied
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    # Normalize volume and limit gain range to between -3 and 3
    normalized_audio = audio.apply_gain(min(max(gain, -NORMALIZE_VOLUME), NORMALIZE_VOLUME))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    max_amplitude = np.max(np.abs(waveform))
    waveform /= max_amplitude  # Normalize

    logger.debug(f"waveform shape: {waveform.shape}")
    logger.debug("waveform in np ndarray, dtype=" + str(waveform.dtype))

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds")
    return {
        "waveform": waveform,
        "name": name
    }
def limit_cpu_for_diarization():
    # return
    """Giới hạn process và các subprocess chỉ chạy trên core 0 và 1"""
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity([i for i in range(0,8)])  # Chỉ dùng core 0 và 1
        print(f"Process {os.getpid()} bị giới hạn trên core: {p.cpu_affinity()}")
    except Exception as e:
        print(f"Không thể thiết lập cpu_affinity: {e}")

def get_diary(model, episode_name, waveform, sample_rate):
    """
    Get diarization results for the given waveform and sample rate.
    Returns the diarization result.
    """
    # Convert waveform to tensor
    waveform.to(torch.device(f"cuda:{args.cuda_id}"))
    # Get diarization result
    diary, embeddings = model({'waveform': waveform, 'sample_rate': sample_rate}, return_embeddings=True)
    # logger.info(f"Processed file: {episode_name}")
    return [diary, embeddings]

# def process_diarization_task(episode_name, waveform, sample_rate):
#     """
#     Get diarization results for the given waveform and sample rate.
#     Returns the diarization result.
#     """
#     # Khởi tạo model trong process con
#     secret = "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"
#     # logger.info("Loading diarization model...")
#     diarization = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1",
#         use_auth_token=secret).to(torch.device(f"cuda:1"))
#     # logger.info("Diarization model loaded successfully")

#     # Convert waveform to tensor
#     waveform.to(torch.device(f"cuda:1"))

#     # Get diarization result
#     diary, embeddings = diarization({'waveform': waveform, 'sample_rate': sample_rate}, return_embeddings=True)
#     # logger.info(f"Processed file: {episode_name}")
#     del diarization
#     torch.cuda.empty_cache()
#     return [diary, embeddings]

def speaker_diarization(results, episode_name, is_save=False):
    """
    Perform speaker diarization on the given audio.
    Returns diarization results and ensures model is cleared from memory.

    Args:
        results (list): List of audio results, where each result is a dictionary containing 'waveform' and 'name'.
    """

    # Giới hạn số lượng CPU cores sử dụng
    os.environ["OMP_NUM_THREADS"] = "10"  # Giới hạn OpenMP threads (cho NumPy, PyTorch)
    os.environ["MKL_NUM_THREADS"] = "10"  # Giới hạn Intel MKL threads
    torch.set_num_threads(10)  # Giới hạn threads cho PyTorch CPU operations


    #Khởi tạo model trong process con
    secret = "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"
    # logger.info("Loading diarization model...")
    diarization = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=secret).to(torch.device(f"cuda:{args.cuda_id}"))
    # logger.info("Diarization model loaded successfully")
    diary_results = []
    try:
        logger.info("Processing diarization with spawn method...")

        for item in results:
            waveform = item["waveform"]
            name = item["name"]
            sample_rate = cfg["save_step"]["standardization"]["sample_rate"]
            
            # Get diarization result
            diary, embeddings = diarization({'waveform': torch.from_numpy(waveform).unsqueeze(0), 'sample_rate': sample_rate}, return_embeddings=True)
            diary_results.append({"diary": diary, "embeddings": embeddings})
            logger.info(f"Processed file: {name}")

        #FIXME: đang sai cần sửa
        if is_save:
            logger.info("Saving diarization results...")
            os.makedirs('./01_raw_diarization', exist_ok=True)
            os.makedirs(f'./01_raw_diarization/{args.playlist_name}', exist_ok=True)
            for i in range(len(diary_results)):
                diary = diary_results[i]['diary']
                name = episode_name[i]
                with open(os.path.join(f'./01_raw_diarization/{args.playlist_name}', f'{name}.pkl'), 'wb') as f:
                    pickle.dump(diary, f)
            logger.info("Saved diarization results")
        return diary_results
    
    finally:
        # Giải phóng bộ nhớ
        if 'diarization' in locals():
            del diarization
            torch.cuda.empty_cache()
            logger.info("Diarization model cleared from GPU memory")
        
def clean_diary(diarys, episode_name, is_save=False):
    """
    Remove overlapping speaker segments from diarization results using multithreading.

    Args:
        diarys (list): List of diarization results, where each result is a list of speaker segments.

    Returns:
        list: List of cleaned diarization results with overlapping segments removed.
    """

    def check_length_segment(segment):
        """Check length of segment"""
        start, end = segment[0]
        if end - start < 1.0:
            return False
        else:
            return True

    def remove_collision(diary):
        """No collision & concated segments
            Step 1: Remove collision
            Step 2: Concatenate segments
            Step 3: Remove segments that have speaker talk total time less than 8%
        """
        origin_log = [
            [round(turn.start, 3), round(turn.end, 3), speaker]
            for turn, _, speaker in diary.itertracks(yield_label=True)
        ]
        print(origin_log)
        
        processed_log = []
        current_segment = []
        last_end = -1
        min_segment_length = 0.1  # Minimum segment length in seconds
        min_selent_length = 0.4

        # Step 1: Remove collision
        for start, end, speaker in origin_log:

            # No collision
            if start >= last_end:
                if current_segment:  
                    processed_log.append(current_segment)
                    current_segment = []
                if end-start>min_segment_length:
                    current_segment = [[start, end], speaker]
            # Collision
            else:
                current_segment = []
            last_end = max(last_end, end)  # Update last_end to handle overlaps
            
        if current_segment:
            processed_log.append(current_segment)

        # Step 2: Concatenate segments
        final_log = []
        current_segment = []
        current_speaker = None
        for segment in processed_log:
            # No segment in queue
            if not current_segment:
                current_segment = segment
                current_speaker = segment[1]
            
            # Segment in queue
            else:
                # Check if the current seent is the same speaker
                if current_speaker == speaker:
                    if segment[0][0] - current_segment[0][1] < min_selent_length:
                        current_segment[0][1] = segment[0][1]
                    else:
                        final_log.append(current_segment) if check_length_segment(current_segment) else None
                        current_segment = segment
                else:
                    final_log.append(current_segment) if check_length_segment(current_segment) else None
                    current_segment = segment
                    current_speaker = segment[1]
        if current_segment:
            final_log.append(current_segment) if check_length_segment(current_segment) else None

        # Step 3: Remove segments that have speaker talk total time less then 8%
        dct = {}
        total_time = 0
        for segment in final_log:
            start, end = segment[0]
            speaker = segment[1]
            if speaker not in dct:
                dct[speaker] = 0
            dct[speaker] += end - start
            total_time += end - start
        for speaker in list(dct.keys()):
            if dct[speaker] / total_time < 0.08:
                for segment in final_log:
                    if segment[1] == speaker:
                        final_log.remove(segment)

        return final_log

    logger.info("Start remove collisions from diarization results...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        cleaned_diarys = list(executor.map(remove_collision, diarys))
    logger.info("Removed collisions and concat audio from diarization results")

    if is_save:
        logger.info("Saving cleaned diarization results...")
        os.makedirs('./01_cleaned_diarization', exist_ok=True)
        os.makedirs(f'./01_cleaned_diarization/{args.playlist_name}', exist_ok=True)
        for i in range(len(diarys)):
            diary = cleaned_diarys[i]
            name = episode_name[i]
            with open(os.path.join(f'./01_cleaned_diarization/{args.playlist_name}', f'{name}.pkl'), 'wb') as f:
                pickle.dump(diary, f)
        logger.info("Saved cleaned diarization results")
    return cleaned_diarys

def save_sub_audio(results, clean_diary):
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
        speaker = segment[1]
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        sub_audio = result["waveform"][start_sample:end_sample]
        sub_audio_path = f"./03_concat_audio/{args.playlist_name}/{os.path.basename(result['name']).rsplit('.', 1)[0]}/{speaker}_{start}_{end}.wav"
        sf.write(sub_audio_path, sub_audio, sr)

    logger.info("Saving sub-audio segments based on cleaned diarization results...")


    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        os.makedirs('./03_concat_audio', exist_ok=True)
        os.makedirs(f'./03_concat_audio/{args.playlist_name}', exist_ok=True)
        for result, diary in zip(results, clean_diary):
            if os.path.exists(os.path.join(f'./03_concat_audio/{args.playlist_name}', os.path.basename(result["name"]).rsplit('.', 1)[0])):
                print("Already have it, skipping")
                continue

            # Create directory for each audio file
            os.makedirs(os.path.join(f'./03_concat_audio/{args.playlist_name}', os.path.basename(result["name"]).rsplit('.', 1)[0]), exist_ok=True)
            for segment in diary:
                futures.append(executor.submit(process_segment, result, segment, sr))
        for future in as_completed(futures):
            future.result()  # Ensure all tasks are completed
    logger.info("Saved sub-audio segments based on cleaned diarization results")


def process_denoise(model, df_state, episode_name):
    print(f"Start denoising {episode_name}")
    episode_path        = os.path.join(f"./03_concat_audio/{args.playlist_name}", f"{episode_name}")
    save_episode_path   = os.path.join(f"./04_denoise/{args.playlist_name}", f"{episode_name}")

    if os.path.exists(save_episode_path):
        print(f"Skip {episode_name}")
        return 0

    os.makedirs(os.path.join(f"./04_denoise/{args.playlist_name}"), exist_ok=True)
    os.makedirs(save_episode_path, exist_ok=True)
    for wav_file in os.listdir(episode_path):
        audio_path = os.path.join(episode_path, wav_file)
        save_path = os.path.join(save_episode_path, wav_file)
        # print(audio_path)
        audio, _ = load_audio(audio_path, sr=df_state.sr())
        # Denoise the audio
        enhanced = enhance(model, df_state, audio)
        # Save for listening
        save_audio(save_path, enhanced, df_state.sr())
    logger.debug(f"Finish denoising {episode_name}")
    return 0

def denoise(audio_files):
    """
    Denoise audio files using DeepFilterNet.

    Args:
        audio_files (list): List of audio file paths to be denoised.
    """
    logger.info("Loading DeepFilterNet model...")
    model, df_state, _, _ = init_df()
    os.makedirs("./04_denoise", exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
        for episode in audio_files:
            save_episode_path = os.path.join(f"./04_denoise/{args.playlist_name}", episode)
            executor.submit(process_denoise, model, df_state, episode)
    logger.info(f"Denoise loaded successfully {audio_files}")

    # clean gpu
    torch.cuda.empty_cache()

def calculate_normalized_mel_mse(audio_raw_path, audio_denoised_path, sr=22050, n_fft=2048, hop_length=512):
    # Load và xử lý audio như trước
    y_raw, _ = librosa.load(audio_raw_path, sr=sr)
    y_denoised, _ = librosa.load(audio_denoised_path, sr=sr)
        
    min_len = min(len(y_raw), len(y_denoised))
    y_raw = y_raw[:min_len]
    y_denoised = y_denoised[:min_len]
    
    # Tính Mel-spectrogram
    mel_raw = librosa.feature.melspectrogram(y=y_raw, sr=sr, n_fft=n_fft, 
                                          hop_length=hop_length)
    mel_denoised = librosa.feature.melspectrogram(y=y_denoised, sr=sr, n_fft=n_fft, 
                                               hop_length=hop_length)
    
    # Chuyển sang dB scale
    mel_raw_db = librosa.power_to_db(mel_raw, ref=np.max)
    mel_denoised_db = librosa.power_to_db(mel_denoised, ref=np.max)
    
    # Tính MSE chuẩn hóa (trung bình trên tất cả các frames và mel bands)
    mse_value = np.mean((mel_raw_db - mel_denoised_db) ** 2)
    return mse_value

def process_file(args):
    episode_name, file, raw, denoised = args
    try:
        audio_raw_path = os.path.join(raw, episode_name, file)
        audio_denoised_path = os.path.join(denoised, episode_name, file)
        return episode_name, file, calculate_normalized_mel_mse(audio_raw_path, audio_denoised_path)
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return None

def split_by_denoise(episodes_name):
    os.makedirs('./04_denoise_extract', exist_ok=True)
    os.makedirs(f'./04_denoise_extract/{args.playlist_name}', exist_ok=True)
    raw         = f"./03_concat_audio/{args.playlist_name}"
    denoised    = f"./04_denosie/{args.playlist_name}"
    
    for episode_name in episodes_name:
        files = [f for f in os.listdir(os.path.join(raw, episode_name)) if f.endswith('.wav')]
        args_list = [(episode_name, file, raw, denoised) for file in files]

        with ProcessPoolExecutor(max_workers=2, initializer=limit_cpu_for_diarization) as executor:
            results = list(executor.map(process_file, args_list)) #[(episode_name, file, mse_value), ...]
        # Save results
        dct = {}
        for result in results:
            if result[0] not in dct:
                dct[result[0]] = {result[1]: result[2]}
            else:
                dct[result[0]][result[1]] = result[2]
        output_path = os.path.join(f'./04_denoise_extract/{args.playlist_name}', f'{episode_name}.json')
        if not os.path.exists(output_path):
            with open(output_path, 'w') as f:       
                json.dump(dct, f, ensure_ascii=False, indent=2)
                
def limit_cpu_for_diarization():
    # return
    """Giới hạn process và các subprocess chỉ chạy trên core 0 và 1"""
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity([i for i in range(32, 40)])  # Chỉ dùng core 0 và 1
        print(f"Process {os.getpid()} bị giới hạn trên core: {p.cpu_affinity()}")
    except Exception as e:
        print(f"Không thể thiết lập cpu_affinity: {e}")




def create_audio(waveform, episode, speaker, start, end, text, sample_rate):
    """
    Create audio segment from waveform and save it as a .wav file.

    Args:
        waveform (np.ndarray): The audio waveform to be saved.
        episode (str): The episode name for the output file.
        speaker (str): The speaker name for the output file.
        start (float): Start time of the segment in seconds.
        end (float): End time of the segment in seconds.
        text (str): Text to be saved in the .lab file.
        sample_rate (int): Sample rate of the audio.
    """
    start = round(start, 2)
    end = round(end, 2)
    save_path = f'./05_transcript/{args.playlist_name}/{episode}'
    os.makedirs(save_path, exist_ok=True)
    sf.write(f'{save_path}/{speaker} - {start} - {end}.wav', waveform, sample_rate)
    
    # Create lab file
    with open(f'{save_path}/{speaker} - {start} - {end}.lab', 'w') as f:
        f.write(text)

def transcript(episodes_name, is_save=False):
    os.makedirs('./05_transcript', exist_ok=True)
    os.makedirs(f'./05_transcript/{args.playlist_name}', exist_ok=True)
    threads = 8
    batch_size = 32  # reduce if low on GPU memory
    chunk_size = 15  # inference with chunks of 16 seconds. Default is 30
    compute_type = "float16"  # change to "int8" if low on GPU memory (may reduce accuracy)
    model_name = "mad1999/pho-whisper-large-ct2" #FIXME: Test with pho-whisper-large
    model = whisperx.load_model(model_name, 'cuda',
                                device_index=args.cuda_id,
                                compute_type=compute_type, language='vi',
                                threads=threads,
                                asr_options={"beam_size":10,
                                            #"repetition_penalty":100,
                                            #"no_repeat_ngram_size": 1,
                                            #'condition_on_previous_text':False,
                                            #"initia   l_prompt": "<|vi|>",
                                            #"word_timestamps": True,
                                            #"prefix": "<|0.24|>",
                                            #"temperatures": [0.6, 0.7, 0.8, 0.9, 1.0],
                                            "suppress_numerals": True})
    os.makedirs(f"./05_transcript/{args.playlist_name}", exist_ok=True)
    for episode in episodes_name:
        audio_folder = os.path.join(f"./04_denoise/{args.playlist_name}", episode)
        transcription_folder = os.path.join(f"./05_transcript/{args.playlist_name}", episode)

        if os.path.exists(transcription_folder):
            print("Already have it, skipping")
            continue

        os.makedirs(transcription_folder, exist_ok=True)
        SAMPLE_RATE = 16000
        for audio_file in tqdm(os.listdir(audio_folder)[::-1]):
            try:
                if audio_file.endswith(".wav") or True: # audios with .opus extension
                    audio_path = os.path.join(audio_folder, audio_file)

                    # Extract key
                    key = audio_file[audio_file.rfind('[') + 1:audio_file.rfind(']')]
                    if len(key) != 11:
                        key = os.path.basename(audio_path)
                    file_name = f"{key}.json"
                    file_path = os.path.join(transcription_folder, file_name)
                    if os.path.exists(file_path):
                        print("Already have it, skipping")
                        continue
                    # Loading audio
                    audio = whisperx.load_audio(audio_path) #NOTE: SAMPLE_RATE (default) = 16000

                    # Transcribing
                    result = model.transcribe(audio, batch_size=batch_size, chunk_size=chunk_size)
                    # {"segments": [{"text":, "start":, "end":},...], "language": "vi"}
                    # print(result)
                    # Store results in individual JSON files
                    if is_save:
                        with open(file_path, 'w', encoding='utf-8') as json_file:
                            print(json_file)
                            json.dump(result, json_file, ensure_ascii=False, indent=2)
                    
                    segments = result['segments']
                    begin_audio = float(audio_file.split('_')[2])
                    for seg in segments:
                        start = seg['start']+begin_audio
                        end = seg['end']+begin_audio
                        text = seg['text']
                        
                        create_audio(waveform=audio[int(seg['start'] * SAMPLE_RATE): int(seg['end']*SAMPLE_RATE)], episode=episode, speaker=audio_file[:10], start=start, end=end, text=text, sample_rate=SAMPLE_RATE)
                    logger.info(f"Transcribed {audio_file}")
                    
                    
            except Exception as e:
                print(audio_file, "failed")
                print(e)
        # clean gpu
        torch.cuda.empty_cache()

def create_matadata_embeddings(emebeddings, episode_name):
    """
    Create metadata embeddings for the given audio files.

    Args:
        emebeddings (list): List of embeddings for each audio file.
        episode_name (list): List of episode names corresponding to the embeddings.
    """
    os.makedirs('./02_embeddings', exist_ok=True)
    os.makedirs(f'./02_embeddings/{args.playlist_name}', exist_ok=True)
    for i in range(len(emebeddings)):
        name = episode_name[i]
        with open(os.path.join(f'./02_embeddings/{args.playlist_name}', f'{name}.pkl'), 'wb') as f:
            pickle.dump(emebeddings[i].tolist(), f)
    logger.info("Saved metadata embeddings")

def label():
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
    
    sorted_files = sort_files_by_audio_length(os.path.join('/home4/tuanlha/DataTest', cfg["playlist_name"]))

    pattern = re.compile(r'\[.*?\]')

    for index, filename in enumerate(sorted_files, start=1):
        # Tạo tên mới với số đánh số ở đầu
        new_name = f"{index:02d} {filename}"

        # Xóa phần [*] của tên file
        new_name = re.sub(pattern, '', new_name).strip()

        # Đường dẫn đầy đủ của file cũ và file mới
        old_file = os.path.join('/home4/tuanlha/DataTest', cfg["playlist_name"], filename)
        new_file = os.path.join('/home4/tuanlha/DataTest', cfg["playlist_name"], new_name)

        # Đổi tên file
        os.rename(old_file, new_file)
        print(f"Đã đổi tên {filename} thành {new_name}")

# def filter_audio(episode_name):
#     """
#     1. Loại đi các audio mà có speaker chiếm tổng dưới 2.5% 

#     Args:
#         episode_name (list): List of episode names to be filtered.
#     """
def process_nisqa(audio_folder_path):
    command = f"python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar --data_dir '{audio_folder_path}' --num_workers 8 --bs 30 --output_dir /home4/tuanlha/EXpressiveTTS/dataRawProcess/06_data_extract"
    print(command)
    subprocess.run(command, shell=True, cwd="/home2/tuannd/tuanlha/NISQA")

def nisqa(episodes_name):
    def run_predict(audio_folder_path):
        os.makedirs(f'/home4/tuanlha/EXpressiveTTS/dataRawProcess/05_data_extract/{args.playlist_name}', exist_ok=True)
        command = f"taskset -c 36,37,38,39 python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar --data_dir '{audio_folder_path}' --num_workers 8 --bs 30 --output_dir /home4/tuanlha/EXpressiveTTS/dataRawProcess/05_data_extract/{args.playlist_name}"
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

    folder_path = './05_transcript/' + args.playlist_name
    episodes = [dir for dir in os.listdir(folder_path)]
    with ThreadPoolExecutor(max_workers=2, initializer=limit_cpu_for_diarization) as executor:
        for cnt, episode in enumerate(os.listdir(folder_path)):
            executor.submit(run_predict, os.path.join(folder_path, episode))    

# def nisqa_filter(episodes_name):


def main_process():

    if args.label_audio:
        label()

    # audio_files = sorted([os.path.join(cfg["source_path"], cfg["playlist_name"], ep) 
    #               for ep in os.listdir(os.path.join(cfg["source_path"], cfg["playlist_name"]))])
    # # print(audio_files)

    # # Process by batch
    # batch_size = 4
    # for i in range(0, len(audio_files), batch_size):
    #     batch = audio_files[i:i+batch_size] #batch: full name path to source
    #     episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in batch]
    #     print(episode_name)
    #     results = standarlization_audio(batch, is_save=True)

    #     #TODO: xóa nhạc nền

    #     diary_with_embeddings = speaker_diarization(results, episode_name, is_save=True) # [{diary, embeddings}, {diary, embeddings}, ...]

    #     # emebeddings = [diary["embeddings"] for diary in diary_with_embeddings]
    #     #NOTE bo buoc nay create_matadata_embeddings(emebeddings, episode_name)

    #     diarys = [diary["diary"] for diary in diary_with_embeddings]
    #     cleaned_diary = clean_diary(diarys, episode_name, is_save=True) # [[[start, end], speaker], [[start, end], speaker], ...]

    #     save_sub_audio(results, cleaned_diary)

    #     denoise(episode_name)

    #     # sileroVAD.py

    #     # cosine_pair.py

    #     # remove.py

        

    #     # standarlize audio vì rè

    #     split_by_denoise(episode_name)

    #     transcript(episode_name, is_save=True)

    #     #bonus (?) filter_transcipt(episode_name)
        
    #     #TODO Kiem tra ngu nghia cua cau 
    #     # # filter_audio(episode_name)
    #     nisqa(episode_name)

        

        #metrix nisqa + wvmos
        # break


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--playlist_name",
        type=str,
        default="",
        help="input folder path, this will override config if set",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config.json",
        help="path to the config file"
    )
    parser.add_argument(
        "--cuda_id",
        type=int,
        default=0,
        help="CUDA device ID to use"
    )
    parser.add_argument(
        "--label_audio",
        type=bool,
        default=False,
        help="Whether to label audio files"
    )

    logger = Logger.get_logger()

    args = parser.parse_args()
    
    # Load the config file
    cfg = load_cfg(args.config_path)

    if args.playlist_name:
        cfg["playlist_name"] = args.playlist_name

    # print("Load models"):
    if detect_gpu():
        print("GPU detected")
        device_name = cfg["device"]
        device_id = args.cuda_id
        device = torch.device(f"{device_name}:{device_id}")
    else:
        print("No GPU detected, using CPU")
        device = torch.device("cpu")

    main_process()