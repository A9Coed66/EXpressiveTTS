
import os, tqdm
import json
import sys
sys.path.append('../')
from utils.logger import Logger
import whisperx
import soundfile as sf
import torch
import shutil
from utils.tool import check_exists
from utils.tool import get_mp3_duration_ffprobe

logger = Logger.get_logger()

def create_audio(waveform, episode, file_name, text, sample_rate, playlist_name):
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
    # start = round(start, 2)
    # end = round(end, 2)
    save_path = f'./06_transcript/{playlist_name}/{episode}'
    os.makedirs(save_path, exist_ok=True)
    # sf.write(f'{save_path}/{file_name}.wav', waveform, sample_rate)
    
    # Create lab file
    with open(f'{save_path}/{file_name}.lab', 'w') as f:
        f.write(text)


# def transcript(episodes_name, playlist_name, cuda_id, is_save=False):
#     os.makedirs('./05_transcript', exist_ok=True)
#     os.makedirs(f'./05_transcript/{playlist_name}', exist_ok=True)
#     threads = 8
#     batch_size = 32  # reduce if low on GPU memory
#     chunk_size = 15  # inference with chunks of 16 seconds. Default is 30
#     compute_type = "float16"  # change to "int8" if low on GPU memory (may reduce accuracy)
#     model_name = "mad1999/pho-whisper-large-ct2" #FIXME: Test with pho-whisper-large
#     model = whisperx.load_model(model_name, 'cuda',
#                                 device_index=cuda_id,
#                                 compute_type=compute_type, language='vi',
#                                 threads=threads,
#                                 asr_options={"beam_size":10,
#                                             #"repetition_penalty":100,
#                                             #"no_repeat_ngram_size": 1,
#                                             #'condition_on_previous_text':False,
#                                             #"initia   l_prompt": "<|vi|>",
#                                             #"word_timestamps": True,
#                                             #"prefix": "<|0.24|>",
#                                             #"temperatures": [0.6, 0.7, 0.8, 0.9, 1.0],
#                                             "suppress_numerals": True})
#     for episode in episodes_name:
#         episode_folder = os.path.join(f"./04_vad/{playlist_name}", episode)
#         transcription_folder = os.path.join(f"./05_transcript/{playlist_name}", episode)

#         if os.path.exists(transcription_folder):
#             print("Already have it, skipping")
#             continue

#         os.makedirs(transcription_folder, exist_ok=True)
#         SAMPLE_RATE = 16000
#         for audio_segment in os.listdir(episode_folder):
#             audio_folder = os.path.join(episode_folder, audio_segment)
#             for audio_file in os.listdir(audio_folder):
#                 try:
#                     if audio_file.endswith(".wav") or True: # audios with .opus extension
#                         audio_path = os.path.join(audio_folder, audio_file)

#                         # Extract key
#                         key = audio_file[audio_file.rfind('[') + 1:audio_file.rfind(']')]
#                         if len(key) != 11:
#                             key = os.path.basename(audio_path)
#                         file_name = f"{audio_segment}_{key}.json"
#                         file_path = os.path.join(transcription_folder, file_name)
#                         if os.path.exists(file_path):
#                             print("Already have it, skipping")
#                             continue
#                         # Loading audio
#                         audio = whisperx.load_audio(audio_path) #NOTE: SAMPLE_RATE (default) = 16000

#                         # Transcribing
#                         result = model.transcribe(audio, batch_size=batch_size, chunk_size=chunk_size)
#                         # {"segments": [{"text":, "start":, "end":},...], "language": "vi"}
#                         # print(result)
#                         # Store results in individual JSON files
#                         if is_save:
#                             with open(file_path, 'w', encoding='utf-8') as json_file:
#                                 print(json_file)
#                                 json.dump(result, json_file, ensure_ascii=False, indent=2)
                        
#                         segments = result['segments']
#                         all_text = " ".join([seg['text'] for seg in segments])
#                         print("Full Transcript:", all_text)
#                         file_name = f"{audio_segment}_{key[:-4]}"
#                         create_audio(waveform=audio, episode=episode, file_name=file_name, text=all_text, sample_rate=SAMPLE_RATE, playlist_name=playlist_name)
#                         logger.info(f"Transcribed {audio_file}")
                        
                        
#                 except Exception as e:
#                     print(audio_file, "failed")
#                     print(e)
#         # clean gpu
#         torch.cuda.empty_cache()

def transcript(args, cfg, is_save=False):
    os.makedirs('./06_transcript', exist_ok=True)
    os.makedirs(f'./06_transcript/{args.playlist_name}', exist_ok=True)
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
    step_path = './06_transcript'
    episode_list = sorted(os.listdir(os.path.join('./00_standardization', args.playlist_name)))
    episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in episode_list]
    episode_name = check_exists(step_path, args.playlist_name, episode_name, type='file')

    for episode in episode_name:
        episode_folder = os.path.join(f"./04_denoise/{args.playlist_name}", episode)
        transcription_folder = os.path.join(f"./06_transcript/{args.playlist_name}", episode)

        if os.path.exists(transcription_folder):
            print("Already have it, skipping")
            continue

        os.makedirs(transcription_folder, exist_ok=True)
        SAMPLE_RATE = 16000
        for audio_segment in os.listdir(episode_folder):
            audio_path = os.path.join(episode_folder, audio_segment)
            try:
                if audio_path.endswith(".wav") or True: # audios with .opus extension
                    x = audio_path

                    # Extract key
                    key = audio_path[audio_path.rfind('[') + 1:audio_path.rfind(']')]

                    file_name = f"{audio_segment.rsplit('.', 1)[0]}"
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
                    all_text = " ".join([seg['text'] for seg in segments])
                    # print("Full Transcript:", all_text)
                    create_audio(waveform=audio, episode=episode, file_name=file_name, text=all_text, sample_rate=SAMPLE_RATE, playlist_name=args.playlist_name)
                    # logger.info(f"Transcribed {audio_file}")
                    
                    
            except Exception as e:
                print(audio_path, "failed")
                print(e)
        # clean gpu
        torch.cuda.empty_cache()


import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_wav(episode_name, wav_name, transcript_path, episode_path):
    wav_path = os.path.join(episode_path, wav_name)
    text_path = os.path.join(transcript_path, episode_name, wav_name.replace('.wav', '.lab'))

    if not os.path.exists(text_path):
        logger.info(f"Transcript not found for {text_path}")
        os.remove(wav_path)
        return
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if len(text) == 0:
        logger.info(f"Empty transcript for {wav_path}, removing audio file.")
        os.remove(wav_path)
        os.remove(text_path)
        return
    duration = get_mp3_duration_ffprobe(wav_path)
    if duration <= 0:
        logger.info(f"Invalid duration for {wav_path}, removing audio file.")
        os.remove(wav_path)
        os.remove(text_path)
        return
    words = text.split()
    words_per_second = len(words) / duration
    if words_per_second > 10:
        logger.info(f"Too many words per second ({words_per_second}) for {wav_path}, removing audio file.")
        os.remove(wav_path)
        os.remove(text_path)
        return

def filter_by_transcript(args, cfg):
    transcript_path = f'./06_transcript/{args.playlist_name}'
    playlist_path = f'./04_denoise/{args.playlist_name}'

    tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:  # Tùy chỉnh số luồng
        for episode_name in os.listdir(playlist_path):
            episode_path = os.path.join(playlist_path, episode_name)
            for wav_name in os.listdir(episode_path):
                if wav_name.lower().endswith('.wav'):
                    tasks.append(executor.submit(process_wav, episode_name, wav_name, transcript_path, episode_path))
        for future in as_completed(tasks):
            future.result()  # Để bắt lỗi nếu có

