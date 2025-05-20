
import os, tqdm
import json
from utils.logger import Logger
import whisperx
import soundfile as sf
import torch
import shutil

logger = Logger()

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
    start = round(start, 2)
    end = round(end, 2)
    save_path = f'./05_transcript/{playlist_name}/{episode}'
    os.makedirs(save_path, exist_ok=True)
    sf.write(f'{save_path}/{file_name}.wav', waveform, sample_rate)
    
    # Create lab file
    with open(f'{save_path}/{file_name}.lab', 'w') as f:
        f.write(text)


def transcript(episodes_name, playlist_name, cuda_id, is_save=False):
    os.makedirs('./05_transcript', exist_ok=True)
    os.makedirs(f'./05_transcript/{playlist_name}', exist_ok=True)
    threads = 8
    batch_size = 32  # reduce if low on GPU memory
    chunk_size = 15  # inference with chunks of 16 seconds. Default is 30
    compute_type = "float16"  # change to "int8" if low on GPU memory (may reduce accuracy)
    model_name = "mad1999/pho-whisper-large-ct2" #FIXME: Test with pho-whisper-large
    model = whisperx.load_model(model_name, 'cuda',
                                device_index=cuda_id,
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
    for episode in episodes_name:
        episode_folder = os.path.join(f"./04_vad/{playlist_name}", episode)
        transcription_folder = os.path.join(f"./05_transcript/{playlist_name}", episode)

        if os.path.exists(transcription_folder):
            print("Already have it, skipping")
            continue

        os.makedirs(transcription_folder, exist_ok=True)
        SAMPLE_RATE = 16000
        for audio_segment in os.listdir(episode_folder):
            audio_folder = os.path.join(episode_folder, audio_segment)
            for audio_file in os.listdir(audio_folder):
                try:
                    if audio_file.endswith(".wav") or True: # audios with .opus extension
                        audio_path = os.path.join(audio_folder, audio_file)

                        # Extract key
                        key = audio_file[audio_file.rfind('[') + 1:audio_file.rfind(']')]
                        if len(key) != 11:
                            key = os.path.basename(audio_path)
                        file_name = f"{audio_segment}_{key}.json"
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
                        print("Full Transcript:", all_text)
                        file_name = f"{audio_segment}_{key[:-4]}"
                        create_audio(waveform=audio, episode=episode, file_name=file_name, text=all_text, sample_rate=SAMPLE_RATE, playlist_name=playlist_name)
                        logger.info(f"Transcribed {audio_file}")
                        
                        
                except Exception as e:
                    print(audio_file, "failed")
                    print(e)
        # clean gpu
        torch.cuda.empty_cache()

def restruct_folder(episodes_name, playlist_name):
    current_path = f'./04_vad/{playlist_name}'
    for episode in episodes_name:
        episode_path = os.path.join(current_path, episode)

        for segment in os.listdir(episode_path):
            segment_path = os.path.join(episode_path, segment)
            for audio_file in os.listdir(segment_path):
                if audio_file.endswith(".wav"):
                    src_path = os.path.join(segment_path, audio_file)
                    dst_path = os.path.join(current_path, episode, f'{segment}_{audio_file}')
                    shutil.move(src_path, dst_path)
            os.rmdir(segment_path)