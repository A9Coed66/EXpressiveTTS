import os
import json
from tqdm import tqdm
from time import time
import whisperx

# Initialize an empty dictionary to store results
all_results = {}

# audio_folder = "/home3/nguyenlm/Fonos_Corpus/data_all"  # Update this path to your audio folder
# audio_folder = "/home4/hainh/fonos/data_radio_station02" # Update this path to your audio folder
# transcription_folder = "transcription_RS1" # Update this path to your transcription folder
# audio_folder = os.environ.get("AUDIO_FOLDER")
# transcription_folder = os.environ.get("TRANSCRIPTION_FOLDER")
root_audio_folder = "/home2/tuannd/tuanlha/EXpressiveTTSDataset/data_unverified/02_concat_audio"
transcription_folder = "/home2/tuannd/tuanlha/EXpressiveTTSDataset/data_unverified/03_transcription"

device = "cuda"
threads = 8
batch_size = 12  # reduce if low on GPU memory
chunk_size = 8  # inference with chunks of 16 seconds. Default is 30
compute_type = "float16"  # change to "int8" if low on GPU memory (may reduce accuracy)
model_name = "mad1999/pho-whisper-large-ct2"
model = whisperx.load_model(model_name, device,
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

if not os.path.exists(transcription_folder):
    os.makedirs(transcription_folder,exist_ok=True)

def check_exist(episode, folder):
    '''Check if diary logs already exist
    exist:      return True
    not exist:  return False'''
    current_episode = [file_name for file_name in os.listdir(folder)]
    for ep in current_episode:
        if episode in ep:
            return True
    return False
stime = time()
for episode in os.listdir(root_audio_folder):


    if check_exist(episode, transcription_folder):
        print(f"Skip {episode}")
        continue
    print(f'Starting episode {episode}')
    audio_folder = os.path.join(root_audio_folder, episode)
    os.makedirs(os.path.join(transcription_folder, episode), exist_ok=True)
    for audio_file in tqdm(os.listdir(audio_folder)[::-1]): # run this script multiple times with different gpu and slice to enable multiprocessing, example [:100], [100:]
        # print(audio_file)
        try:
            if audio_file.endswith(".wav") or True: # audios with .opus extension
                audio_path = os.path.join(audio_folder, audio_file)

                # Extract key
                key = audio_file[audio_file.rfind('[') + 1:audio_file.rfind(']')]
                if len(key) != 11:
                    key = os.path.basename(audio_path)
                file_name = f"{key}.json"
                file_path = os.path.join(transcription_folder, episode, file_name)
                if os.path.exists(file_path):
                    print("Already have it, skipping")
                    continue
                # Loading audio
                audio = whisperx.load_audio(audio_path)

                # Transcribing
                result = model.transcribe(audio, batch_size=batch_size, chunk_size=chunk_size)
                #print(result)
                # Store results in individual JSON files
                with open(file_path, 'w', encoding='utf-8') as json_file:
                    print(json_file)
                    json.dump(result, json_file, ensure_ascii=False, indent=2)
        except Exception as e:
            print(audio_file, "failed")
            print(e)

print('Transcription time: ', time() - stime)