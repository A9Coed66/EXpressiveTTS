"""
1. Chunk audio
    output: 1. chunked audio: audio file
2. Denoise audio
    output: 2.denoise audio: audio file
3. Diarization
    output: 3. diary: dataframe [columns: start, end, speaker]
4. Speaker labeled from diary logs
    output: 4. speaker labeled: dataframe [columns: audio_path, diary_label, model_label, score, verified]
5. Concat near audio files to one file
    output: 5. concatenated audio:  dataframe [columns: start. end, speaker]
                                    audio file
6. Audio speech recognition:
    output: 6. transcript: dataframe [columns: audio_path, transcript, speaker]
"""
import argparse
### 0. setup data

ap = argparse.ArgumentParser()
ap.add_argument("-ap", "--audio_path", type=str, default=0,
    help="path of audio file")
ap.add_argument("-fn", "--folder_name", type=str)
args = vars(ap.parse_args())

### 1. Chunk audio
from split_audio import split_audio
from denoise import denoise
from create_diarization import create_diarization
from remove_collision import remove_collision, split_diarization
from verify_speaker import verify_speaker
from concat_audio import concat_audio




if __name__ == "__main__":
    audio_path = args['audio_path']
    if args['folder_name']:
        folder_name = args['folder_name']
    else:
        folder_name = audio_path.split('/')[-1]

    split_audio(audio_path, folder_name)    # Step 1
    denoise(folder_name)                    # Step 2
    create_diarization(folder_name)         # Step 3
    remove_collision(folder_name)
    #TODO: fix split_diarization
    split_diarization(folder_name)
    verify_speaker(folder_name)             # Step 4
    concat_audio(folder_name)               # Step 5    
    # TODO: create transcript func
    transcript(folder_name)

