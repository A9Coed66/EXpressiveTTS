import argparse
# import gc
# from split_audio import split
# from denoise import denoise   
# from create_diarization import create_diarization
# from remove_collision import remove_collision, split_diarization
from verify_speaker import verify_speaker
from concat_audio import concat_audio_verified
from transcript import trans

ap = argparse.ArgumentParser()
ap.add_argument("-ap", "--audio_path", type=str, default=0,
    help="path of audio file")
ap.add_argument("-fn", "--folder_name", type=str)
args = vars(ap.parse_args())

if __name__ == "__main__": 
    audio_path = args['audio_path']
    folder_name = args['folder_name']
    # split(audio_path, folder_name)
    # denoise(folder_name)
    # create_diarization(folder_name)
    # remove_collision(folder_name)
    # split_diarization(folder_name)
    # verify_speaker(folder_name)             # Step 4
    # concat_audio(folder_name)               # Step 5    
    trans(folder_name)