import audio_processing.config
# from audio_processing.audio_utils import standardize_audio, save_sub_audio
# from audio_processing.diarization import diarization_results, clean_diarization_results, process_clean_diary, process_audio_diarization
# from audio_processing.diarization import clean_diarization_results
# from audio_processing.sileroVADcopy import vad, remove_and_rename, create_sub_vad
# from audio_processing.denoising import denoise
# from audio_processing.rating_audio import rating_audio, filter_by_rating
from audio_processing.transcript import transcript, filter_by_transcript

# from utils.manage_memory import set_cpu_affinity
from utils.tool import safe_name
import os

args = audio_processing.config.parse_args()
cfg = audio_processing.config.load_cfg(args.config_path)

cuda_id = args.cuda_id

if args.label_audio:
    safe_name(args.data_path, args.playlist_name)

# process_audio_diarization(args, cfg)

# clean_diarization_results(args, cfg)

# vad(args, cfg)

# create_sub_vad(args, cfg)

# denoise(args, cfg)

# rating_audio(args, cfg)

# filter_by_rating(args, cfg)

# transcript(args, cfg)

filter_by_transcript(args, cfg)