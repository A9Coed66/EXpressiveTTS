from audio_processing.audio_utils import standardize_audio, save_sub_audio
from audio_processing.diarization import diarization_results, clean_diarization_results, process_clean_diary
import audio_processing.config
from audio_processing.denoising import denoise
from audio_processing.sileroVADcopy import vad, remove_and_rename, create_sub_vad
from audio_processing.cosine_pair import compare, remove, saiba_momoi,saibamomoi_2
from audio_processing.transcript import transcript, restruct_folder
from audio_processing.rating_audio import rating_audio

from utils.manage_memory import set_cpu_affinity
from utils.tool import label
import os

args = audio_processing.config.parse_args()
cfg = audio_processing.config.load_cfg(args.config_path)

cuda_id = args.cuda_id

if args.label_audio:
    label(args.data_path, args.playlist_name)

playlist_path = os.path.join(args.data_path, args.playlist_name)
episode_list = sorted(os.listdir(playlist_path))

batch_size = 4
for i in range(0, len(episode_list), batch_size):
    batch = episode_list[i:i+batch_size] #batch: full name path to source

    # Get list episode name
    episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in batch]

    episode_path = [os.path.join(playlist_path, ep) for ep in batch]

    # Standardize audio
    results = standardize_audio(audio_files=episode_path,
                                cfg=cfg,
                                playlist_name=args.playlist_name,
                                is_save=False)

    # Get diarization results
    diary_results = diarization_results(results=results,
                                              cfg=cfg,
                                              device_id=cuda_id)
    diary = [item["diary"] for item in diary_results]
    cleaned_diary = clean_diarization_results(diary_results=diary, cfg=0)

    # Create sub audio and denoise
    save_sub_audio(results, cleaned_diary, args.playlist_name, cfg)
    # denoise(episode_name, args.playlist_name)
    # #NOTE: can delete save sub audio folder after denoising

    # # VAD, extract vad, and remove audio have more than 2 speaker
    # denoise_path = [f'./04_denoise/{args.playlist_name}/{episode}' for episode in episode_name]
    # vad(episode_name, args.playlist_name)
    # create_sub_vad(episode_name, args.playlist_name)
    # rating_audio(episode_name, args.playlist_name)

    # for path in denoise_path:
    #     os.makedirs(f'./04_vad/{args.playlist_name}/{os.path.basename(path)}', exist_ok=True)
    #     os.makedirs(f'./04_vad_extract/{args.playlist_name}', exist_ok=True)
    #     os.makedirs(f'./05_similarity/{args.playlist_name}', exist_ok=True)
    #     compare(wav_dir=f'./04_vad/{args.playlist_name}/{os.path.basename(path)}', file_csv=f'./05_similarity/{args.playlist_name}/{os.path.basename(path)}.csv')
    #     remove(args.playlist_name, f'{os.path.basename(path)}', 0.3)

    # restruct_folder(episode_name, args.playlist_name)


    # transcript(episode_name, args.playlist_name, cuda_id, is_save=True)
# saiba_momoi()
# saibamomoi_2()