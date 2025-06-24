from utils.logger import Logger
import os
from df.enhance import enhance, init_df, load_audio, save_audio
from concurrent.futures import ThreadPoolExecutor
import torch

logger = Logger.get_logger()

def process_denoise(model, df_state, episode_name, playlist_name):
    print(f"Start denoising {episode_name}")
    episode_path        = os.path.join(f"./03_vad/{playlist_name}", f"{episode_name}")
    save_episode_path   = os.path.join(f"./04_denoise/{playlist_name}", f"{episode_name}")

    # if os.path.exists(save_episode_path):
    #     print(f"Skip {episode_name}")
    #     return 0

    os.makedirs(os.path.join(f"./04_denoise/{playlist_name}"), exist_ok=True)
    os.makedirs(save_episode_path, exist_ok=True)
    for wav_file in os.listdir(episode_path):
        audio_path = os.path.join(episode_path, wav_file)
        save_path = os.path.join(save_episode_path, wav_file)
        if os.path.exists(save_path):
            logger.debug(f"Skip {wav_file} in {episode_name}")
            continue
        # print(audio_path)
        audio, _ = load_audio(audio_path, sr=df_state.sr())
        # Denoise the audio
        enhanced = enhance(model, df_state, audio)
        # Save for listening
        save_audio(save_path, enhanced, df_state.sr())
    logger.debug(f"Finish denoising {episode_name}")
    return 0

# def denoise(audio_files, playlist_name):
#     """
#     Denoise audio files using DeepFilterNet.

#     Args:
#         audio_files (list): List of audio file paths to be denoised.
#     """
#     logger.info("Loading DeepFilterNet model...")
#     model, df_state, _, _ = init_df()
#     os.makedirs("./04_denoise", exist_ok=True)
#     with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
#         for episode in audio_files:
#             save_episode_path = os.path.join(f"./04_denoise/{playlist_name}", episode)
#             executor.submit(process_denoise, model, df_state, episode, playlist_name)
#     logger.info(f"Denoise loaded successfully {audio_files}")

#     # clean gpu
#     torch.cuda.empty_cache()

def denoise(args, cfg):
    """
    Denoise audio files using DeepFilterNet.

    Args:
        audio_files (list): List of audio file paths to be denoised.
    """
    logger.info("Loading DeepFilterNet model...")
    model, df_state, _, _ = init_df()
    episode_list = sorted(os.listdir(os.path.join('./00_standardization', args.playlist_name)))
    episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in episode_list]

    os.makedirs("./04_denoise", exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(2, os.cpu_count())) as executor:
        for episode in episode_name:
            save_episode_path = os.path.join(f"./04_denoise/{args.playlist_name}", episode)
            executor.submit(process_denoise, model, df_state, episode, args.playlist_name)
    logger.info(f"Denoise loaded successfully")

    # clean gpu
    torch.cuda.empty_cache()