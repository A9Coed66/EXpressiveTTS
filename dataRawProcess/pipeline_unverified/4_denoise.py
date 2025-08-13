from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
from concurrent.futures import ThreadPoolExecutor
import argparse
import os

parser = argparse.ArgumentParser(description="Denoise audio files")
parser.add_argument("--data_path",type=str, default="/home4/tuanlha/DataProcessStep/03_concat_audio", help="Directory containing audio files to denoise")
parser.add_argument("--save_path", type=str, default="/home4/tuanlha/DataProcessStep/04_denoise", help="Directory to save denoised audio files")
parser.add_argument("--playlist_name", type=str, default="Temp", help="Playlist name")

args = parser.parse_args()

def denoise(model, df_state, episode_name):
    print(f"Start denoising {episode_name}")
    episode_path        = os.path.join(args.data_path, args.playlist_name, episode_name)
    save_episode_path   = os.path.join(args.save_path, args.playlist_name, episode_name)
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
    print(f"Finish denoising {episode_name}")
    return 0

if __name__ == "__main__":
    # Load default model
    model, df_state, _, _ = init_df()
    # Download and open some audio file. You use your audio files here
    audio_path = download_file(
        "https://github.com/Rikorose/DeepFilterNet/raw/e031053/assets/noisy_snr0.wav",
        download_dir=".",
    )
    # audio, _ = load_audio(audio_path, sr=df_state.sr())
    # # Denoise the audio
    # enhanced = enhance(model, df_state, audio)
    # # Save for listening
    # save_audio("enhanced.wav", enhanced, df_state.sr())
    data_dir = os.path.join(args.data_path, args.playlist_name)
    os.makedirs(data_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for episode in os.listdir(data_dir):
            save_episode_path = os.path.join(args.save_path, args.playlist_name, episode)
            if os.path.exists(save_episode_path):
                print(f"Skip {episode}")
                continue
            os.makedirs(os.path.join(args.save_path, args.playlist_name, episode), exist_ok=True)
            executor.submit(denoise, model, df_state, episode)