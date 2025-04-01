import argparse
import json
from pprint import pprint
from utils.tool import (
    export_to_mp3,
    load_cfg,
    get_audio_files,
    detect_gpu,
    check_env,
    calculate_audio_stats,
)
from pyannote.audio import Pipeline
import time
from utils.logger import Logger, time_logger
import torch, os
from pydub import AudioSegment
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import pickle

def process_batch(audio_files):
    """Xử lý song song nhiều file audio cùng lúc"""
    max_threads = 4
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        results = list(executor.map(standardization, audio_files))
        return results

def standardization(audio):
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object, the audio to be preprocessed.

    Returns:
        dict: A dictionary containing the preprocessed audio waveform, audio file name, and sample rate, formatted as:
              {
                  "waveform": np.ndarray, the preprocessed audio waveform, dtype is np.float32, shape is (num_samples,)
                  "name": str, the audio file name
                  "sample_rate": int, the audio sample rate
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
    # elif isinstance(audio, AudioSegment):
    #     name = f"audio_{audio_count}"
    #     audio_count += 1
    else:
        raise ValueError("Invalid audio type")

    logger.debug("Entering the preprocessing of audio")

    # Convert the audio file to WAV format
    audio = (audio.set_frame_rate(cfg["save_step"]["standardization"]["sample_rate"])).set_sample_width(2).set_channels(1)
    # audio = audio.set_sample_width(2)  # Set bit depth to 16bit
    # audio = audio.set_channels(1)  # Set to mono

    logger.debug("Audio file converted to WAV format")

    # Calculate the gain to be applied
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    # Normalize volume and limit gain range to between -3 and 3
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

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
        "name": name,
        # "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"],
    }

def speaker_diarization(results):
    """
    Perform speaker diarization on the given audio.
    Returns diarization results and ensures model is cleared from memory.
    """
    logger.info("Loading diarization model...")
    secret = "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"
    
    try:
        # Load model
        logger.info("Loading diarization model...")
        diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=secret).to(torch.device(f"cuda:2"))
        logger.info("Diarization model loaded successfully")
        diary = []
        for result in results:
            waveform = torch.from_numpy(result['waveform']).unsqueeze(0)
            diary.append(diarization({'waveform': waveform, 'sample_rate': cfg["save_step"]["standardization"]["sample_rate"]}))
            logger.info(f"Processed file: {result['name']}")
            # Save the diarization result
            with open(f'./{os.path.basename(result["name"])}.pkl', 'wb') as f:
                pickle.dump(diary, f)
        
        return diary
    
    finally:
        # Giải phóng bộ nhớ
        if 'diarization' in locals():
            del diarization
            torch.cuda.empty_cache()
            logger.info("Diarization model cleared from GPU memory")
        






def main_process():
    audio_files = sorted([os.path.join(cfg["source_path"], cfg["playlist_name"], ep) 
                  for ep in os.listdir(os.path.join(cfg["source_path"], cfg["playlist_name"]))])
    print(audio_files)
    # Xử lý theo batch (ví dụ: mỗi batch 10 files)
    batch_size = 4
    for i in range(0, len(audio_files), batch_size):
        start_batch_time = time.time()
        batch = audio_files[i:i+batch_size]
        results = process_batch(batch)
        end_batch_time = time.time()
        elapsed_batch_time = end_batch_time - start_batch_time
        logger.info(f"Batch {i//batch_size + 1} processed in {elapsed_batch_time:.2f} seconds")

        diary = speaker_diarization(results)


if __name__ == "__main__":
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
        device_id = cfg["device_id"]
        device = torch.device(f"{device_name}:{device_id}")
    else:
        print("No GPU detected, using CPU")
        device = torch.device("cpu")

    main_process()