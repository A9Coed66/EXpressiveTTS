import os
import torch
import pickle
from pyannote.audio import Pipeline
from utils.logger import Logger
import psutil
from concurrent.futures import ThreadPoolExecutor
import torchaudio
from audio_processing.audio_utils import standardize
logger = Logger.get_logger()

def limit_cpu_cores(cores):
    """Limit process to specific CPU cores"""
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity(cores)
        logger.info(f"Process limited to cores: {cores}")
    except Exception as e:
        logger.error(f"Failed to set CPU affinity: {e}")

def initialize_diarization_model(cuda_id, auth_token):
    """Initialize speaker diarization pipeline"""
    logger.info("Loading diarization model...")
    model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=auth_token
    ).to(torch.device(f"cuda:{cuda_id}"))
    logger.info("Diarization model loaded")
    return model

def perform_diarization(model, waveform, sample_rate):
    """Run diarization on single audio segment"""
    return model({'waveform': waveform, 'sample_rate': sample_rate}, return_embeddings=True)

def diarization_results(results, cfg, device_id):
    """Clean and process diarization results"""
    secret = "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"
    model = initialize_diarization_model(device_id, secret)

    diary_results = []
    logger.info("Performing diarization...")
    for item in results:
        waveform = item['waveform']
        name     = item['name']

        # Get diarization results
        diary, embeddings = perform_diarization(model, torch.from_numpy(waveform).unsqueeze(0).to(f'cuda:{device_id}'), cfg["save_step"]["standardization"]["sample_rate"])
        diary_results.append({"diary": diary, "embeddings": embeddings})
        logger.info(f"Processed file: {name}")
    
    torch.cuda.empty_cache()
    logger.info("Diarization model cleared from GPU memory")
    
    return diary_results

def save_diarization_results(diary_results, playlist_name, basename):
    """Save diarization results to disk"""
    os.makedirs(os.path.join('./00_diarization', playlist_name), exist_ok=True)
    save_path = os.path.join('./00_diarization', playlist_name, f'{basename}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(diary_results, f)
    logger.info(f"Diarization results saved to {save_path}")

def process_clean_diary(diary):
    processed_log = []
    
    queue_segment = []

    conflict = True
    # Step 1: seperate by time
    for turn, _, speaker in diary.itertracks(yield_label=True):
        last_end = queue_segment[0][-1] if queue_segment else 0

        start, end = turn.start, turn.end
        current_segment     = [[start, end], speaker]

        if start > last_end :
            if not conflict and queue_segment:
                processed_log.append(queue_segment)
            conflict = False
            queue_segment = current_segment
            last_end = end
            continue

        if start <= last_end:
            queue_segment=None
            conflict = True
            last_end = max(last_end, end)
    
    if queue_segment:
        processed_log.append(queue_segment)

    #FIXME Step 2: filter minimum segment length
    MIN_SEGMENT_LENGTH = 1
    SILENT_THRESHOLD   = 1
    filter_segments_length = [item for item in processed_log if item[0][1] - item[0][0] > MIN_SEGMENT_LENGTH]
    return filter_segments_length

def clean_diarization_results(diary_results, cfg):
    """Clean and process diarization that each segment has only one speaker
    
    Args:
        diary_results (list): List of diarization results []
        cfg (dict): Configuration dictionary
    """


    with ThreadPoolExecutor(max_workers=2) as executor:
        cleaned_diarys = list(executor.map(process_clean_diary, diary_results))

    logger.info("Diarization results cleaned")
    return cleaned_diarys

def process_audio_diarization(args, cfg):
    # Standardize audio
    secret = "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"
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=secret).to(torch.device(f"cuda:{args.cuda_id}"))
    logger.info("Diarization model loaded")
    # Get list episode name
    episode_list = sorted(os.listdir(os.path.join(args.data_path, args.playlist_name)))
    episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in episode_list]
    sample_rate = cfg["save_step"]["standardization"]["sample_rate"]
    for episode in episode_list:
        episode_path = os.path.join(args.data_path, args.playlist_name, episode)
        logger.info("Processing episode: %s", episode)
        # Get audio file
        data = standardize(episode_path, cfg)
        waveform = data["waveform"]
        #Push waveform to GPU
        try:
            waveform = torch.from_numpy(waveform).unsqueeze(0).to(f'cuda:{args.cuda_id}')
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"GPU {args.cuda_id} is out of memory. Skipping {episode.rsplit('.', 1)[0]}.")
                continue
        # Get diarization results
        diary = pipeline({'waveform': waveform, 'sample_rate': sample_rate})
        # Save diarization results
        save_diarization_results(diary, args.playlist_name, os.path.basename(episode).rsplit('.', 1)[0])
        logger.info("Done processing episode: %s", episode)
        torch.cuda.empty_cache()
