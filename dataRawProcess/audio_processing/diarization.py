import os
import torch
import pickle
from pyannote.audio import Pipeline
from utils.logger import Logger
import psutil
from concurrent.futures import ThreadPoolExecutor

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

def save_diarization_results(diary_results, save_path):
    """Save diarization results to disk"""
    for item in diary_results:
        name = item['name']
        diary = item['diary']
        # embeddings = item['embeddings']

        file_path = os.path.join(save_path, f'{name}.pkl')
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                pickle.dump(diary, f)
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

def process_audio_diarization(audio_files, cfg, playlist_name):
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
    
    pass