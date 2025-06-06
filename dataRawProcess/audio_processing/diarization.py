import os
import torch
import pickle
from pyannote.audio import Pipeline
from utils.logger import Logger
import psutil
from concurrent.futures import ThreadPoolExecutor
import torchaudio
from audio_processing.audio_utils import standardize
import numpy as np
logger = Logger.get_logger()
from utils.tool import check_exists
import soundfile as sf

secret      = "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"
pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=secret)

logger.info("Diarization model loaded")
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

# def process_clean_diary(cfg, playlist_name, episode_name):
#     processed_log = []
#     queue_segment = []

#     # Load diarization results
#     episode_diary_path = os.path.join('./00_diarization', playlist_name, f'{episode_name}.pkl')
#     with open(episode_diary_path, 'rb') as f:
#         diary = pickle.load(f)

#     conflict = True
#     # Step 1: seperate by time
#     for turn, _, speaker in diary.itertracks(yield_label=True):
#         last_end = queue_segment[0][-1] if queue_segment else 0

#         start, end = turn.start, turn.end
#         current_segment     = [[start, end], speaker]

#         if start > last_end :
#             if not conflict and queue_segment:
#                 processed_log.append(queue_segment)
#             conflict = False
#             queue_segment = current_segment
#             last_end = end
#             continue

#         if start <= last_end:
#             queue_segment=None
#             conflict = True
#             last_end = max(last_end, end)
    
#     if queue_segment:
#         processed_log.append(queue_segment)

#     #FIXME Step 2: filter minimum segment length and merge segments
#     MIN_SEGMENT_LENGTH = cfg["save_step"]["diarization"]["min_segment_length"]
#     SILENT_THRESHOLD   = cfg["save_step"]["diarization"]["silent_threshold"]
#     MAX_SEGMENT_LENGTH = cfg["save_step"]["diarization"]["max_segment_length"]
#     filter_segments_length = [item for item in processed_log if item[0][1] - item[0][0] > MIN_SEGMENT_LENGTH]
#     final_segments = []
#     last_speaker = None
#     for segment in filter_segments_length:
#         last_speaker = final_segments[-1][1] if final_segments else None
#         if last_speaker is None or last_speaker != segment[1]:
#             final_segments.append(segment)
#             continue
#         ## Mặc định đã tồn tại last speaker và speaker giống cái trước
#         if segment[0][1] - final_segments[-1][0][0] > MAX_SEGMENT_LENGTH:
#             final_segments.append(segment)
#         elif segment[0][0] - final_segments[-1][0][1] < SILENT_THRESHOLD:
#             final_segments[-1][0][1] = segment[0][1]
#         else:
#             final_segments.append(segment)

#     # Step 3: save segments
#     os.makedirs(os.path.join('./01_clean_diarization', playlist_name), exist_ok=True)
#     save_path = os.path.join('./01_clean_diarization', playlist_name, f'{episode_name}.pkl')
#     with open(save_path, 'wb') as f:
#         pickle.dump(final_segments, f)
#     logger.info(f"Cleaned diarization results saved to {save_path}")


#     return filter_segments_length

def process_clean_diary(args, cfg, playlist_name, episode_name):
    processed_log = []
    queue_segment = []

    # Load diarization results
    episode_diary_path = os.path.join('./00_diarization', playlist_name, f'{episode_name}.pkl')
    with open(episode_diary_path, 'rb') as f:
        diary = pickle.load(f)


    # Step 1: segment_type: [start, end, speaker, is_start, is_end]
    last_end = None
    single_segment = []
    pre_queue = []
    current_queue = []
    for turn, _, speaker in diary.itertracks(yield_label=True):
        pre_queue = current_queue.copy() if current_queue else []
        if last_end is None or turn.start > last_end:
            if pre_queue:
                single_segment.append(pre_queue)
            current_queue = [turn.start, turn.end, speaker, True, True]
        else:   # turn.start <= last_end
            # Xử lý pre_queue
            if turn.start > pre_queue[0]:
                single_segment.append([pre_queue[0], turn.start, pre_queue[2], pre_queue[3], False])
            else:
                pass
            
            # Xử lý current_queue
            if turn.end < pre_queue[0]:
                pass
            elif turn.end < pre_queue[1]:
                current_queue[0] = turn.end
                current_queue[3] = False
                current_queue[2] = pre_queue[2]
            else: #turn.end >= pre_queue[1]
                current_queue = [pre_queue[1], turn.end, speaker, False, True]
        last_end = max(turn.end, last_end) if last_end is not None else turn.end
        

    if current_queue:
        single_segment.append(current_queue)

    #FIXME Step 2: filter minimum segment length and merge segments
    # List of segment_type ([start, end, speaker, is_start, is_end], [start, end, speaker, is_start, is_end], ...)
    MIN_SEGMENT_LENGTH = cfg["save_step"]["diarization"]["min_segment_length"]
    SILENT_THRESHOLD   = cfg["save_step"]["diarization"]["silent_threshold"]
    MAX_SEGMENT_LENGTH = cfg["save_step"]["diarization"]["max_segment_length"]
    filter_segments_length = [item for item in single_segment if item[1] - item[0] > MIN_SEGMENT_LENGTH]
    final_segments = []
    last_speaker = None
    for segment in filter_segments_length:
        last_speaker = final_segments[-1][1] if final_segments else None
        if last_speaker is None or last_speaker != segment[1]:
            final_segments.append(segment)
            continue
        ## Mặc định đã tồn tại last speaker và speaker giống cái trước
        if segment[1] - final_segments[-1][0] > MAX_SEGMENT_LENGTH:
            final_segments.append(segment)
        elif segment[0] - final_segments[-1][1] < SILENT_THRESHOLD:
            if segment[3] is not False and final_segments[-1][3] is not False:
                final_segments[-1][0][1] = segment[0][1]
            else:
                final_segments.append(segment)
        else:
            final_segments.append(segment)

    if args.remove_minority:
        # Step 4: Remove minority speakers
        speaker_count = {}
        total_duration = 0
        for segment in final_segments:
            speaker = segment[2]
            if speaker not in speaker_count:
                speaker_count[speaker] = 0
            speaker_count[speaker] += segment[1] - segment[0]
            total_duration += segment[1] - segment[0]
        logger.info(f"Speaker count: {speaker_count}")
        # Remove speaker if they have less than 10% of total duration
        threshold = total_duration * 0.08
        final_segments = [seg for seg in final_segments if speaker_count[seg[2]] >= threshold]
    
    # Step 3: save segments
    os.makedirs(os.path.join('./01_clean_diarization', playlist_name), exist_ok=True)
    save_path = os.path.join('./01_clean_diarization', playlist_name, f'{episode_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(final_segments, f)
    logger.info(f"Cleaned diarization results saved to {save_path}")


    return filter_segments_length

def clean_diarization_results(args, cfg):
    """Clean and process diarization that each segment has only one speaker
    
    Args:
        diary_results (list): List of diarization results []
        cfg (dict): Configuration dictionary
    """
    os.makedirs(os.path.join('./01_clean_diarization'), exist_ok=True)
    os.makedirs(os.path.join('./01_clean_diarization', args.playlist_name), exist_ok=True)

    
    step_path = './01_clean_diarization'
    episode_list = sorted(os.listdir(os.path.join(args.data_path, args.playlist_name)))
    episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in episode_list]
    episode_name = check_exists(step_path, args.playlist_name, episode_name, type='file')

    for episode in episode_name:
        process_clean_diary(args, cfg, args.playlist_name, episode)

    logger.info("Diarization results cleaned")
    return 

def process_audio_diarization(args, cfg):
    # Standardize audio
    pipeline.to(torch.device(f"cuda:{args.cuda_id}"))
    # Get list episode name
    
    os.makedirs(os.path.join('./00_diarization', args.playlist_name), exist_ok=True)

    step_path = './00_diarization'
    episode_list = sorted(os.listdir(os.path.join(args.data_path, args.playlist_name)))
    episode_name = [os.path.basename(ep).rsplit('.', 1)[0] for ep in episode_list]
    episode_name = check_exists(step_path, args.playlist_name, episode_name, type='file')
    
    sample_rate = cfg["save_step"]["standardization"]["sample_rate"]

    for episode in episode_name:
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
        save_diarization_results(diary, args.playlist_name, episode)
        logger.info("Done processing episode: %s", episode)
        standardize_path = os.path.join('./00_standardization', args.playlist_name)
        os.makedirs(standardize_path, exist_ok=True)
        standardize_audio_path = os.path.join(standardize_path, f'{episode}.wav')
        sf.write(standardize_audio_path, np.ravel(waveform.cpu().numpy()), sample_rate)
        # sf.write(standardize_audio_path, waveform, sample_rate)
        torch.cuda.empty_cache()
