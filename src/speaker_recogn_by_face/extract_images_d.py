from extract_images import read_video, save_frames, get_out_dir

import os
import sys
import cv2

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../.."
sys.path.append(os.path.join(PROJECT_DIR, "src/utils"))

from utils.loop import loop_index_csv
RAW_VIDEOS_DIR = None   # TODO: call the ambulance


def extract_images_one_video(video_path, idx):
    out_dir = get_out_dir(idx)
    
    if os.path.exists(out_dir):
        print(f"Frames already extracted for {video_path}")
        return
    
    cap, fps = read_video(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if frame_width < 1280:
        print(f"Video {video_path} is too small. Skipping...")
        return
    
    second_count = save_frames(cap, out_dir, idx, fps)
    cap.release()

    return second_count


def _extract_image_fn(params):
    idx, row = params
    idx = row["basename"]
    video_fname = row["video"]
    video_path = os.path.join(RAW_VIDEOS_DIR, video_fname)

    print(f"{idx}: Extracting frames from {video_path}...")

    second_count = extract_images_one_video(video_path, idx)

    print(f"Extracted {second_count} frames from {video_path}")

if __name__ == "__main__":
    loop_index_csv(os.path.join(PROJECT_DIR, "data/index.csv"), 
                   _extract_image_fn)