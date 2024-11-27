import os
import cv2
import sys
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../.."
sys.path.append(os.path.join(PROJECT_DIR, "src/utils"))


def read_video(video_path):
    cap             = cv2.VideoCapture(video_path)
    fps             = cap.get(cv2.CAP_PROP_FPS)
    frame_count     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size      = (frame_width, frame_height)

    print(f"FPS: {fps}")
    print(f"Frame count: {frame_count}")
    print(f"Frame size: {frame_size}")

    return cap, fps


def save_frames(cap, output_dir, idx, fps):
    os.makedirs(output_dir, exist_ok=True)

    # Save frames png each 1 second
    frame_interval = fps
    _frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Variables for counting seconds
    second_count = 0

    for current_time in range(0, _frame_count, int(frame_interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_time)
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame
        output_path = os.path.join(output_dir, f"{idx}_{second_count:04d}s.png")
        if not os.path.exists(output_path):
            cv2.imwrite(output_path, frame)

        second_count += 1

    return second_count

def read_and_save_frame_ffmpeg(video_path, output_dir, idx, fps):
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"ffmpeg -i {video_path} -r {fps} {output_dir}/{idx}_%04ds.png"
    os.system(cmd)

def get_out_dir(idx):
    return os.path.join(PROJECT_DIR, "output", "1_extracted", "frames", idx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=False)
    parser.add_argument("--idx", type=str, required=True)

    args = parser.parse_args()

    video_path = args.video
    idx = args.idx
    output_dir = get_out_dir(idx)

    if video_path is None:
        import pandas as pd
        df = pd.read_csv("output/index.csv", sep="|")
        row = df[df["basename"] == idx]
        video_path = os.path.join("output/0_raw_videos", row.iloc[0].video)
        print(f"Using video path from index.csv: {video_path}")

    cap, fps = read_video(video_path)
    num_frames = save_frames(cap, output_dir, idx, fps)
    cap.release()

    print(f"Saved {num_frames} frames to {output_dir}")