import os
import sys
import numpy as np

from utils.time_converter import second_to_time
from utils.split import run_ffmpeg, run_ffmpeg_webm

def select_range_from_points(
    seq_start: float, seq_end: float, points: list[float], padding=30.0
):
    points = [p for p in points if p>=seq_start and p<=seq_end]

    if len(points)==0:
        return None
    
    sorted_points = sorted(points)

    point_groups = []
    cur_group = []
    for p in sorted_points:
        if len(cur_group) == 0:
            cur_group.append(p)
        elif p - cur_group[-1] <= padding * 2:
            cur_group.append(p)
        else:
            point_groups.append(cur_group)
            cur_group = [p]
    point_groups.append(cur_group)

    return [
        {
            "start": max(seq_start, min(group) - padding),
            "end": min(seq_end, max(group) + padding),
        }
        for group in point_groups
    ]

def open_face_recogn_split_json(fpath, cus_threshold=None):
    import json

    with open(fpath, "r") as f:
        data = json.load(f)
        data = [(int(k.replace("s", "")), v[0], float(v[1])) for k, v in data.items()]

    if not cus_threshold:
        points = [d[0] for d in data if d[1]]
    else:
        points = [d[0] for d in data if d[2] < cus_threshold]

    return points

if __name__ == "__main__":
    OUT_DIR = "output/2_splitted/videos"
    FACE_RECOGN_RESULT_DIR = "output/1_extracted/frames_face_recogn_result"
    RAW_VIDEO_DIR = "output/0_raw_videos"

    os.makedirs(OUT_DIR, exist_ok=True)
    face_recogn_result_files = os.listdir(FACE_RECOGN_RESULT_DIR)

    INDEX_DF_PATH = "output/index.csv"
    import pandas as pd
    df = pd.read_csv(INDEX_DF_PATH, sep="|")

    for fname in face_recogn_result_files:
        basename = fname.replace(".json", "")
        row = df[df["basename"] == basename]
        if len(row) == 0:
            print(f"Cannot find {basename} in index.csv")
            continue

        cus_threshold = 20.5
        padding = 30.0
        # FIXME: hardcode
        if "VDVK" in basename:
            cus_threshold = 21.5
            padding = 60.0
        points = open_face_recogn_split_json(os.path.join(FACE_RECOGN_RESULT_DIR, fname), cus_threshold=cus_threshold)
        rs = select_range_from_points(0, row.iloc[0].duration, points, padding=padding)

        if rs is None:
            print(f"No range found for {basename}")
            continue

        for r in rs:
            start = r["start"]
            end = r["end"]
            start_t = second_to_time(start)
            end_t = second_to_time(end)
            
            out_path = f"{OUT_DIR}/{basename}-{start_t}-{end_t}.mp4"
            if os.path.exists(out_path):
                print(f"Skip {basename}-{start_t}-{end_t}.mp4")
                continue
            
            # cut_video(
            video_path = row.iloc[0].video
            if "webm" in video_path:
                run_ffmpeg_webm(
                    os.path.join(RAW_VIDEO_DIR, video_path),
                    start,
                    end,
                    out_path,
                )
            else:
                run_ffmpeg(
                    os.path.join(RAW_VIDEO_DIR, video_path),
                    start,
                    end,
                    out_path,
                )