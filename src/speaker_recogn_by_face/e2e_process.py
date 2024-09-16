import os
import sys
import cv2
import json
from tqdm import tqdm
from extract_images import read_video
from utils.loop import loop_dir
from utils.loop import loop_index_csv
from utils.load_emb_dir import load_nb_emb_dir
from sub_task.face_recogn.insight_face_pipeline import InsightFacePipeline
from sub_task.face_recogn.compare_emb import compare_faces

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../.."
sys.path.append(os.path.join(PROJECT_DIR, "src"))


# ------------------------------------------------------------

# NOTE: overlap with part of code in extract_images.py
def frame_generator(cap, frame_interval):
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        local_cycle = int(round(frame_count / frame_interval) * frame_interval)
        if (
            (frame_count == 0)
            or (local_cycle != 0)
            and (frame_count % local_cycle == 0)
        ):
            yield frame, frame_count

        frame_count += 1

# ------------------------------------------------------------




RAW_VIDEOS_DIR  = os.path.join(PROJECT_DIR, "output/0_raw_videos")
EMB_DIR         = os.path.join(PROJECT_DIR, "data/annot_face_embs")
OUTPUT_JSON_DIR = os.path.join(PROJECT_DIR, "output/1_extracted/frames_face_recogn_result")

os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
assert os.path.exists(RAW_VIDEOS_DIR), f"Directory {RAW_VIDEOS_DIR} does not exist"
if not os.path.exists(EMB_DIR):
    sys.exit(
        f"""
        Directory {EMB_DIR} does not exist. 
        Please create data/annot_faces and manually annotate face image.
        Them generate embedding with extract_annot_face_emb.py embeddings.
        """
    )

print("Loading InsightFace model...")
pipeline = InsightFacePipeline()

print("Loading speaker embeddings...")
# load speaker embeddings
spk_embs = {}
def load_emb_fn(row):
    basename = row["basename"]
    path = os.path.join(EMB_DIR, basename)
    embs = load_nb_emb_dir(path, stack=True)
    spk_embs[basename] = embs
loop_dir(EMB_DIR, load_emb_fn)


import numpy as np

def detect_face_frame(frame):
    faces = pipeline(frame)

    if not faces:
        return False, 1000

    frame_embs = np.array([r["embedding"] for r in faces])

    rs = []
    distancess = []
    found_flag = False
    min_dist = 1000
    for frame_emb in frame_embs:
        preds = []
        dis = []
        for spk, embs in spk_embs.items():
            _rs, _dis = compare_faces(
                embs.T, frame_emb.reshape(1, -1), tolerance=20.0
            )  # NOTE: ACCEPT MANY
            dis.append(_dis.min())
            if any(_rs):
                preds.append(spk)
                found_flag = True
                break
        pred = ",".join(preds) if preds else "unknown"
        rs.append(pred)
        distancess.append(f"{dis}")
        min_dist = min(min_dist, min(dis))
    
    return found_flag, min_dist

# ------------------------------------------------------------

def handle_video_fn(row):
    idx = row["basename"]
    video_fname = row["video"]
    video_path = os.path.join(RAW_VIDEOS_DIR, video_fname)
    cap, fps = read_video(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = {}
    out_json_file = os.path.join(OUTPUT_JSON_DIR, idx + ".json")
    if os.path.exists(out_json_file):
        print(f"Skip {idx}")
        return

    for frame, sec in tqdm(frame_generator(cap, 1), total=frame_count):
        found_flag, min_dist = detect_face_frame(frame)
        out[sec] = [found_flag, str(min_dist)]
        
    with open(out_json_file, "w") as f:
        json.dump(out, f, indent=4)

    cap.release()
    return out


if __name__ == "__main__":
    print("Classifying speaker...")
    loop_index_csv(
        os.path.join(PROJECT_DIR, "output", "index.csv"),
        handle_video_fn,
    )