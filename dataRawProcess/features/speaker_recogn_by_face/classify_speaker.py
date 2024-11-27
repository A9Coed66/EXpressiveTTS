import os
import cv2
import sys
import json
import numpy as np

from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../.."
sys.path.append(os.path.join(PROJECT_DIR, "src"))


from utils.load_emb_dir import load_nb_emb_dir
from utils.loop import loop_dir
from sub_task.face_recogn.insight_face_pipeline import InsightFacePipeline
from sub_task.face_recogn.compare_emb import compare_faces


EMB_DIR = os.path.join(PROJECT_DIR, "data/annot_face_embs")
FRAME_DIR = os.path.join(PROJECT_DIR, "output/1_extracted/frames")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output/1_extracted/frames_classified_debug")
OUTPUT_JSON_DIR = os.path.join(
    PROJECT_DIR, "output/1_extracted/frames_face_recogn_result"
)


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

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
    """
    Load embedded data each speaker
    -> spk_embs"""
    basename = row["basename"]
    path = os.path.join(EMB_DIR, basename)
    embs = load_nb_emb_dir(path, stack=True)
    spk_embs[basename] = embs

loop_dir(EMB_DIR, load_emb_fn)


def detect_face(path, idx):
    # TODO: Tối ưu hóa hàm này vì chưa trả về giá trị thực sự có nhiều ý nghĩa
    """
    return: found_flag, min_dist
    :found_flag: True if found speaker
    :min_dist: min distance to speaker, but not return speaker with min distance
    """
    fname = os.path.basename(path)
    out_img_path = os.path.join(OUTPUT_DIR, fname)
    # if not os.path.exists(out_img_path):

    img = cv2.imread(path)
    faces = pipeline(img)

    if not faces:
        return False, 1000
    
    frame_embs  = np.array([r["embedding"] for r in faces])

    rs          = []
    distancess  = []
    found_flag  = False
    min_dist    = 1000
    for frame_emb in frame_embs:
        # Load each face in images
        preds = []
        dis = []
        for spk, embs in spk_embs.items():
            # Compare with each embedded speaker data
            _rs, _dis = compare_faces(
                embs.T, frame_emb.reshape(1, -1), tolerance=20.0
            )  # NOTE: MAY ACCEPT MANY
            dis.append(_dis.min())
            if any(_rs):
                preds.append(spk)
                found_flag = True
                break
        pred = ",".join(preds) if preds else "unknown"
        rs.append(pred)
        distancess.append(f"{dis}")
        min_dist = min(min_dist, min(dis))
    
    # FOR DEBUG
    if True: # found_flag: # TRUE
        for r, face, d in zip(rs, faces, distancess):
            bbox = face["bbox"]
            if r == "unknown":
                cv2.putText(
                    img,
                    r + d,
                    (int(bbox[0]), int(bbox[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    img,
                    r + d,
                    (int(bbox[0]), int(bbox[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.rectangle(
                img,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2,
            )

        cv2.imwrite(out_img_path, img)

    return found_flag, min_dist

def detect_face_dir_fn(kwarg):
    path = kwarg["path"]
    basename = kwarg["basename"]
    out = {}
    print(f"Process {path}")
    out_json_file = os.path.join(OUTPUT_JSON_DIR, basename + ".json")
    if os.path.exists(out_json_file):
        print(f"File {out_json_file} already exists. Skip.")
        return
    
    for img_name in tqdm(os.listdir(path)):
        seconds = img_name.split(".")[0].split("_")[-1]
        found, min_dist = detect_face(os.path.join(path, img_name), basename)
        # TODO: xem xét thêm thông tin speaker vào out
        out[seconds] = [found, str(min_dist)]
    with open(out_json_file, "w") as f:
        json.dump(out, f, indent=4)


print("Classifying speaker...")
loop_dir(FRAME_DIR, detect_face_dir_fn)