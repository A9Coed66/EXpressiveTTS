import os
import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis
# np.int = np.int32

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../.."
sys.path.append(os.path.join(PROJECT_DIR, "src/utils"))


# Initialize the InsightFace model
print("Initializing InsightFace model...")
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

ANNOT_FACES_DIR = os.path.join(PROJECT_DIR, "data", "annot_faces")
OUT_EMB_DIR = os.path.join(PROJECT_DIR, "data", "annot_face_embs")

os.makedirs(OUT_EMB_DIR, exist_ok=True)


# Extracting embeddings from annotated faces
print("Extracting embeddings...")
for spk in os.listdir(ANNOT_FACES_DIR):
    print(f"Processing {spk}...")
    spk_dir = os.path.join(ANNOT_FACES_DIR, spk)
    out_emb_dir = os.path.join(OUT_EMB_DIR, spk)
    os.makedirs(out_emb_dir, exist_ok=True)

    for img_name in os.listdir(spk_dir):
        img_path = os.path.join(spk_dir, img_name)
        img = cv2.imread(img_path)
        faces = app.get(img)

        if len(faces) == 0:
            print(f"No face detected in {img_path}")
            continue

        face = faces[0]
        emb = face.embedding
        emb_path = os.path.join(out_emb_dir, img_name.split(".")[0] + ".npy")
        np.save(emb_path, emb)