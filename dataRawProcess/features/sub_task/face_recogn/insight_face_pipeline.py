import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis


class InsightFacePipeline:
    """
    Pipeline for face detection
    _input: file or image
    _output: list of face
    """
    def __init__(self):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def __call__(self, img_or_path: np.ndarray | str) -> dict | None:
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path

        faces = self.app.get(img)
        if len(faces) == 0:
            return None

        return faces