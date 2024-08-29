import os
import numpy as np

def load_nb_emb_dir(emb_dir, stack=False):
    """
    Load directory contains list of numpy embeddings like:
        emb_dir:
        - emb1.npy
        - emb2.npy
        - ...
    """
    emb_dict = {}
    for file in os.listdir(emb_dir):
        if file.endswith('.npy'):
            emb = np.load(os.path.join(emb_dir, file))
            emb_dict[file] = emb

    if stack:
        emb_dict = {k: np.vstack(v) for k, v in emb_dict.items()}
        embs = np.hstack(list(emb_dict.values()))
        return embs

    return emb_dict