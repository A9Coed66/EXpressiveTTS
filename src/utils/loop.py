import os
import torch
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm


def loop_dir(dir_to_loop, fn=None):
    """
    Apply function to each directory in dir_to_loop"""
    dirs = os.listdir(dir_to_loop)
    dirs = sorted(dirs, reverse=True)

    for d in tqdm(dirs):
        if not os.path.isdir(os.path.join(dir_to_loop, d)):
            continue
        if fn is not None:
            fn({"path": os.path.join(dir_to_loop, d), "basename": d})
        else:
            print(d)

def loop_index_csv(index_file, fn=None):
    num_worker = mp.cpu_count()
    if torch.cuda.is_available():
        num_worker = max(torch.cuda.get_device_properties(0).multi_processor_count//2, num_worker)

    df = pd.read_csv(index_file, sep="|")
    
    with mp.Pool(num_worker) as pool:
        pool.map(fn, df.iterrows())