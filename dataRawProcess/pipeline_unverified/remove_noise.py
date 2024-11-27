import os
import pandas as pd

folder_path = "/home2/tuannd/tuanlha/temp"
for char in os.listdir(folder_path):
    df = pd.read_csv(os.path.join(folder_path, char))