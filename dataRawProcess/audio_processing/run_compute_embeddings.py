from cosine_pair import saibamomoi_3
import os

path = '/home4/tuanlha/EXpressiveTTS/dataRawProcess/04_denoise'
for playlist_name in os.listdir(path):
    saibamomoi_3(playlist_name)