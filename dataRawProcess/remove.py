import argparse
from ast import arg
import glob
import librosa
import torchaudio
import numpy as np
from scipy.io import wavfile
import numpy as np
from scipy.io.wavfile import write
import os
import pandas as pd
import seaborn as sns
from pathlib import Path

def remove(file_csv, thresh_hold):
    file_csv = pd.read_csv(file_csv)
    for i in range(len(file_csv)):
        if ((file_csv['MinCos'][i])<float(thresh_hold)):
            os.remove(file_csv['Path'][i])