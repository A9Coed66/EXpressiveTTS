from IPython import display as disp
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import os

def denoise(folder_name, get_audio = True):
    model = pretrained.dns64().cuda()
    for file_name in os.listdir(f'chunk/{folder_name}'):    # file_name = chunk/yeunhaudi/chunk_{i}.mp3
        file_path = os.path.join(f'chunk/{folder_name}', file_name)
        wav, sr = torchaudio.load(file_path)
        wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
        with torch.no_grad():
            denoised = model(wav[None])[0]
        if get_audio:
            torchaudio.save(f'denoised/{folder_name}/{file_name}', denoised.to('cpu'), sr)
            print("Audio 'output.mp3' created!")
        return 