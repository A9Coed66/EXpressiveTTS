from IPython import display as disp
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import os

def denoise(folder_name, get_audio = True):
    model = pretrained.dns64().cuda()
    data_path = '../../data'
    save_path = f'{data_path}/denoised/{folder_name}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file_name in os.listdir(f'{data_path}/chunk/{folder_name}'):    # file_name = chunk/yeunhaudi/chunk_{i}.mp3
        file_path = os.path.join(f'{data_path}/chunk/{folder_name}', file_name)
        wav, sr = torchaudio.load(file_path)
        wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
        with torch.no_grad():
            denoised = model(wav[None])[0]
        if get_audio:
            torchaudio.save(f'{save_path}/{file_name}', denoised.to('cpu'), sr)
            print(f"Save {file_name}")