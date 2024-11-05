from IPython import display as disp
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import os
import IPython.display as ipd

def denoise(folder_name, get_audio = True, use_denoise=False):
    data_path = '../data'
    save_path = f'{data_path}/denoised/{folder_name}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file_name in os.listdir(f'{data_path}/chunk/{folder_name}'):    # file_name = chunk/yeunhaudi/chunk_{i}.mp3
        file_path = os.path.join(f'{data_path}/chunk/{folder_name}', file_name)
        wav, sr = torchaudio.load(file_path)
        # wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        # ipd.display(ipd.Audio(wav.numpy(), rate=16000))
        
        if use_denoise:
            model = pretrained.dns64().cuda()
            wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
            with torch.no_grad():
                # Denoise the audio
                denoised = model(wav[None])[0]
            
            # Optionally save the denoised audio to disk
            if get_audio:
                torchaudio.save(f'{save_path}/{file_name}', denoised.to('cpu'), model.sample_rate)
                print(f"Saved {file_name} to {save_path}")
        else:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
            torchaudio.save(f'{save_path}/{file_name}', wav.to('cpu'), model.sample_rate)
            print(f"Save {file_name}")