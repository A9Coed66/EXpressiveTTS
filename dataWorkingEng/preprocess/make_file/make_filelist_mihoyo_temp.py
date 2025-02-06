import os
import random
import argparse
import audeer
import audonnx
import librosa
import yaml
import numpy as np
import tqdm

import sys
sys.path.append(os.path.abspath('./'))
from src.utils import seed_init, MakeDir, parse_filelist, Config

import os

import audeer
import csv


model_root = 'model'
cache_root = 'cache'


# audeer.mkdir(cache_root)
# def cache_path(file):
#     return os.path.join(cache_root, file)


# url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
# dst_path = cache_path('model.zip')

# if not os.path.exists(dst_path):
#     audeer.download_url(
#         url, 
#         dst_path, 
#         verbose=True,
#     )
    
# if not os.path.exists(model_root):
#     audeer.extract_archive(
#         dst_path, 
#         model_root, 
#         verbose=True,
#     )

model = audonnx.load(model_root, device='cuda')


def split_train_val_test(write_path, wav_path, mel_path, spk_dict):
    
    mel_filelist = os.listdir(mel_path)
    pitch_energy_saved_path = '/home2/tuannd/tuanlha/PreprocessedData/preprocessed'
    # print(meta_dic)
    
    filelist  = []
    text_list = []
    # cnt = 0
    for mel_file in mel_filelist:
        # print(cnt)
        # cnt+=1
        spk, basename = mel_file.split('-')[0], mel_file.split('-')[-1][:-4]  # remove .npy
        text_path     = os.path.join(wav_path, spk, basename+'.lab')
        wav           = os.path.join(wav_path, spk, basename+'.wav')     
        y, sr         = librosa.load(wav, sr=16000)
        result        = model(y,sr)
        arousal, dominance = result['logits'][0][1], result['logits'][0][2]
        pitch_target, energy_target = np.load(os.path.join(pitch_energy_saved_path, 'duration')).round(4), np.load(os.path.join(pitch_energy_saved_path, 'duration')).round(4)
         
        
        with open(text_path, "r") as f:
            txt = f.readline().strip("\n")
        
        spk       = str(spk_dict[spk])
        file_path = os.path.join(mel_path, mel_file)
        strings   = '|'.join([file_path, txt, spk, f'{round(arousal,4)}', f'{round(dominance,4)}', pitch_target, f'{energy_target}' +'\n'])
        print(strings)
        filelist.append(strings)
        text_list.append(txt + '\n')

    filelist = sorted(filelist)
    random.shuffle(filelist)
    
    val_size  = int(0.90 * len(filelist))
    test_size = int(0.95 * len(filelist))
    train_filelist = filelist[:val_size]
    val_filelist   = filelist[val_size:test_size]
    test_filelist  = filelist[test_size:]
    with open(f"{write_path}/filelist.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        for line in filelist:
            writer.writerow(line.split('|'))
    print(len(filelist), len(train_filelist), len(val_filelist), len(test_filelist))
    with open(f"{write_path}/train.txt", "w") as file:
        file.writelines(train_filelist)
    with open(f"{write_path}/valid.txt", "w") as file:
        file.writelines(val_filelist)
    with open(f"{write_path}/test.txt", "w") as file:
        file.writelines(test_filelist)
        
    text_list = sorted(list(set(text_list)))    
    random.shuffle(text_list)
        
    with open("test_sentence/mihoyo_sentence.txt", "w", encoding="utf-8") as file:
        file.writelines(text_list)
           
    
def make_unseen_filelist(write_path, unseen_spk):
    
    for phase in ['train', 'valid']:
        filtered_list = []

        with open(os.path.join(write_path, f'{phase}.txt'), "r", encoding="utf-8") as f:
            
            strings = f.readlines()
            for i, line in enumerate(strings):
                mel_path, text, spk, *_ = line.strip("\n").split("|")
                
                if int(spk) in unseen_spk:
                    continue
                else:
                    filtered_list.append(line)        
        
        num_origin = len(strings)
        num_new    = len(filtered_list)

        with open(os.path.join(write_path, f"{phase}_unseen.txt"), "w", encoding="utf-8") as file:
            file.writelines(filtered_list)

        print(f'{phase} size: {num_origin} --> {num_new}')
    

def main(cfg):
    
    seed_init(seed=100)
    
    write_path = f'./filelists/{cfg.dataset}'
    wav_path   = cfg.path.raw_path
    mel_path   = f'{cfg.path.preprocessed_path}/mel'
    MakeDir(write_path)

    spk_list = sorted(os.listdir(wav_path))
    print('Number of speakers:', len(spk_list))
    spk_dict = {k:v for v,k in enumerate(spk_list)}
    print(spk_dict)
    unseen_spk   = sorted(random.sample(range(len(spk_dict)), k=0))
    print('Unseen speaker:', unseen_spk)
    
    
    ### Save filelist ####
    split_train_val_test(write_path, wav_path, mel_path, spk_dict)
    
    
    ### filter for unseen spekaer ####
    make_unseen_filelist(write_path, unseen_spk)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/Mihoyo/preprocess.yaml")
    args = parser.parse_args()

    cfg = Config(args.config)
    
    main(cfg)
    