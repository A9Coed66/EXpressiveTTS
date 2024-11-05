from transformers import pipeline
import os
import pandas as pd

transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-large", device='cuda')

def transer(audio_path):
    return transcriber(audio_path)['text']


    
    

def trans(folder_name):
    """
    Use:
        concat audio
        
    Output:
        transcripted audio
            audio_path, transcript, speaker
    """
    data_path = '../data'
    save_path = f'{data_path}/transcribered/{folder_name}'
    
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file_name in os.listdir('../data/logs_concat'):
        file_path = os.path.join('../data/logs_concat', file_name)
        
        # concat dataframe
        df = pd.read_csv(file_path)
        df['script'] = df['audio_path'].apply(transer)
        df.to_csv(f"{save_path}/{file_name}")
    
    return
            