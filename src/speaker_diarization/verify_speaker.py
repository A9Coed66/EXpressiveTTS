import os
import pandas as pd
from speechbrain.inference.speaker import SpeakerRecognition # type: ignore

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")


anchor_path = '../data/anchor'

def compute_similarities_score(unverified_path, anchor_path):
    scores = []
    for audio in os.listdir(anchor_path):
        audio_path = os.path.join(anchor_path, audio)
        score, _ = verification.verify_files(unverified_path, audio_path)
        scores.append(score)
    return sum(scores)/len(scores)


def verify_speaker(folder_name):
    """
    Use:
        split_diarization audio
        anchor

    Output:
        verified_speaker
            Nguoi phan xu
                chunk_1.csv
                ...        
    """
    for dir in os.listdir(f'split_diary/{folder_name}'):    #chunk_1, chunk_2,...
        dir_path = os.path.join(f'split_diary/{folder_name}', dir)
        df = pd.DataFrame(columns=['audio_path', 'model_label', 'score', 'verified'])
        for audio_name in os.listdir(dir_path):
            audio_path = os.path.join(dir_path, audio_name)
            for anchor in os.listdir(anchor_path):
                d[anchor] = compute_similarities_score(audio_path, os.path.join(anchor_path, anchor))
            max_key = max(d, key=lambda x: d[x].item())
            df = df.append({'audio_path': audio_path, 'model_label': max_key, 'score': d[max_key].item(), 'verified': (d[max_key]>=0.25)}, ignore_index=True)
        df.to_csv(f'verified_speaker/{folder_name}/{dir}.csv', index=False)


    # df = pd.DataFrame(columns=['audio_path', 'diary_label', 'model_label', 'score', 'verified'])
    # for path in os.listdir(folder_name):
    #     d = {}
    #     diary_label = path.split()[1]
    #     audio_path = os.path.join(folder_name, path)
    #     for anchor in os.listdir(anchor_path):
    #         d[anchor] = compute_similarities_score(audio_path, os.path.join(anchor_path, anchor))
    #     print(d, path)
    #     """
    #     {'BichNgoc': tensor([-0.0291]), 'DucAnh': tensor([0.2324])} output SPEAKER_02 213.7 to 214.3.wav"""
    #     max_key = max(d, key=lambda x: d[x].item())
    #     df = df.append({'audio_path': audio_path, 'diary_label': diary_label, 'model_label': max_key, 'score': d[max_key].item(), 'verified': (d[max_key]>=0.25)}, ignore_index=True)
    return
