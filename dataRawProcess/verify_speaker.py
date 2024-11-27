import os
import pandas as pd
from speechbrain.inference.speaker import SpeakerRecognition # type: ignore
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})

anchor_path = '../data/cut'

def compute_similarities_score(unverified_path, anchor_path):
    scores = []
    for audio in os.listdir(anchor_path):
        audio_path = os.path.join(anchor_path, audio)
        print(unverified_path, audio_path)
        try:
            score, _ = verification.verify_files(unverified_path, audio_path)
        except RuntimeError as e:
            # print(f"Error with path {unverified_path}, {audio_path}")
            score = 0
        scores.append(score)
    if len(scores)==0:
        return 0
    else:
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
    data_path = '../data'
    save_path = f'{data_path}/verified_speaker/{folder_name}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for sub_folder_name in os.listdir(f'{data_path}/split_diary/{folder_name}'):    #chunk_1, chunk_2,...
        df = pd.DataFrame(columns=['audio_path', 'diary_label', 'model_label', 'start', 'end', 'score', 'verified'])
        d = {}
        sub_folder_path = os.path.join(f'{data_path}/split_diary/{folder_name}', sub_folder_name)
        
        for file_name in os.listdir(sub_folder_path):
            diary_label = file_name.split(' ')[-1][:-4]
            start, end = float(file_name.split(' ')[0]), float(file_name.split(' ')[1])
            audio_path = os.path.join(sub_folder_path, file_name)
            for anchor in os.listdir(anchor_path):
                d[anchor] = compute_similarities_score(audio_path, os.path.join(anchor_path, anchor))
            max_key = max(d, key=lambda x: d[x].item())
            new_row = pd.DataFrame([{'audio_path': audio_path, 'diary_label': diary_label, 'model_label': max_key, 'start':start, 'end':end, 'score': d[max_key].item(), 'verified': (d[max_key]>=0.25).item()}])
            df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values(by='start', ascending=True)
        df.to_csv(f'{save_path}/{sub_folder_name}.csv', index=False)


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
