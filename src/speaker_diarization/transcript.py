from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-large", device='cuda')

def transcriber(folder_name):
    """
    Use:
        concat audio
        
    Output:
        transcripted audio
            audio_path, transcript, speaker
    """