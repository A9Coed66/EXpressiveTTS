from pydub import AudioSegment
import os

def split(audio_file, folder_name, duration=300):
  """
  Chia một file âm thanh thành các đoạn nhỏ hơn.

  Args:
    audio_file (str): Đường dẫn đến file âm thanh.
    duration (int): Độ dài của mỗi đoạn (tính bằng giây).

  Returns:
    list: Danh sách các đoạn âm thanh đã chia.
  """
  if not os.path.exists("../data/chunk/" + folder_name):
    os.makedirs("../data/chunk/" + folder_name)
        
  audio = AudioSegment.from_mp3(audio_file)
  chunks = []
  for i in range(0, len(audio), duration*1000):
    chunk = audio[i:i+duration*1000]
    chunks.append(chunk)
  for i, chunk in enumerate(chunks):
    chunk.export(f"../data/chunk/{folder_name}/chunk_{i}.mp3", format="mp3")
  return 