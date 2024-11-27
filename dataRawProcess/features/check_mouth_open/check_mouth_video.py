import argparse
import cv2
import os

from check_mouth_img import check_frame

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--vid", type=str, default=0,
	help="path of video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to output video file")
ap.add_argument("-s", "--start", type=float, default=0,
    help="start time")
ap.add_argument("-e", "--end", type=float, default=0,
    help="end time")
ap.add_argument("-i", "--interval", type=float, default=0.2,
    help="frame interval")
ap.add_argument("-sv", "--save", type=bool, default=True,
    help="save images")
args = vars(ap.parse_args())

def check_mouth_video(video_path, start, end, output, frame_interval=0.2, is_save = True):
    cap = cv2.VideoCapture(video_path)
    
    output_folder = output
    os.makedirs(output_folder, exist_ok=True)
    
    frame_interval = frame_interval
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    current_time = start
    
    while current_time<end:
        frame_number = int(frame_rate * current_time)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        check_frame(frame=frame, output=os.path.join(output_folder, f'{current_time}.jpg'), is_save=is_save)
        current_time += frame_interval
    return

if __name__ == "__main__":
	check_mouth_video(args["vid"], args["start"], args["end"], args["output"], args["interval"], args["save"])
	