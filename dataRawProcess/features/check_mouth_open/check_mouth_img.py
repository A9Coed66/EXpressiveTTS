# from https://github.com/mauckc/mouth-open #

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import PIL

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)
	# return the mouth aspect ratio
	return mar


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
	help="path to facial landmark predictor")
ap.add_argument("-w", "--img", type=str, default=0,
	help="path of image file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to output file")
args = vars(ap.parse_args())


# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.79


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

# start the video stream thread
frame = cv2.imdecode(np.fromfile(args["img"], dtype=np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

frame_width = 640
frame_height = 360

rects = detector(gray, 0)

def check_frame(frame, output, is_save = False):
	gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	
    # take the first face
	rect = rects[0]
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	mouth =shape[mStart:mEnd]
	mouthMAR = mouth_aspect_ratio(mouth)
	mouthHull = cv2.convexHull(mouth)

	# draw mouth
	cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
	mar = mouthMAR
	cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# save image with drawed mouth
	if is_save: 
		cv2.imwrite(output, frame)

	# NOTE: CHECK IF MOUTH IS OPEN
	if mar > MOUTH_AR_THRESH:
		print("Mout is opening")
		cv2.putText(frame, "Mouth is Open!", (30,60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
		return True
	else: 
		print("Mouth not open")
		return False

def check_image(image_path: str, output):
	if type(image_path) == str:
		frame = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
	else:
		frame = image_path
	return check_frame(frame, output)

# main
if __name__ == "__main__":
	check_image(args["img"], args["output"])
	

# do a bit of cleanup