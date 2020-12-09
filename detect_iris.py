# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[INFO] loading facial landmark predictor...")
# detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] input iris image")
path = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\acv6\HW\S5003R06.jpg'
# path = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\acv6\HW\S5001R03.jpg'

img = cv2.imread(path)
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rmin = int(img.shape[0]/2) - 100
cmin = int(img.shape[1]/3) - 100
rmax = img.shape[0] - 100
cmax = img.shape[1] - 100
dlib_rect = dlib.rectangle(rmin, cmin, rmax, cmax)
shape = predictor(gray, dlib_rect)

shape = face_utils.shape_to_np(shape)

# loop over the (x, y)-coordinates from our dlib shape
# predictor model draw them on the image
for (sX, sY) in shape:
    cv2.circle(img, (sX, sY), 3, (0, 0, 255), -1)

cv2.imshow("Frame", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
