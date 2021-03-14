from collections import deque
import numpy as np
import argparse
import imutils
import cv2

#construct the argument parse and parse the arguments
from src.court_detection import CourtDetector

args = {'video': '../videos/vid1.mp4'}


camera = cv2.VideoCapture(args["video"])

params = cv2.SimpleBlobDetector_Params()

params.filterByColor = True
params.blobColor = 255
# Set Area filtering parameters
params.filterByArea = True
params.minArea = 0
params.maxArea = 20

# Set Circularity filtering parameters
params.filterByCircularity = False

# Set Convexity filtering parameters
params.filterByConvexity = True
params.minConvexity = 0.2

# Set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0
params.maxInertiaRatio = 1

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
court_detector = CourtDetector()
backSub = cv2.createBackgroundSubtractorKNN(3)
frame_i = 0
while True:
    ret, frame = camera.read()

    if not ret:
        break
    frame_i += 1
    if frame_i == 1:
        court_detector.detect(frame)
    court_detector.track_court(frame)

    mask = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    mask = cv2.medianBlur(mask, 3)

    fgMask = backSub.apply(mask)
    fgMask = cv2.dilate(fgMask, np.ones((10, 10)))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_HITMISS, np.ones((10, 10)))

    white_ref = court_detector.court_reference.get_court_mask()
    white_mask = cv2.warpPerspective(white_ref, court_detector.court_warp_matrix[-1], frame.shape[1::-1])
    image_court = frame.copy()
    image_court[white_mask == 0, :] = (0, 0, 0)
    image_court[fgMask == 0, :] = (0, 0, 0)
    # Detect blobs
    keypoints = detector.detect(image_court)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(image_court, keypoints, blank, (0, 0, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    # Show blobs
    cv2.imshow("Filtering Circular Blobs Only", blobs)
    cv2.waitKey(1)
camera.release()
cv2.destroyAllWindows()