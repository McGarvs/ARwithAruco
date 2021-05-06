import cv2
import sys
import cv2.aruco as aruco
import numpy as np
import imutils
from imutils.video import VideoStream
import os
from collections import deque
import time
import pyglet

USE_CACHE = True
CACHED_REF_PTS = None
MY_IDS = (800, 700, 40, 1000)
# MY_IDS = (1000, 40, 800, 700)
DICT = aruco.DICT_ARUCO_ORIGINAL

FILE_PATH = os.path.join(os.getcwd(), "RR.mp4")
# source = cv2.imread(FILE_PATH)
# source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

def find_and_warp(frame, source, cornerIDs, arucoDict, arucoParams, useCache=False):
    # Get ref to cached pts
    global CACHED_REF_PTS
    # Get Image sizes
    (imgH, imgW) = frame.shape[:2]
    (srcH, srcW) = source.shape[:2]
    # Detect Markers
    (corners, ids, rejects) = aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    # Initialize Corners
    ids = np.array([]) if len(corners) != 4 else ids.flatten()
    # print("Detecting", ids)
    # Initalize ref pts
    refPts = []
    # Look for the correct corner Markers in the frame
    for i in cornerIDs:
        # Grab corner's index
        j = np.squeeze(np.where(ids == i))
        # Skip if can't find marker
        if j.size == 0:
            continue
        # Append to reference points
        corner = np.squeeze(corners[j])
        refPts.append(corner)
    # Check for failure to find markers
    if len(refPts) != 4:
        # print("FAILED to find 4")
        # Try to use our cached points
        if useCache and CACHED_REF_PTS is not None:
            refPts = CACHED_REF_PTS
        else:
            return None
    # Update the cached points
    if useCache:
        CACHED_REF_PTS = refPts
    # Unpack ref point
    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    dstMat = np.array(dstMat)
    # Use source Dims 
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
    # Compute Homography
    (H, _) = cv2.findHomography(srcMat, dstMat)
    warped = cv2.warpPerspective(source, H, (imgW, imgH))
    # Construct mask
    mask = np.zeros((imgH,imgW), dtype="uint8")
    cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv2.LINE_AA)
    # Optional Border
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, rect, iterations=2)
    # Modify mask to 3 channels
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)
    # Create output
    warpMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
    frameMultiplied = cv2.multiply(frame.astype("float"), 1.0 - maskScaled)
    output = cv2.add(warpMultiplied, frameMultiplied)
    output = output.astype("uint8")
    return output

### End of Helper Code ###

# Prep the sound to be played

# Load the the proper Aruco Dictionary
aruco_dict = aruco.Dictionary_get(DICT)
# Initialize detector params w/ defaults
parameters = aruco.DetectorParameters_create()
# Initialize Video file stream
vf = cv2.VideoCapture(FILE_PATH)
Q = deque(maxlen=128)
(grabbed, source) = vf.read()
Q.appendleft(source)
# Initialize Video steam
vs = VideoStream(src=0).start()

# Loop over frames of video stream
while len(Q) > 0:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    # Do the frame warp
    warped = find_and_warp(frame, source, MY_IDS, aruco_dict, parameters, useCache=USE_CACHE)
    # Check if all corners found
    if warped is not None:
        # print("WARP SUCCESS")
        frame = warped
        source = Q.popleft()
    # Get the next frames ready (ahead of time)
    if len(Q) != Q.maxlen:
        (grabbed, nextFrame) = vf.read()
        if grabbed:
            Q.append(nextFrame)
    # Display output
    cv2.imshow("Frame", frame)
    # Hit q key to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
# Release when finished
cv2.destroyAllWindows()
vs.stop()