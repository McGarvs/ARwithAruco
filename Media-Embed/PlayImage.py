import cv2
import sys
import cv2.aruco as aruco
import numpy as np
import os

DICT = aruco.DICT_ARUCO_ORIGINAL
COLOR = (0, 255, 255)
THICKNESS = 3
SQUARE_MULTIPLIER = 1.5

FILE_PATH = os.path.join(os.getcwd(), "Painting.png")
source = cv2.imread(FILE_PATH)
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

video_capture = cv2.VideoCapture(0)

curr_frame = 0
while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()


    # Check if frame is not empty
    if not ret:
        continue

    # # Convert from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Frame Dimensions
    (frameH, frameW) = frame.shape[:2]
    
    # Load the the proper Aruco Dictionary
    aruco_dict = aruco.Dictionary_get(DICT)

    # Initialize detector params w/ defaults
    parameters = aruco.DetectorParameters_create()

    # Detect the markers in the current frame
    corners, ids, rejects = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    # frame = aruco.drawDetectedMarkers(frame, corners, ids)

    if not( cv2.waitKey(1) == ord('s') ):
        if len(corners) == 4:
            # Flatten
            ids = ids.flatten()
            refPts = []
            for i in (800, 700, 40, 1000):
                j = np.squeeze(np.where(ids==i))
                corner = np.squeeze(corners[j])
                refPts.append(corner)
            # Unpack ref point
            (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
            dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
            dstMat = np.array(dstMat)
            # Grab source dims
            (srcH, srcW) = source.shape[:2]
            srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
            # Compute Homography
            (H, _) = cv2.findHomography(srcMat, dstMat)
            warped = cv2.warpPerspective(source, H, (frameW, frameH))
            # Construct mask
            mask = np.zeros((frameH,frameW), dtype="uint8")
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
            frame = cv2.add(warpMultiplied, frameMultiplied)
            frame = frame.astype("uint8")


    # print("Frame:", curr_frame, "Detected Ids:", ids)
    
    # Display the resulting frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', frame)
    curr_frame += 1

    # Hit q key to quit the window
    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()