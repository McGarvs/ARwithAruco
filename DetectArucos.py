import cv2
import sys
import cv2.aruco as aruco
import numpy as np

DICT = aruco.DICT_ARUCO_ORIGINAL
COLOR = (0, 255, 255)
THICKNESS = 3
SQUARE_MULTIPLIER = 1.5


video_capture = cv2.VideoCapture(0)

rect_array = []
curr_frame = 0
counter = 0
while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Check if frame is not empty
    if not ret:
        continue

    # Convert from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Load the the proper Aruco Dictionary
    aruco_dict = aruco.Dictionary_get(DICT)

    # Initialize detector params w/ defaults
    parameters = aruco.DetectorParameters_create()

    # Detect the markers in the current frame
    corners, ids, rejects = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    frame = aruco.drawDetectedMarkers(frame, corners, ids)

    print("Frame:", curr_frame, "Detected Ids:", ids)
    
    # Display the resulting frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', frame)
    curr_frame += 1

    # print(corners) # Top Left, Top Right, Bot Right, Bot Left
    if cv2.waitKey(1) & 0xFF == ord('c'):
        rect_array = []
    # Hit q key to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()