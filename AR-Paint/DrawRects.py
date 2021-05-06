import cv2
import sys
import cv2.aruco as aruco
import numpy as np
import random

RECORD = False
DICT = aruco.DICT_ARUCO_ORIGINAL
COLOR = (0, 255, 255)
THICKNESS = 3
SQUARE_MULTIPLIER = 1.5
MAX_SHAPES = 30

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))

video_capture = cv2.VideoCapture(0)
if RECORD:
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter('DrawShapesV.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))
shape_array = []
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
    # Find Marker Coords!
    for i in range(len(corners)):
        top_left = int(corners[i][0][0][0]), int(corners[i][0][0][1])
        bot_right = int(corners[i][0][2][0]), int(corners[i][0][2][1])
        mid = (top_left[0] + bot_right[0])//2, (top_left[1] + bot_right[1])//2
        myID = ids[i][0]
        # top_right = corners[i][0][1]
        # bot_left = corners[i][0][3]
        # print("Top_Left:", top_left, "Top_Right:", bot_right, "Mid:", mid)
        # print("ID:", ids[i][0], "\nCorners:\n", corners[i][0], "\n")
        if len(shape_array) <= MAX_SHAPES:
            shape_array.append([top_left, bot_right, mid, myID])
        elif len(shape_array) > MAX_SHAPES:
            shape_array.pop(0)
        else:
            shape_array.pop(0)
            shape_array.append([top_left, bot_right, mid, myID])  
    # print("shape", shape_array)

    # Draw all shapes
    for i in range(len(shape_array)):
        # Make copy for transparency overlay
        overlay = frame.copy()
        alpha = 0.0 + (i*0.05)
        # Draw rectangels
        p0 = shape_array[i][0]
        p1 = shape_array[i][1]
        mid = shape_array[i][2]
        myID = shape_array[i][3]
        if myID == 10:
            cv2.rectangle(overlay, p0, p1, COLOR, THICKNESS)
        if myID == 20:
            cv2.circle(overlay, mid, 20, COLOR, THICKNESS)
        # Make overlay w/ transparent rectangles
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Display the resulting frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', frame)
    if RECORD:
        writer.write(frame)
    curr_frame += 1

    # Top Left, Top Right, Bot Right, Bot Left
    if cv2.waitKey(1) & 0xFF == ord('x'):
        COLOR = random_color()
    if cv2.waitKey(1) & 0xFF == ord('c'):
        shape_array = []
    # Hit q key to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
if RECORD:
    writer.release()
cv2.destroyAllWindows()