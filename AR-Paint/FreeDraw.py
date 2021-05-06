import cv2
import sys
import cv2.aruco as aruco
import numpy as np

RECORD = False
DICT = aruco.DICT_ARUCO_ORIGINAL
COLOR = (0, 255, 255)
THICKNESS = 3


video_capture = cv2.VideoCapture(0)
if RECORD:
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter('FreeDrawV.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))

line_array = []
curr_frame = 0
counter = 0
while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

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
        # print("Top_Left:", top_left, "Top_Right:", bot_right, "Mid:", mid)
        # print("ID:", ids[i][0], "\nCorners:\n", corners[i][0], "\n")
        if len(line_array) <= 200:
            line_array.append(mid)
        elif len(line_array) > 200:
            line_array.pop(0)
        else:
            line_array.pop(0)
            line_array.append(mid)  
    # print("Line", mid)

    # Draw lines
    lines = []
    for i in range(len(line_array)):
        if i == 0:
            continue
        else:
            # Draw Lines
            p0 = line_array[i]
            p1 = line_array[i-1]
            lines.append([p0, p1])
            # cv2.rectangle(frame, p0, p1, COLOR, THICKNESS)
            # cv2.line(frame, p0, p1, COLOR, THICKNESS)

    tmp = np.array(lines, np.int32)
    tmp = tmp.reshape((-1,1,2))
    cv2.polylines(frame, [tmp], False, COLOR, THICKNESS)

    # Display the resulting frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', frame)
    if RECORD:
        writer.write(frame)
    curr_frame += 1

    # Hit c to clear drawings from the frame
    if cv2.waitKey(1) & 0xFF == ord('c'):
        line_array = []

    # Hit q key to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
if RECORD:
    writer.release()
cv2.destroyAllWindows()