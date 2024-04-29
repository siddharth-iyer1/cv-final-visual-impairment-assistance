import numpy as np
import cv2

# Load the camera calibration data
cv_file = cv2.FileStorage()
cv_file.open("calibration_data.xml", cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
stereoMapR_x = cv_file.getNode("stereoMapR_x").mat()
stereoMapR_y = cv_file.getNode("stereoMapR_y").mat()

# Open Cameras
cap_right = cv2.VideoCapture(1)
cap_left = cv2.VideoCapture(0)

while(cap_left.isOpened and cap_right.isOpened):
    retR, frameR = cap_right.read()
    retL, frameL = cap_left.read()

    # frame_right = cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    # frame_left = cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    cv2.imshow('frame_right', frameR)
    cv2.imshow('frame_left', frameL)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()


