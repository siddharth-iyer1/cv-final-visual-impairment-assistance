import imutils
from matplotlib import pyplot as plt
import triangulation as tri
import undistort as calibrate

import time

print('done')

# import cv2

# cap1 = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(1)

# num_pics = 0

# while cap1.isOpened() and cap2.isOpened():

#     ret1, frame1 = cap1.read()
#     ret2, frame2 = cap2.read()

#     k = cv2.waitKey(5)

#     if k == 27:
#         break
#     elif k == ord('s'):
#         num_pics += 1
#         cv2.imwrite(f'/Users/siddharthiyer/Documents/GitHub/cv-final-visual-impairment-assistance/camera calibration/checkboard calibration/images/left cam/' + str(num_pics) + '.png', frame2)
#         cv2.imwrite(f'/Users/siddharthiyer/Documents/GitHub/cv-final-visual-impairment-assistance/camera calibration/checkboard calibration/images/right cam/' + str(num_pics) + '.png', frame1)
#         print("images saved")

#     cv2.imshow('frame1', frame1)
#     cv2.imshow('frame2', frame2)

# cap1.release()
# cap2.release()

# cv2.destroyAllWindows()