import numpy as np
import cv2
import glob

checkerboard_size = (8, 6)
frame_size = (1280, 960)

termination_critera = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# object points
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

objp = objp * 28 # 28 mm between each square
print(objp)

objpoints = []  # real world
imgpointsL = [] # left
imgpointsR = []

left_images = glob.glob('/Users/siddharthiyer/Documents/GitHub/cv-final-visual-impairment-assistance/camera calibration/checkboard calibration/images/left cam/*.png')
right_images = glob.glob('/Users/siddharthiyer/Documents/GitHub/cv-final-visual-impairment-assistance/camera calibration/checkboard calibration/images/right cam/*.png')

for imgLeft, imgRight in zip(left_images, right_images):
    imgL = cv2.imread(imgLeft)
    imgR = cv2.imread(imgRight)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, checkerboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, checkerboard_size, None)

    if retL and retR:
        objpoints.append(objp)

        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), termination_critera)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), termination_critera)

        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

        imgL = cv2.drawChessboardCorners(imgL, checkerboard_size, cornersL, retL)
        imgR = cv2.drawChessboardCorners(imgR, checkerboard_size, cornersR, retR)

        cv2.imshow('imgL', imgL)
        cv2.imshow('imgR', imgR)
        cv2.waitKey(500)

cv2.destroyAllWindows()