import numpy as np
import cv2 as cv
import glob

# Finding Chessboard Corners
chessboardSize = (7, 4)
frameSize = (1280, 960)

critera = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32) # xyz coordinates for each point in the real world
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# appended everytime we find a matching chessboard point
objpoints = []
imgpointsL = []
imgpointsR = []

imagesLeft = glob.glob('/Users/siddharthiyer/Documents/GitHub/cv-final-visual-impairment-assistance/camera calibration/checkboard calibration/images/left cam/*.png')
imagesRight = glob.glob('/Users/siddharthiyer/Documents/GitHub/cv-final-visual-impairment-assistance/camera calibration/checkboard calibration/images/right cam/*.png')

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    if retL == True and retR == True:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), critera)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), critera)
        imgpointsR.append(cornersR)

        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('imgL', imgL)
        cv.waitKey(500)

        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('imgR', imgR)
        cv.waitKey(500)

cv.destroyAllWindows()

# Camera Matrix Calibration
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

# Stereo Vision Calibration
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, (widthL, heightL), criteria_stereo, flags)

# Stereo Rectification

rectifyScale = 0
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale, (0, 0))

stereoMapLx, stereoMapLy = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapRx, stereoMapRy = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

cv_file = cv.FileStorage('/Users/siddharthiyer/Documents/GitHub/cv-final-visual-impairment-assistance/camera calibration/checkboard calibration/calibration_data.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x', stereoMapLx)
cv_file.write('stereoMapL_y', stereoMapLy)
cv_file.write('stereoMapR_x', stereoMapRx)
cv_file.write('stereoMapR_y', stereoMapRy)

cv_file.release()

print("done")