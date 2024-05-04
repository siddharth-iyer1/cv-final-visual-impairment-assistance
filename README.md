# ECE 379K Computer Vision Final Project - Assitive Technology for the Visually Impaired

Project Flow
Camera Calibration:
The camera calibration folder contains scripts for calibrating the stereo cameras.
This step produces the camera matrices and stereo maps necessary for accurate depth estimation.
Object Detection:
The yoloModel folder provides the necessary YOLO model files for object detection.
The depth estimator script uses these files to detect objects and create bounding boxes around them.
Depth Estimation:
The depth estimator script integrates the calibrated camera matrices with the object detection results.
It computes the depth of detected objects using triangulation based on the disparity between the left and right images.
Audio Feedback:
The audio folder contains code for generating sound feedback based on the depth estimates.
The depth estimator script calls the audio script to produce sounds indicating the proximity of detected objects.

Demo Link: https://www.youtube.com/watch?v=qF5j_y0ehN4
