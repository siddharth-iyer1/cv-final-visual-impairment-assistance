# ECE 379K Computer Vision Final Project - Assistive Technology for the Visually Impaired

### Project Flow

1. **Camera Calibration**:
   - The `camera calibration` folder contains scripts and notebooks where we calibrated our stereo cameras
   - This step produces the camera matrices and stereo maps used in the depth estimation script.

2. **Object Detection**:
   - The `yoloModel` folder provides the necessary YOLO model files for object detection.
   - This YOLO setup, including the dependency files was brought into the depth estimation folder to perform object detection before triangulation.

3. **Depth Estimation**:
   - The `depth estimator` script integrates the calibrated camera matrices with the object detection results.
   - It computes the depth of detected objects using triangulation based on the disparity between the left and right images.

4. **Audio Feedback**:
   - The `audio` folder contains code for generating sound feedback based on the depth estimates.
   - The `depth estimator` script calls the audio script to produce sounds indicating the proximity of detected objects.

Demo Link: https://www.youtube.com/watch?v=qF5j_y0ehN4
