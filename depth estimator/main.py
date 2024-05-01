import cv2
print(cv2.cuda.getCudaEnabledDeviceCount())
print(cv2.__version__)
import numpy as np
import simpleaudio as sa
import time
import triangulation as tri
from sounds import generate_sound
import undistort as calibration

# Load YOLO
net = cv2.dnn.readNet("yolo dependencies/yolov4-tiny.weights", "yolo dependencies/yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
classes = []
with open("yolo dependencies/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
# For OpenCV >= 4.0.0
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# If you encounter the error, try this instead:
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize both cameras
cap_right = cv2.VideoCapture(2)  # Camera 0
cap_left = cv2.VideoCapture(1)   # Camera 1

frame_rate = 15
B = 6.35 #cm
f = 4 #mm
alpha = 5 #degrees
prev = 0

while True:
    time_elapsed = time.time() - prev
    if time_elapsed > 1./frame_rate:  # Process only at specified FPS
        prev = time.time()

        success_right, frame_right = cap_right.read()
        success_left, frame_left = cap_left.read()

        if not success_right or not success_left:
            print("Failed to grab frames")
            break

        object_points_right = {}
        object_points_left = {}

        for frame, frame_name in [(frame_right, "Right Camera"), (frame_left, "Left Camera")]:
            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        label = classes[class_id]

                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        # Center of bounding box
                        center_point = (int(center_x), int(center_y))

                        if frame_name == "Right Camera":
                            object_points_right[label] = center_point
                        else:
                            object_points_left[label] = center_point

                        cv2.circle(frame, center_point, 5, (0, 255, 0), -1)  # Draw center point

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    color = np.random.uniform(0, 255, size=(3,))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

            cv2.imshow(frame_name, frame)

        for label in object_points_right:
            if label in object_points_left:
                depth = tri.find_depth(object_points_right[label], object_points_left[label], frame_right, frame_left, B, f, alpha)
                print(f"Detected '{label}' at depth: {round((depth * 0.0393701), 1)} inches")
                generate_sound(depth)  # Generate sound based on depth


        object_points_left.clear()
        object_points_right.clear()

        key = cv2.waitKey(1)
        if key == 27:  # ESC key to break
            break

# Release resources
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()