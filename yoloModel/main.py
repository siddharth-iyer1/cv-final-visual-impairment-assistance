import cv2
print(cv2.cuda.getCudaEnabledDeviceCount())
print(cv2.__version__)
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize both cameras
cap_right = cv2.VideoCapture(0)  # Camera 0
cap_left = cv2.VideoCapture(1)   # Camera 1

# Check if the cameras are opened successfully
if not cap_right.isOpened():
    print("Error opening right camera")
    exit()

if not cap_left.isOpened():
    print("Error opening left camera")
    exit()

print("Both cameras are opened successfully")

frame_rate = 0.5
prev = 0

while True:
    time_elapsed = time.time() - prev
    if time_elapsed > 1./frame_rate:
        prev = time.time()

        ret_right, frame_right = cap_right.read()
        ret_left, frame_left = cap_left.read()

        if not ret_right:
            print("Error reading frame from right camera")
            break

        if not ret_left:
            print("Error reading frame from left camera")
            break

        # Process each frame
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
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = np.random.uniform(0, 255, size=(3,))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

            cv2.imshow(frame_name, frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()