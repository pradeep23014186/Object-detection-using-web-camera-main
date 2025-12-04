# Object_detection_using_web_camera
## Aim:
To develop a program that performs real-time object detection using a web camera.

## Procedure:

1. Import cv2 (OpenCV) and numpy for image and numerical operations.

2. Load YOLOv4 Model

3. Access the Web Camera using cv2.VideoCapture(0) for real-time video streaming.

4. Normalize pixel values and resize the frame to fit the YOLOv4 input format.

5. Perform Object Detection

6. Pass the preprocessed blob to the YOLO network.

7. Extract the detected bounding boxes, class labels, and confidence scores.

8. Capture and Save Frames

9. Press ‘s’ to save a snapshot of the current frame with detected objects.

10. Press ‘q’ to quit the application.

## Program
```py
import cv2
import numpy as np

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set up video capture for webcam
cap = cv2.VideoCapture(0)

img_counter = 0  # Counter for saved images

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the image for YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO output
    outputs = net.forward(output_layers)

    # Initialize lists to store detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of the box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the image with detected objects
    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('s'):  # Press 's' to save the current frame
        img_name = f"detected_image_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved!")
        img_counter += 1

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
```

## Output:
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/25d91806-c652-49b5-8947-eae725e987a1" />

## Result
Thus the program to perform real-time object detection using a web camera is successfully executed. 
