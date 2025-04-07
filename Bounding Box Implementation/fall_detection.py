import cv2
import cvzone
import math
from ultralytics import YOLO

# Load video input
cap = cv2.VideoCapture("E:\\Work Files\\Project - Worker Efficiency & Safety\\Model - Fall Detection\\src\\fall.mp4")  # Replace with your video file
cap.set(3, 980)  # Set width
cap.set(4, 740)  # Set height

# Load YOLO model
model = YOLO('yolov8m.pt')

# Load class names
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Set up video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("E:\\Work Files\\Project - Worker Efficiency & Safety\\Model - Fall Detection\\src\\fall_output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends

    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = int(box.cls[0])
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect} ({conf}%)', [x1 + 8, y1 - 12], thickness=2, scale=2)
            
            if threshold < 0:
                cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 30], thickness=2, scale=2)

    out.write(frame)  # Save frame to output video

    # Display the frame in real-time
    cv2.imshow('Fall Detection', frame)
    
    # Introduce delay based on video FPS for smoother playback
    if cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF == ord('q'):
        break  # Press 'q' to exit

cap.release()
out.release()
cv2.destroyAllWindows()
