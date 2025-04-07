import cv2
import cvzone
import math
from ultralytics import YOLO

# Load YOLO models
person_model = YOLO('yolov8m.pt')  # Default model for person detection
phone_model = YOLO(r"E:\Work Files\Project - Worker Efficiency & Safety\Model - Phone Detection\src\runs\detect\model_optimized2\weights\best.pt")  # Custom model for phone detection

cap = cv2.VideoCapture(0)  # Use laptop webcam
cap.set(3, 980)  # Set width
cap.set(4, 740)  # Set height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    person_results = person_model(frame)
    phone_results = phone_model(frame)

    person_boxes = []
    phone_boxes = []

    # Detect persons
    for info in person_results:
        for box in info.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100)
            if conf > 80:
                person_boxes.append((x1, y1, x2, y2))
                cvzone.cornerRect(frame, [x1, y1, x2 - x1, y2 - y1], l=30, rt=6)
                cvzone.putTextRect(frame, 'Person', [x1 + 8, y1 - 12], thickness=2, scale=2)

    # Detect mobile phones using the trained model
    for info in phone_results:
        for box in info.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100)
            if conf > 80:
                phone_boxes.append((x1, y1, x2, y2))
                cvzone.cornerRect(frame, [x1, y1, x2 - x1, y2 - y1], l=30, rt=6, colorR=(0, 255, 0))
                cvzone.putTextRect(frame, 'Phone', [x1 + 8, y1 - 12], thickness=2, scale=2)

    # Check if a phone is inside a person's bounding box
    for px1, py1, px2, py2 in person_boxes:
        for fx1, fy1, fx2, fy2 in phone_boxes:
            if fx1 > px1 and fy1 > py1 and fx2 < px2 and fy2 < py2:  # Phone inside person's box
                cvzone.putTextRect(frame, 'Phone in Hand!', [px1, py1 - 30], thickness=2, scale=2, colorR=(0, 0, 255))

    cv2.imshow('Live Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit

cap.release()
cv2.destroyAllWindows()
