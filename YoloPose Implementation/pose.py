from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO models
pose_model = YOLO("yolov8n-pose.pt")  # Pretrained for pose estimation
phone_model = YOLO("yolov8n.pt")  # Pretrained for object detection (modify if custom trained)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO object detection for phone detection
    phone_results = phone_model(frame)
    phone_detected = any(d for d in phone_results[0].boxes if d.cls == "phone")

    # Run YOLO pose estimation
    pose_results = pose_model(frame)

    for r in pose_results:
        keypoints = r.keypoints.xy if r.keypoints is not None else None

        if keypoints is not None and len(keypoints) >= 11:
            head = keypoints[0]  # Head position
            left_hand, right_hand = keypoints[9], keypoints[10]  # Hands
            left_foot, right_foot = keypoints[15], keypoints[16]  # Feet

            status = "Idle"

            # 1️⃣ **Phone Usage Detection**
            if phone_detected:
                if abs(head[1] - left_hand[1]) < 50 or abs(head[1] - right_hand[1]) < 50:
                    status = "On Call"
                else:
                    status = "Scrolling"

            # 2️⃣ **Sleeping Detection**
            elif head[1] > 400:  # Adjust threshold as needed
                status = "Sleeping"

            # 3️⃣ **Running Detection**
            elif abs(left_foot[0] - right_foot[0]) > 50:  # Feet apart indicates running
                status = "Running"

            # Display status
            cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw keypoints
        frame = r.plot()

    cv2.imshow("Worker Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
