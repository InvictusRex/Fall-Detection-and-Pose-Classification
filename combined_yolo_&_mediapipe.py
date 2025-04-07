import cv2
import torch
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("E:\Work Files\Project - Worker Efficiency & Safety\Model - Phone Detection\src\Phone Detection Model\yolov8m.pt")  # Change to your trained model path

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)
    phone_detected = False
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            if "phone" in label.lower():
                phone_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Run Pose Estimation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    hand_near_face = False
    
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Check if hand is near face
        if abs(left_hand.x - nose.x) < 0.1 and abs(left_hand.y - nose.y) < 0.1:
            hand_near_face = True
        if abs(right_hand.x - nose.x) < 0.1 and abs(right_hand.y - nose.y) < 0.1:
            hand_near_face = True
    
    # Decision Logic
    if phone_detected and hand_near_face:
        cv2.putText(frame, "Using Phone", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    elif phone_detected:
        cv2.putText(frame, "Holding Phone", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    else:
        cv2.putText(frame, "No Phone Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    cv2.imshow("Phone Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
