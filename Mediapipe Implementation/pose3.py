import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize Pose Model
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

prev_left_ankle_y = None
prev_right_ankle_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    action = "Standing"

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

            # Get landmark coordinates
        left_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y])
        right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y])
        left_ear = np.array([landmarks[mp_pose.PoseLandmark.LEFT_EAR].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR].y])
        right_ear = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y])
        nose = landmarks[mp_pose.PoseLandmark.NOSE].y
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        avg_shoulder = (left_shoulder + right_shoulder) / 2
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y

            # Talking on Phone - Wrist close to Ear
        left_distance = np.linalg.norm(left_wrist - left_ear)
        right_distance = np.linalg.norm(right_wrist - right_ear)

        if left_distance < 0.05 or right_distance < 0.05:  # Adjust as needed
            action = "Talking on Phone"

            # Sleeping - Nose below Shoulders
        elif nose > avg_shoulder:
            action = "Fall Detected"

            # Running - Large difference in ankle movement
        elif prev_left_ankle_y is not None and prev_right_ankle_y is not None:
            ankle_movement = abs(left_ankle - prev_left_ankle_y) + abs(right_ankle - prev_right_ankle_y)
            if ankle_movement > 0.02:  # Adjust threshold for movement
                action = "Standing"

            # Update previous frame data
        prev_left_ankle_y = left_ankle
        prev_right_ankle_y = right_ankle

        # Display detected action
    cv2.putText(image, f"Action: {action}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Pose-Based Action Detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
