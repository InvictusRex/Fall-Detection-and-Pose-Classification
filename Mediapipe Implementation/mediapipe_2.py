import mediapipe as mp
import numpy as np
import cv2
import sys
import logging
from datetime import datetime
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pose_estimation_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class PoseEstimator:
    def __init__(self):
        try:
            logging.info("Initializing PoseEstimator...")
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.setup_parameters()
            logging.info("PoseEstimator initialized successfully.")
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}", exc_info=True)
            raise

    def setup_parameters(self):
        """Initialize tracking parameters"""
        self.frame_times = deque(maxlen=30)
        self.activity_history = deque(maxlen=30)
        self.joint_indices = {
            'head': 0, 'neck': 1,
            'shoulders': {'left': 11, 'right': 12},
            'elbows': {'left': 13, 'right': 14},
            'wrists': {'left': 15, 'right': 16},
            'hips': {'left': 23, 'right': 24},
            'knees': {'left': 25, 'right': 26},
            'ankles': {'left': 27, 'right': 28}
        }
        logging.info("Tracking parameters initialized.")

    def process_frame(self, frame):
        """Process frame and return annotated frame and detected activity"""
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame received.")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                angles = self._calculate_angles(landmarks)
                activity, confidence = self._detect_activity(angles)
                annotated_frame = self._draw_debug_visualization(frame.copy(), results.pose_landmarks, activity, confidence)
                return {'frame': annotated_frame, 'activity': activity, 'confidence': confidence}

            return {'frame': frame}
        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}", exc_info=True)
            return {'frame': frame}

    def _calculate_angles(self, landmarks):
        """Calculate joint angles"""
        try:
            angles = np.zeros(3)
            head_vec = landmarks[self.joint_indices['head']] - landmarks[self.joint_indices['neck']]
            angles[0] = np.arctan2(head_vec[1], head_vec[0]) * 180 / np.pi
            return np.clip(angles, -90, 90)
        except Exception as e:
            logging.error(f"Angle calculation failed: {str(e)}", exc_info=True)
            return np.zeros(3)

    def _detect_activity(self, angles):
        """Detect activity from angles"""
        try:
            activity = 'other'
            confidence = 50
            if np.abs(angles[0]) < 15:
                activity = 'neutral'
                confidence = 80
            elif np.abs(angles[0]) > 45:
                activity = 'looking_down'
                confidence = 70
            return activity, confidence
        except Exception as e:
            logging.error(f"Activity detection failed: {str(e)}", exc_info=True)
            return 'unknown', 0

    def _draw_debug_visualization(self, frame, landmarks, activity, confidence):
        """Draw debug information on frame"""
        try:
            self.mp_draw.draw_landmarks(frame, landmarks, self.mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Activity: {activity} ({confidence:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame
        except Exception as e:
            logging.error(f"Visualization error: {str(e)}", exc_info=True)
            return frame

def main():
    logging.info("Starting pose estimation...")
    try:
        estimator = PoseEstimator()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera.")
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame.")
                break
            results = estimator.process_frame(frame)
            cv2.imshow('Pose Estimation', results['frame'])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        logging.error(f"Runtime error: {str(e)}", exc_info=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Cleanup complete.")

if __name__ == "__main__":
    main()
