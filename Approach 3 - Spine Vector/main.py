import mediapipe as mp
import numpy as np
import cv2
import sys
import logging
import threading
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

class DebugPoseEstimator:
    def __init__(self):
        try:
            logging.info("Initializing DebugPoseEstimator...")
            
            # MediaPipe initialization
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logging.info("MediaPipe initialized successfully")
            
            # Initialize parameters
            self.setup_parameters()
            self.frame_count = 0
            logging.info("Parameter setup complete")
            
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}", exc_info=True)
            raise

    def setup_parameters(self):
        """Initialize tracking parameters with error checking"""
        try:
            self.frame_times = deque(maxlen=30)
            self.activity_history = deque(maxlen=30)
            self.velocity_history = deque(maxlen=10)
            
            self.joint_indices = {
                "right_shoulder": 11,
                "left_shoulder": 12,
                "right_hip": 23,
                "left_hip": 24,
                "head": 0
            }
            
            logging.info("Parameters initialized successfully")
            
        except Exception as e:
            logging.error(f"Parameter setup failed: {str(e)}", exc_info=True)
            raise

    def process_frame(self, frame):
        """Process frame with detailed error checking"""
        self.frame_count += 1
        logging.debug(f"Processing frame {self.frame_count}")
        
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame received")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                logging.debug("Landmarks detected")
                landmarks = self._extract_landmarks(results.pose_landmarks)
                activity, confidence = self._detect_activity(landmarks)
                
                annotated_frame = self._draw_debug_visualization(
                    frame.copy(), results.pose_landmarks, activity, confidence
                )
                
                return {'frame': annotated_frame, 'activity': activity, 'confidence': confidence}
            else:
                logging.debug("No landmarks detected")
                return {'frame': frame}
                
        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}", exc_info=True)
            return {'frame': frame}

    def _extract_landmarks(self, pose_landmarks):
        """Extract landmarks with validation"""
        try:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark])
            if landmarks.shape[0] < 25:
                raise ValueError(f"Invalid landmarks shape: {landmarks.shape}")
            return landmarks
        except Exception as e:
            logging.error(f"Landmark extraction failed: {str(e)}", exc_info=True)
            raise

    def _detect_activity(self, landmarks):
        """Detect activity with improved fall detection"""
        try:
            shoulder_mid = (landmarks[self.joint_indices["right_shoulder"]] + 
                            landmarks[self.joint_indices["left_shoulder"]]) / 2
            hip_mid = (landmarks[self.joint_indices["right_hip"]] + 
                       landmarks[self.joint_indices["left_hip"]]) / 2
            head = landmarks[self.joint_indices["head"]]
            
            spinal_vector = hip_mid - shoulder_mid
            spinal_vector /= np.linalg.norm(spinal_vector)  # Normalize
            
            vertical_axis = np.array([0, -1, 0])  # Y-axis (negative because y increases downward)
            cosine_angle = np.dot(spinal_vector, vertical_axis)
            angle_from_vertical = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * (180 / np.pi)
            
            head_height = head[1]
            hip_height = hip_mid[1]
            
            is_falling = angle_from_vertical > 70 and head_height > hip_height
            confidence = (1 / (1 + np.exp(-((angle_from_vertical - 50) / 10)))) * 100
            activity = "FALLING" if is_falling else "Standing"
            
            return activity, confidence
        except Exception as e:
            logging.error(f"Activity detection failed: {str(e)}", exc_info=True)
            return "Unknown", 0

    def _draw_debug_visualization(self, frame, landmarks, activity, confidence):
        """Draw debug visualization with error handling"""
        try:
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            debug_info = [
                f"Frame: {self.frame_count}",
                f"Activity: {activity} ({confidence:.1f}%)"
            ]
            
            for i, text in enumerate(debug_info):
                cv2.putText(
                    frame, text, (10, 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2
                )
            
            return frame
        except Exception as e:
            logging.error(f"Visualization failed: {str(e)}", exc_info=True)
            return frame

class VideoCaptureThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.ret = False
        self.frame = None
    
    def run(self):
        while True:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.cap.release()

if __name__ == "__main__":
    cap_thread = VideoCaptureThread()
    cap_thread.start()
    pose_estimator = DebugPoseEstimator()
    while True:
        ret, frame = cap_thread.read()
        if not ret:
            continue
        result = pose_estimator.process_frame(frame)
        cv2.imshow("Pose Estimation", result['frame'])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap_thread.release()
    cv2.destroyAllWindows()
