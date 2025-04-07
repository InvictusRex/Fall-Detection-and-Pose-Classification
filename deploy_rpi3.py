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
            # Performance tracking
            self.frame_times = deque(maxlen=30)
            self.activity_history = deque(maxlen=30)
            
            # Joint mapping
            self.joint_indices = {
                'head': 0,
                'neck': 1,
                'shoulders': {'left': 11, 'right': 12},
                'elbows': {'left': 13, 'right': 14},
                'wrists': {'left': 15, 'right': 16},
                'hips': {'left': 23, 'right': 24},
                'knees': {'left': 25, 'right': 26},
                'ankles': {'left': 27, 'right': 28}
            }
            
            # Angle storage
            self.angles = {
                'sagittal': np.zeros(12),
                'frontal': np.zeros(7),
                'transverse': np.zeros(4)
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
            if frame is None:
                raise ValueError("Received None frame")
                
            if frame.size == 0:
                raise ValueError("Received empty frame")
                
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get pose estimation
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                logging.debug("Landmarks detected")
                
                # Extract landmarks
                landmarks = self._extract_landmarks(results.pose_landmarks)
                
                # Calculate angles
                angles = self._calculate_angles(landmarks)
                
                # Detect activity
                activity, confidence = self._detect_activity(angles)
                
                # Draw visualization
                annotated_frame = self._draw_debug_visualization(
                    frame.copy(), 
                    results.pose_landmarks,
                    activity,
                    confidence,
                    angles
                )
                
                return {
                    'frame': annotated_frame,
                    'activity': activity,
                    'confidence': confidence,
                    'angles': angles
                }
            
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
            
            if landmarks.shape != (33, 3):
                raise ValueError(f"Invalid landmarks shape: {landmarks.shape}")
                
            return landmarks
            
        except Exception as e:
            logging.error(f"Landmark extraction failed: {str(e)}", exc_info=True)
            raise

    def _calculate_angles(self, landmarks):
        """Calculate angles with validation"""
        try:
            angles = {
                'sagittal': self._calculate_sagittal_angles(landmarks),
                'frontal': self._calculate_frontal_angles(landmarks),
                'transverse': self._calculate_transverse_angles(landmarks)
            }
            
            # Validate angles
            for plane, values in angles.items():
                if not isinstance(values, np.ndarray):
                    raise ValueError(f"Invalid {plane} angles type: {type(values)}")
                    
            return angles
            
        except Exception as e:
            logging.error(f"Angle calculation failed: {str(e)}", exc_info=True)
            raise

    def _calculate_sagittal_angles(self, landmarks):
        """Calculate sagittal plane angles"""
        try:
            angles = np.zeros(12)
            
            # Head angle
            head_vector = landmarks[self.joint_indices['head']] - landmarks[self.joint_indices['neck']]
            angles[0] = np.arctan2(head_vector[1], head_vector[0]) * 180 / np.pi
            
            # Shoulder angles
            for i, side in enumerate(['left', 'right']):
                shoulder_idx = self.joint_indices['shoulders'][side]
                elbow_idx = self.joint_indices['elbows'][side]
                vec = landmarks[elbow_idx] - landmarks[shoulder_idx]
                angles[i+1] = np.arctan2(vec[1], vec[0]) * 180 / np.pi
            
            return np.clip(angles, -90, 90)
            
        except Exception as e:
            logging.error(f"Sagittal angle calculation failed: {str(e)}", exc_info=True)
            raise

    def _calculate_frontal_angles(self, landmarks):
        """Calculate frontal plane angles"""
        try:
            angles = np.zeros(7)
            
            # Shoulder angles
            for i, side in enumerate(['left', 'right']):
                shoulder_idx = self.joint_indices['shoulders'][side]
                elbow_idx = self.joint_indices['elbows'][side]
                vec = landmarks[elbow_idx] - landmarks[shoulder_idx]
                angles[i] = np.arctan2(vec[2], vec[1]) * 180 / np.pi
            
            return np.clip(angles, -45, 45)
            
        except Exception as e:
            logging.error(f"Frontal angle calculation failed: {str(e)}", exc_info=True)
            raise

    def _calculate_transverse_angles(self, landmarks):
        """Calculate transverse plane angles"""
        try:
            angles = np.zeros(4)
            
            # Head rotation
            head_vector = landmarks[self.joint_indices['head']] - landmarks[self.joint_indices['neck']]
            angles[0] = np.arctan2(head_vector[0], head_vector[2]) * 180 / np.pi
            
            return np.clip(angles, -60, 60)
            
        except Exception as e:
            logging.error(f"Transverse angle calculation failed: {str(e)}", exc_info=True)
            raise

    def _detect_activity(self, angles):
        """Detect activity with validation"""
        try:
            sagittal = angles['sagittal']
            frontal = angles['frontal']
            
            # Activity detection logic
            if np.abs(sagittal[0]) < 15 and np.abs(frontal[0]) < 10:
                activity = 'sleeping'
                confidence = 0.5
            elif np.any(np.abs(sagittal[4:8]) > 45):
                activity = 'running'
                confidence = 0.5
            elif np.abs(sagittal[1]) > 45 and np.abs(frontal[0]) > 30:
                activity = 'talking_on_phone'
                confidence = 0.4
            else:
                activity = 'other'
                confidence = 0.7
            
            self.activity_history.append(activity)
            confidence = (self.activity_history.count(activity) / len(self.activity_history)) * 100
            
            return activity, confidence
            
        except Exception as e:
            logging.error(f"Activity detection failed: {str(e)}", exc_info=True)
            return 'unknown', 0

    def _draw_debug_visualization(self, frame, landmarks, activity, confidence, angles):
        """Draw debug visualization with error handling"""
        try:
            # Draw skeleton
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Draw debug info
            debug_info = [
                f"Frame: {self.frame_count}",
                f"Activity: {activity} ({confidence:.1f}%)",
                f"Sagittal[0]: {angles['sagittal'][0]:.1f}°",
                f"Frontal[0]: {angles['frontal'][0]:.1f}°"
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

def main():
    logging.info("Starting main function")
    
    try:
        # Initialize pose estimator
        estimator = DebugPoseEstimator()
        
        # Try multiple camera indices
        camera_indices = [0, 1, 2]
        cap = None
        
        for idx in camera_indices:
            logging.info(f"Trying camera index {idx}")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                logging.info(f"Successfully opened camera {idx}")
                break
        
        if cap is None or not cap.isOpened():
            raise RuntimeError("Could not open any camera")
        
        # Create window
        cv2.namedWindow('Debug Pose Estimation', cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame")
                break
            
            # Process frame
            results = estimator.process_frame(frame)
            
            # Display frame
            cv2.imshow('Debug Pose Estimation', results['frame'])
            
            # Break on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("User requested exit")
                break
        
    except Exception as e:
        logging.error(f"Main loop error: {str(e)}", exc_info=True)
    
    finally:
        logging.info("Cleaning up...")
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        logging.info("Cleanup complete")

if __name__ == "__main__":
    main()