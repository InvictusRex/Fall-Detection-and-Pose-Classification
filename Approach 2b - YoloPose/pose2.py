import cv2
import torch
import numpy as np
import logging
from collections import deque
from ultralytics import YOLO  # YOLOv8

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_pose_debug.log'),
        logging.StreamHandler()
    ]
)

class YOLOPoseEstimator:
    def __init__(self):
        try:
            logging.info("Initializing YOLOPoseEstimator...")

            # Load YOLOv8 Pose Model
            self.model = YOLO("yolov8n-pose.pt")  # Using YOLOv8n-pose model
            logging.info("YOLO model loaded successfully")

            # Tracking variables
            self.frame_count = 0
            self.activity_history = deque(maxlen=30)

        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}", exc_info=True)
            raise

    def process_frame(self, frame):
        """Process frame using YOLO-Pose"""
        self.frame_count += 1
        logging.debug(f"Processing frame {self.frame_count}")

        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame received")

            # Run YOLO-Pose inference
            results = self.model(frame)

            # Extract keypoints
            if len(results) > 0 and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy.cpu().numpy()[0]  # Extract 17 keypoints
                logging.debug("Keypoints detected")

                # Calculate angles
                angles = self._calculate_angles(keypoints)

                # Detect activity
                activity, confidence = self._detect_activity(angles)

                # Draw keypoints & skeleton
                annotated_frame = self._draw_debug_visualization(frame, keypoints, activity, confidence)

                return {
                    'frame': annotated_frame,
                    'activity': activity,
                    'confidence': confidence,
                    'angles': angles
                }

            else:
                logging.debug("No keypoints detected")
                return {'frame': frame}

        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}", exc_info=True)
            return {'frame': frame}

    def _calculate_angles(self, keypoints):
        """Calculate joint angles"""
        try:
            angles = np.zeros(3)

            # Head tilt (between nose and shoulders)
            nose, left_shoulder, right_shoulder = keypoints[0], keypoints[5], keypoints[6]
            angles[0] = np.arctan2(left_shoulder[1] - nose[1], left_shoulder[0] - nose[0]) * 180 / np.pi

            # Arm angles (shoulder to wrist)
            left_elbow, left_wrist = keypoints[7], keypoints[9]
            right_elbow, right_wrist = keypoints[8], keypoints[10]

            angles[1] = np.arctan2(left_wrist[1] - left_elbow[1], left_wrist[0] - left_elbow[0]) * 180 / np.pi
            angles[2] = np.arctan2(right_wrist[1] - right_elbow[1], right_wrist[0] - right_elbow[0]) * 180 / np.pi

            return angles

        except Exception as e:
            logging.error(f"Angle calculation failed: {str(e)}", exc_info=True)
            return np.zeros(3)

    def _detect_activity(self, angles):
        """Detect activity based on angles"""
        try:
            if abs(angles[0]) > 30:
                activity = "sleeping"
            elif abs(angles[1]) > 60 or abs(angles[2]) > 60:
                activity = "talking_on_phone"
            else:
                activity = "standing"

            self.activity_history.append(activity)
            confidence = (self.activity_history.count(activity) / len(self.activity_history)) * 100

            return activity, confidence

        except Exception as e:
            logging.error(f"Activity detection failed: {str(e)}", exc_info=True)
            return 'unknown', 0

    def _draw_debug_visualization(self, frame, keypoints, activity, confidence):
        """Draw YOLO keypoints and activity status"""
        try:
            for kp in keypoints:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

            cv2.putText(
                frame, f"Activity: {activity} ({confidence:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            return frame

        except Exception as e:
            logging.error(f"Visualization failed: {str(e)}", exc_info=True)
            return frame

def main():
    logging.info("Starting main function")
    
    try:
        estimator = YOLOPoseEstimator()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Camera not found")

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame")
                break

            results = estimator.process_frame(frame)

            cv2.imshow('YOLO Pose Estimation', results['frame'])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logging.error(f"Main loop error: {str(e)}", exc_info=True)

    finally:
        logging.info("Cleaning up...")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        logging.info("Cleanup complete")

if __name__ == "__main__":
    main()
