# 🛡️ Fall Detection and Pose Classification

A hybrid AI system that leverages **YOLOv8** and **MediaPipe** for real-time detection of falls, phone usage, and potentially unsafe worker behavior on industrial floors. Built for edge deployment on devices like the **Raspberry Pi 3**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Approaches](#approaches)
- [Hardware & Software](#hardware--software)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Challenges](#challenges)
- [Future Work](#future-work)
- [Contributors](#contributors)

---

## 🧠 Overview

This project aims to improve industrial safety by:

- Detecting **fall incidents**
- Identifying if a worker is **using a phone**
- Classifying poses like **standing, sitting, falling, or sleeping**

We achieve this by integrating:

- Object detection via **YOLOv8**
- Pose estimation via **MediaPipe**
- Custom logic to classify behavior based on pose landmarks

---

## 🔬 Approaches

### ⚙️ Approach 1 – Bounding Box Fall Detection

This approach relies on bounding box geometry during person detection:

- If a person’s bounding box suddenly shifts from **vertical to horizontal**, it's flagged as a potential fall.
- The width-to-height ratio is used to infer orientation.
- Limitation: Not reliable if person is partially visible or occluded.

### 🕺 Approach 2a – MediaPipe Pose Classification

This uses Google’s **MediaPipe** to extract 33 key body landmarks:

- Uses position and visibility of landmarks (e.g., shoulders, hips, nose).
- Custom rules classify posture:
  - Low head + no limb movement → **Sleeping**
  - Rapid vertical drop in hip/ankle Y → **Fall**
  - One hand near ear/head → **Phone call**
- Accurate in good lighting, but fails under occlusion.

### 🧍‍♂️ Approach 2b – YoloPose (Pose via YOLO)

YOLOPose integrates keypoint detection into YOLOv8 architecture:

- It predicts joint locations as an extension of person detection.
- Combines the speed of YOLO with pose estimation accuracy.
- Requires custom dataset and training setup.

### 📐 Approach 3 – Spine Vector Deviation

This novel technique computes the **spine angle** using:

- A vector from shoulder center to hip center.
- A tilt beyond a certain threshold implies a fall.
- More mathematically grounded and reliable for fall detection without relying on bounding boxes.

---

## 🧰 Hardware & Software

### 📦 Software

- Python 3.10+
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://google.github.io/mediapipe/)
- OpenCV
- NumPy
- Matplotlib

### 💻 Hardware

- Webcam (1080p)
- Raspberry Pi 3 (for edge deployment)

---

## 📁 Directory Structure

```
Fall-Detection-and-Pose-Classification/
│
├── Approach 1 - Bounding Box/
├── Approach 2a - Mediapipe/
├── Approach 2b - YoloPose/
├── Approach 3 - Spine Vector/
├── Output Images/
│
├── combined_yolo_&_mediapipe.py     # Main hybrid implementation
├── deploy_rpi3.py                   # Raspberry Pi deployment script
├── Implementation Report.docx       # Detailed technical report
└── README.md                        # This file
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/InvictusRex/Fall-Detection-and-Pose-Classification.git
cd Fall-Detection-and-Pose-Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Run YOLOv8 + MediaPipe hybrid detection:

```bash
python combined_yolo_&_mediapipe.py
```

### Deploy on Raspberry Pi:

```bash
python deploy_rpi3.py
```

---

## 📊 Results

- Detected falls using bounding box aspect ratio changes and spine vector tilt.
- Identified mobile usage using YOLOv8 object detection.
- Pose classification based on MediaPipe landmarks.
- Real-time performance tested on Raspberry Pi 3 (with optimization).

Sample output images are available in `/Output Images`.

---

## ⚠️ Challenges

- Lighting variation affects YOLO detection.
- MediaPipe fails in extreme body angles or occlusion.
- Limited dataset for edge cases (e.g., crawling, kneeling).
- Resource constraints on Raspberry Pi during concurrent model inference.

---

## 🔮 Future Work

- Improve pose classification with LSTM/Transformer-based temporal models.
- Extend detection to other safety gear (e.g., helmet, gloves).
- Alert system integration with MQTT or SMS.
- Deploy on NVIDIA Jetson Nano for better real-time performance.
