# YOLOv8 Human Pose Wrist Keypoint Detection

Real-time wrist keypoint detection using YOLOv8 pose estimation and OpenCV.

---

## Overview

This project uses the YOLOv8 pose model to detect human body keypoints and extract wrist coordinates from input images.

The detected wrist points are:
- highlighted on the image
- exported as JSON data
- saved as output files

---

## Features

- YOLOv8 pose estimation
- Wrist keypoint extraction
- Annotated output image
- JSON export support
- Organized project structure

---

## Tech Stack

- Python
- OpenCV
- YOLOv8
- Ultralytics

---

## Project Structure

```text
yolov8-hand-keypoint-detection/
│
├── assets/
├── docs/
├── models/
├── outputs/
├── src/
│   └── hand_pose.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/atulsharma47/yolov8-hand-keypoint-detection.git
```

Move into the project directory:

```bash
cd yolov8-hand-keypoint-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the project:

```bash
python src/hand_pose.py
```

---

## Output

The project generates:

- Annotated output image
- Wrist coordinate JSON file

Saved inside:

```text
outputs/
```

---

## Sample Output

<p align="center">
  <img src="outputs/annotated_image.jpg" width="500"/>
</p>

---

## Future Improvements

- Real-time webcam detection
- Multi-person tracking
- Gesture recognition
- Streamlit web app

---

## Author

Atul Sharma

GitHub: https://github.com/atulsharma47
