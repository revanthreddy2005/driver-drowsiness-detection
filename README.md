# Driver Drowsiness Detection System
A real-time computer vision system that monitors a driver through a webcam and detects signs of drowsiness.

The system uses facial landmarks, eye aspect ratio (EAR), and head movement analysis to trigger an alert when fatigue is detected.

## Demo

[Watch the demo video](demo/drowsiness_demo.mp4.mp4)
(Note: If audio does not autoplay in browser, download the video to hear the alert.)


## Features
- Real-time webcam monitoring
- Eye closure detection using EAR
- Head drop detection
- Audio alert system
- Manual reset support

## Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy
- Threading

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run the application:
python run.py

## Project Structure

driver-drowsiness-detection/
│
├── src/        # detection logic
├── assets/     # alarm audio
├── run.py      # entry point
├── requirements.txt
└── README.md

## How It Works

The system captures live video from the webcam and uses MediaPipe Face Mesh to extract facial landmarks.

Eye Aspect Ratio (EAR) is computed to determine whether the driver's eyes are closed for a prolonged duration.

A normalized head-drop ratio is also calculated to detect downward head movement.

If thresholds are exceeded for a number of consecutive frames, an alert sound is triggered using a background thread while detection continues in real time.

## Future Improvements

- Improve robustness under low-light conditions
- Add yawn detection
- Deploy on edge devices like Raspberry Pi
- Store drowsiness events for analytics
- Build a dashboard for fleet monitoring

## Author
Developed by Bommala Revanth Reddy
AI & Data Science Student
📧 saireddybommala2005@gmail.com
