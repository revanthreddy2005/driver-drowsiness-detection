# drowsiness_detector.py
# EAR-based drowsiness detector with smoothing + normalized head-drop detection (inverted ratio handling)
# and looping "ALERT ALERT ALERT" voice alarm.

import cv2
import mediapipe as mp
import numpy as np
import threading
from playsound import playsound
import os
import time

# --------- CONFIGURATION (tune these) ----------
EYE_CLOSED_RATIO_THRESHOLD = 0.18
SMOOTHING_ALPHA = 0.25
MICROSLEEP_FRAMES = 40                # Eyes closed long -> alarm
HEAD_DROP_FRAMES = 30                  # frames of (EAR low + head dropped) to trigger faster alarm
# NOTE: For your camera we observed inverted behavior; use lower threshold and inverted comparison
HEAD_DROP_THRESHOLD_RATIO = 0.40       # normalized threshold (chin-nose / face-height)
NO_FACE_TOLERANCE = 12                 # Frames allowed with face missing
ALERT_FILE = "assets/alert_voice_alert.wav"   # your voice alert file (must be in project folder)
# Mediapipe eye & face landmark indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
# Outer eye indices for eye-center
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_IDX = 1
CHIN_IDX = 152
# ------------------------------------------------

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Alarm thread controls
_alert_thread_running = False
_alert_thread_lock = threading.Lock()

def _alert_loop(path):
    """Repeatedly play alert audio until stopped."""
    global _alert_thread_running
    try:
        while True:
            with _alert_thread_lock:
                if not _alert_thread_running:
                    break
            try:
                playsound(path)
            except:
                time.sleep(0.25)
    finally:
        with _alert_thread_lock:
            _alert_thread_running = False

def start_alert_loop(path):
    """Start alarm sound loop (idempotent)."""
    global _alert_thread_running
    with _alert_thread_lock:
        if _alert_thread_running:
            return
        _alert_thread_running = True
    t = threading.Thread(target=_alert_loop, args=(path,), daemon=True)
    t.start()

def stop_alert_loop():
    """Stop alarm loop."""
    global _alert_thread_running
    with _alert_thread_lock:
        _alert_thread_running = False

def eye_aspect_ratio(landmarks, eye_idx, img_w, img_h):
    """Compute EAR using 6 landmarks for an eye (mediapipe normalized landmarks)."""
    pts = []
    for idx in eye_idx:
        lm = landmarks[idx]
        pts.append((int(lm.x * img_w), int(lm.y * img_h)))
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def compute_head_drop_ratio_stable(face_landmarks, img_h):
    """
    Stable normalized head-drop:
      - eye_center_y = midpoint of outer eye landmarks
      - nose_y, chin_y in pixels
      - face_height = chin_y - eye_center_y
      - drop_px = chin_y - nose_y
      - drop_ratio = drop_px / face_height
    Return (drop_px, drop_ratio)
    """
    nose = face_landmarks[NOSE_IDX]
    chin = face_landmarks[CHIN_IDX]
    left_eye_outer = face_landmarks[LEFT_EYE_OUTER]
    right_eye_outer = face_landmarks[RIGHT_EYE_OUTER]

    nose_y = float(nose.y * img_h)
    chin_y = float(chin.y * img_h)
    eye_center_y = float(((left_eye_outer.y + right_eye_outer.y) / 2.0) * img_h)

    face_height = max(1.0, chin_y - eye_center_y)
    drop_px = chin_y - nose_y
    drop_ratio = drop_px / face_height

    return drop_px, drop_ratio

def main():
    # check audio presence
    if not os.path.exists(ALERT_FILE):
        print(f"WARNING: '{ALERT_FILE}' not found in project folder. Program will run visually but no audio will play.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    smoothed_ear = None
    closed_eye_frames = 0
    closed_head_frames = 0
    no_face_frames = 0
    alarm_on = False

    print("SYSTEM READY — Press ESC to exit, 'r' to reset alarm manually")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from webcam.")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                no_face_frames = 0
                face_landmarks = results.multi_face_landmarks[0].landmark

                # EAR calculation
                left_ear_raw = eye_aspect_ratio(face_landmarks, LEFT_EYE_IDX, w, h)
                right_ear_raw = eye_aspect_ratio(face_landmarks, RIGHT_EYE_IDX, w, h)
                ear_raw = (left_ear_raw + right_ear_raw) / 2.0

                if smoothed_ear is None:
                    smoothed_ear = ear_raw
                else:
                    smoothed_ear = SMOOTHING_ALPHA * ear_raw + (1 - SMOOTHING_ALPHA) * smoothed_ear

                ear = smoothed_ear

                # Stable head-drop ratio
                drop_px, drop_ratio = compute_head_drop_ratio_stable(face_landmarks, h)

                # *** INVERTED-RATIO HANDLING FOR YOUR CAMERA ***
                # Your camera produces smaller ratio when head tilts DOWN,
                # so treat smaller ratio as "head down".
                head_dropped = drop_ratio < HEAD_DROP_THRESHOLD_RATIO

                # draw debug info
                cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.putText(frame, f"DROP: {int(drop_px)}px", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
                cv2.putText(frame, f"RATIO: {drop_ratio:.2f}", (10, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
                if head_dropped:
                    cv2.putText(frame, "HEAD DOWN", (10, 112),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

                # detection counters
                if ear < EYE_CLOSED_RATIO_THRESHOLD:
                    closed_eye_frames += 1
                else:
                    closed_eye_frames = 0
                    closed_head_frames = 0
                    if alarm_on:
                        alarm_on = False
                        stop_alert_loop()

                # combined head+eye logic
                if ear < (EYE_CLOSED_RATIO_THRESHOLD + 0.02) and head_dropped:
                    closed_head_frames += 1
                else:
                    closed_head_frames = max(0, closed_head_frames - 1)

                # alarm triggers
                if closed_eye_frames >= MICROSLEEP_FRAMES or closed_head_frames >= HEAD_DROP_FRAMES:
                    if not alarm_on:
                        alarm_on = True
                        if os.path.exists(ALERT_FILE):
                            start_alert_loop(ALERT_FILE)

                # draw eye points
                for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                    lm = face_landmarks[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 2, (255,255,0), -1)

            else:
                # no face detected: allow short tolerance before resetting counters
                no_face_frames += 1
                if no_face_frames > NO_FACE_TOLERANCE:
                    closed_eye_frames = 0
                    closed_head_frames = 0
                    smoothed_ear = None
                    if alarm_on:
                        alarm_on = False
                        stop_alert_loop()

            # visual alarm text
            if alarm_on:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)

            cv2.imshow("Driver Drowsiness Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord('r'):
                alarm_on = False
                closed_eye_frames = 0
                closed_head_frames = 0
                smoothed_ear = None
                stop_alert_loop()
                print("Alarm reset by user.")

    finally:
        stop_alert_loop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
