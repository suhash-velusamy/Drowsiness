import os
import sys

# Suppress TensorFlow & MediaPipe logging spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'

import cv2
import numpy as np
import time
import winsound
from collections import deque

# Threshold parameters for Drowsiness Detection
EAR_THRESHOLD = 0.20        # Eye Aspect Ratio threshold (EAR < 0.20 means eyes closed)
DROWSY_TIME_LIMIT = 0.8     # Seconds of continuous closed eyes to sound alarm
SEQ_LENGTH = 15

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    """
    Calculates Eye Aspect Ratio (EAR) from 3D facial landmarks.
    Indices structure: [p1_corner, p2_top, p3_top, p4_corner, p5_bot, p6_bot]
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * img_w, lm.y * img_h]))

    p1, p2, p3, p4, p5, p6 = pts

    # Vertical eye distances
    dist_v1 = np.linalg.norm(p2 - p6)
    dist_v2 = np.linalg.norm(p3 - p5)

    # Horizontal eye distance
    dist_h = np.linalg.norm(p1 - p4)

    if dist_h == 0:
        return 0.0

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return ear

def main():
    # 1. Check & Initialize MediaPipe FaceLandmarker
    landmarker_model = "face_landmarker.task"
    if not os.path.exists(landmarker_model):
        print("Downloading precision landmark model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, landmarker_model)

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=landmarker_model)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Facial landmark indices for left and right eyes
    # Left Eye:  [corner_left, top1, top2, corner_right, bot2, bot1]
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    # Right Eye: [corner_left, top1, top2, corner_right, bot2, bot1]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n=========================================================")
    print("  ACCURATE EAR FACIAL LANDMARK DROWSINESS DETECTION      ")
    print("=========================================================")
    print(f"  EAR Threshold: < {EAR_THRESHOLD:.2f} (Closed Eyes)     ")
    print(f"  Alarm triggers after: {DROWSY_TIME_LIMIT}s closed eyes ")
    print("  Press 'q' to exit.                                     \n")

    closed_eyes_start_time = None
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        # Flip horizontally for intuitive mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert OpenCV BGR to MediaPipe RGB Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect Face Landmarks
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
            landmarks = detection_result.face_landmarks[0]

            # Calculate EAR for both eyes
            left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES, w, h)

            avg_ear = (left_ear + right_ear) / 2.0

            # Draw eye keypoints
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                px = int(landmarks[idx].x * w)
                py = int(landmarks[idx].y * h)
                cv2.circle(frame, (px, py), 2, (255, 255, 0), -1)

            # Evaluate Eye Closure
            if avg_ear < EAR_THRESHOLD:
                if closed_eyes_start_time is None:
                    closed_eyes_start_time = time.time()
                closed_duration = time.time() - closed_eyes_start_time
            else:
                closed_eyes_start_time = None
                closed_duration = 0.0

            # Trigger Alert & Alarm
            if closed_duration >= DROWSY_TIME_LIMIT:
                status_label = f"DROWSY ALERT! (Closed: {closed_duration:.1f}s | EAR: {avg_ear:.2f})"
                box_color = (0, 0, 255) # Red
                # Sound loud alarm beep
                winsound.Beep(1600, 150)
            else:
                status_label = f"DRIVER ALERT (Eyes Open | EAR: {avg_ear:.2f})"
                box_color = (0, 255, 0) # Green

            # Draw face bounding box from landmarks
            all_x = [int(lm.x * w) for lm in landmarks]
            all_y = [int(lm.y * h) for lm in landmarks]
            min_x, max_x = max(0, min(all_x)), min(w, max(all_x))
            min_y, max_y = max(0, min(all_y)), min(h, max(all_y))
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), box_color, 3)

        else:
            closed_eyes_start_time = None
            status_label = "No Driver Face Detected"
            box_color = (128, 128, 128)

        # Top Header Overlay
        cv2.rectangle(frame, (0, 0), (640, 55), (15, 15, 15), -1)
        cv2.putText(frame, status_label, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (530, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Live Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
