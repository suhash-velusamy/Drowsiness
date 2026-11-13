import os
import sys
import time
import json
import threading
from datetime import datetime
from collections import deque

# Suppress noisy TensorFlow / MediaPipe C++ logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# ==========================================
# Core Detection & Telemetry Engine
# ==========================================
class DriverMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.cap = None
        self.thread = None
        
        # Adjustable Threshold Settings
        self.ear_threshold = 0.20
        self.mar_threshold = 0.65
        self.drowsy_time_limit = 0.8  # Seconds of continuous closure for drowsy alarm
        self.yawn_time_limit = 1.5    # Seconds of continuous yawning
        self.audio_enabled = True

        # Telemetry State
        self.status = "OFFLINE"
        self.ear = 0.0
        self.mar = 0.0
        self.fps = 0.0
        self.face_detected = False
        self.is_drowsy = False
        self.is_yawning = False
        self.closed_duration = 0.0
        self.yawn_duration = 0.0
        
        # Session & Trip Analytics
        self.trip_start_time = None
        self.total_blinks = 0
        self.blink_timestamps = deque(maxlen=200)
        self.blinks_per_minute = 0
        self.drowsy_incidents_count = 0
        self.yawn_incidents_count = 0
        self.safety_score = 100.0
        self.incidents_log = []

        # Internal state helpers
        self._closed_start_time = None
        self._yawn_start_time = None
        self._last_eye_state_closed = False
        self._last_incident_logged_drowsy = False
        self._last_incident_logged_yawn = False
        self._current_frame_bytes = None

        # Facial Landmark Indices
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        # Mouth indices: [left_corner, top1, top2, right_corner, bot2, bot1]
        self.MOUTH_INDICES = [61, 81, 13, 291, 14, 178]

        # Init MediaPipe FaceLandmarker
        self.detector = None
        self.init_detector()

    def init_detector(self):
        landmarker_model = "face_landmarker.task"
        if not os.path.exists(landmarker_model):
            print("[INFO] Downloading face_landmarker.task...")
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
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.mp = mp

    def calculate_ear(self, landmarks, eye_indices, img_w, img_h):
        pts = [np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]) for idx in eye_indices]
        p1, p2, p3, p4, p5, p6 = pts
        dist_v1 = np.linalg.norm(p2 - p6)
        dist_v2 = np.linalg.norm(p3 - p5)
        dist_h = np.linalg.norm(p1 - p4)
        if dist_h == 0:
            return 0.0
        return (dist_v1 + dist_v2) / (2.0 * dist_h)

    def calculate_mar(self, landmarks, mouth_indices, img_w, img_h):
        pts = [np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]) for idx in mouth_indices]
        p1, p2, p3, p4, p5, p6 = pts  # 61, 81, 13, 291, 14, 178
        dist_v1 = np.linalg.norm(p2 - p6)
        dist_v2 = np.linalg.norm(p3 - p5)
        dist_h = np.linalg.norm(p1 - p4)
        if dist_h == 0:
            return 0.0
        return (dist_v1 + dist_v2) / (2.0 * dist_h)

    def start_camera(self):
        with self.lock:
            if self.is_running:
                return True
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("[ERROR] Could not open camera device.")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            if self.trip_start_time is None:
                self.trip_start_time = time.time()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True

    def stop_camera(self):
        with self.lock:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.status = "STANDBY"
            self._current_frame_bytes = None

    def reset_trip(self):
        with self.lock:
            self.trip_start_time = time.time() if self.is_running else None
            self.total_blinks = 0
            self.blink_timestamps.clear()
            self.blinks_per_minute = 0
            self.drowsy_incidents_count = 0
            self.yawn_incidents_count = 0
            self.safety_score = 100.0
            self.incidents_log.clear()
            self._closed_start_time = None
            self._yawn_start_time = None
            self._last_incident_logged_drowsy = False
            self._last_incident_logged_yawn = False

    def update_settings(self, data):
        with self.lock:
            if 'ear_threshold' in data:
                self.ear_threshold = float(data['ear_threshold'])
            if 'mar_threshold' in data:
                self.mar_threshold = float(data['mar_threshold'])
            if 'drowsy_time_limit' in data:
                self.drowsy_time_limit = float(data['drowsy_time_limit'])
            if 'audio_enabled' in data:
                self.audio_enabled = bool(data['audio_enabled'])

    def log_incident(self, event_type, severity, details):
        incident = {
            "id": len(self.incidents_log) + 1,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": event_type,
            "severity": severity,
            "details": details,
            "ear": round(self.ear, 3),
            "mar": round(self.mar, 3)
        }
        self.incidents_log.insert(0, incident) # prepend newest first
        # Reduce safety score according to severity
        if severity == "HIGH":
            self.safety_score = max(0.0, self.safety_score - 5.0)
        elif severity == "MEDIUM":
            self.safety_score = max(0.0, self.safety_score - 2.5)

    def _capture_loop(self):
        prev_time = time.time()
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            curr_time = time.time()
            self.fps = 1.0 / (curr_time - prev_time + 1e-5)
            prev_time = curr_time

            # Flip horizontally for natural mirror display
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            face_present = bool(detection_result.face_landmarks and len(detection_result.face_landmarks) > 0)
            self.face_detected = face_present

            # Calculate Blinks Per Minute (window = last 60s)
            now = time.time()
            while self.blink_timestamps and (now - self.blink_timestamps[0] > 60.0):
                self.blink_timestamps.popleft()
            self.blinks_per_minute = len(self.blink_timestamps)

            if face_present:
                landmarks = detection_result.face_landmarks[0]
                left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_INDICES, w, h)
                right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_INDICES, w, h)
                self.ear = (left_ear + right_ear) / 2.0
                self.mar = self.calculate_mar(landmarks, self.MOUTH_INDICES, w, h)

                # Draw eye landmarks mesh
                for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
                    px = int(landmarks[idx].x * w)
                    py = int(landmarks[idx].y * h)
                    cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

                # Draw mouth landmarks mesh
                for idx in self.MOUTH_INDICES:
                    px = int(landmarks[idx].x * w)
                    py = int(landmarks[idx].y * h)
                    cv2.circle(frame, (px, py), 2, (255, 128, 0), -1)

                # Blink detection logic
                is_eye_closed = self.ear < self.ear_threshold
                if is_eye_closed and not self._last_eye_state_closed:
                    pass # started closing
                elif not is_eye_closed and self._last_eye_state_closed:
                    # Eye just reopened -> count as blink if closed time was short (< 0.5s)
                    if self._closed_start_time and (now - self._closed_start_time < 0.6):
                        self.total_blinks += 1
                        self.blink_timestamps.append(now)
                self._last_eye_state_closed = is_eye_closed

                # Eye Closure / Drowsiness Duration Tracking
                if is_eye_closed:
                    if self._closed_start_time is None:
                        self._closed_start_time = now
                    self.closed_duration = now - self._closed_start_time
                else:
                    self._closed_start_time = None
                    self.closed_duration = 0.0

                # Yawning Duration Tracking
                is_yawn = self.mar > self.mar_threshold
                if is_yawn:
                    if self._yawn_start_time is None:
                        self._yawn_start_time = now
                    self.yawn_duration = now - self._yawn_start_time
                else:
                    self._yawn_start_time = None
                    self.yawn_duration = 0.0

                # Drowsy Evaluation
                if self.closed_duration >= self.drowsy_time_limit:
                    self.is_drowsy = True
                    self.status = "DROWSY WARNING"
                    box_color = (0, 0, 255) # Red
                    if not self._last_incident_logged_drowsy:
                        self.drowsy_incidents_count += 1
                        self.log_incident("Drowsiness Alert", "HIGH", f"Eyes closed for {self.closed_duration:.1f}s")
                        self._last_incident_logged_drowsy = True
                else:
                    self.is_drowsy = False
                    self._last_incident_logged_drowsy = False

                # Yawn Evaluation
                if self.yawn_duration >= self.yawn_time_limit:
                    self.is_yawning = True
                    if not self.is_drowsy:
                        self.status = "YAWN DETECTED"
                        box_color = (0, 165, 255) # Orange
                    if not self._last_incident_logged_yawn:
                        self.yawn_incidents_count += 1
                        self.log_incident("Fatigue / Yawning", "MEDIUM", f"Mouth open for {self.yawn_duration:.1f}s")
                        self._last_incident_logged_yawn = True
                else:
                    self.is_yawning = False
                    self._last_incident_logged_yawn = False

                if not self.is_drowsy and not self.is_yawning:
                    self.status = "ACTIVE & ALERT"
                    box_color = (0, 255, 0) # Green

                # Bounding Box around Face
                all_x = [int(lm.x * w) for lm in landmarks]
                all_y = [int(lm.y * h) for lm in landmarks]
                min_x, max_x = max(0, min(all_x)), min(w, max(all_x))
                min_y, max_y = max(0, min(all_y)), min(h, max(all_y))
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), box_color, 2)

                # Modern HUD Pill over bounding box
                cv2.putText(frame, f"EAR: {self.ear:.2f} | MAR: {self.mar:.2f}", 
                            (min_x, max(20, min_y - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
            else:
                self.ear = 0.0
                self.mar = 0.0
                self.closed_duration = 0.0
                self.yawn_duration = 0.0
                self.is_drowsy = False
                self.is_yawning = False
                self.status = "NO FACE DETECTED"
                self._closed_start_time = None
                self._yawn_start_time = None

            # Encode as JPEG
            ret_enc, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret_enc:
                self._current_frame_bytes = buffer.tobytes()

            time.sleep(0.015)

    def get_frame(self):
        return self._current_frame_bytes

    def get_telemetry(self):
        trip_sec = int(time.time() - self.trip_start_time) if (self.trip_start_time and self.is_running) else 0
        minutes, seconds = divmod(trip_sec, 60)
        hours, minutes = divmod(minutes, 60)
        trip_duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return {
            "status": self.status,
            "is_running": self.is_running,
            "face_detected": self.face_detected,
            "ear": round(float(self.ear), 3),
            "mar": round(float(self.mar), 3),
            "is_drowsy": self.is_drowsy,
            "is_yawning": self.is_yawning,
            "closed_duration": round(float(self.closed_duration), 2),
            "yawn_duration": round(float(self.yawn_duration), 2),
            "total_blinks": self.total_blinks,
            "blinks_per_minute": self.blinks_per_minute,
            "drowsy_incidents_count": self.drowsy_incidents_count,
            "yawn_incidents_count": self.yawn_incidents_count,
            "safety_score": round(self.safety_score, 1),
            "fps": round(float(self.fps), 1),
            "trip_duration": trip_duration_str,
            "ear_threshold": self.ear_threshold,
            "mar_threshold": self.mar_threshold,
            "drowsy_time_limit": self.drowsy_time_limit,
            "audio_enabled": self.audio_enabled
        }


# Global Monitor Instance
monitor = DriverMonitor()

# ==========================================
# Routes & API Endpoints
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        frame_bytes = monitor.get_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def api_telemetry():
    return jsonify(monitor.get_telemetry())

@app.route('/api/camera/toggle', methods=['POST'])
def api_camera_toggle():
    action = request.json.get('action', 'toggle')
    if action == 'start' or (action == 'toggle' and not monitor.is_running):
        success = monitor.start_camera()
        return jsonify({"success": success, "is_running": monitor.is_running})
    else:
        monitor.stop_camera()
        return jsonify({"success": True, "is_running": False})

@app.route('/api/settings', methods=['POST'])
def api_settings():
    data = request.json or {}
    monitor.update_settings(data)
    return jsonify({"success": True, "settings": {
        "ear_threshold": monitor.ear_threshold,
        "mar_threshold": monitor.mar_threshold,
        "drowsy_time_limit": monitor.drowsy_time_limit,
        "audio_enabled": monitor.audio_enabled
    }})

@app.route('/api/trip/reset', methods=['POST'])
def api_trip_reset():
    monitor.reset_trip()
    return jsonify({"success": True})

@app.route('/api/incidents')
def api_incidents():
    return jsonify({"incidents": monitor.incidents_log})

@app.route('/api/trip/export')
def api_trip_export():
    import io
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Incident ID', 'Timestamp', 'Event Type', 'Severity', 'Details', 'EAR', 'MAR'])
    for item in monitor.incidents_log:
        writer.writerow([item['id'], item['timestamp'], item['type'], item['severity'], item['details'], item['ear'], item['mar']])
    
    response = Response(output.getvalue(), mimetype='text/csv')
    response.headers['Content-Disposition'] = f'attachment; filename=driver_safety_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    return response

@app.route('/api/process_frame', methods=['POST'])
def api_process_frame():
    import base64
    data = request.json or {}
    image_b64 = data.get('image', '')
    if not image_b64:
        return jsonify({"error": "No image data provided"}), 400

    try:
        if ',' in image_b64:
            image_b64 = image_b64.split(',', 1)[1]
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid frame decode"}), 400

        # Process frame
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = monitor.mp.Image(image_format=monitor.mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = monitor.detector.detect(mp_image)

        face_present = bool(detection_result.face_landmarks and len(detection_result.face_landmarks) > 0)
        monitor.face_detected = face_present

        now = time.time()
        if face_present:
            landmarks = detection_result.face_landmarks[0]
            left_ear = monitor.calculate_ear(landmarks, monitor.LEFT_EYE_INDICES, w, h)
            right_ear = monitor.calculate_ear(landmarks, monitor.RIGHT_EYE_INDICES, w, h)
            monitor.ear = (left_ear + right_ear) / 2.0
            monitor.mar = monitor.calculate_mar(landmarks, monitor.MOUTH_INDICES, w, h)

            # Blink detection
            is_eye_closed = monitor.ear < monitor.ear_threshold
            if not is_eye_closed and monitor._last_eye_state_closed:
                if monitor._closed_start_time and (now - monitor._closed_start_time < 0.6):
                    monitor.total_blinks += 1
                    monitor.blink_timestamps.append(now)
            monitor._last_eye_state_closed = is_eye_closed

            if is_eye_closed:
                if monitor._closed_start_time is None:
                    monitor._closed_start_time = now
                monitor.closed_duration = now - monitor._closed_start_time
            else:
                monitor._closed_start_time = None
                monitor.closed_duration = 0.0

            # Yawning
            is_yawn = monitor.mar > monitor.mar_threshold
            if is_yawn:
                if monitor._yawn_start_time is None:
                    monitor._yawn_start_time = now
                monitor.yawn_duration = now - monitor._yawn_start_time
            else:
                monitor._yawn_start_time = None
                monitor.yawn_duration = 0.0

            if monitor.closed_duration >= monitor.drowsy_time_limit:
                monitor.is_drowsy = True
                monitor.status = "DROWSY WARNING"
                if not monitor._last_incident_logged_drowsy:
                    monitor.drowsy_incidents_count += 1
                    monitor.log_incident("Drowsiness Alert", "HIGH", f"Eyes closed for {monitor.closed_duration:.1f}s")
                    monitor._last_incident_logged_drowsy = True
            else:
                monitor.is_drowsy = False
                monitor._last_incident_logged_drowsy = False

            if monitor.yawn_duration >= monitor.yawn_time_limit:
                monitor.is_yawning = True
                if not monitor.is_drowsy:
                    monitor.status = "YAWN DETECTED"
                if not monitor._last_incident_logged_yawn:
                    monitor.yawn_incidents_count += 1
                    monitor.log_incident("Fatigue / Yawning", "MEDIUM", f"Mouth open for {monitor.yawn_duration:.1f}s")
                    monitor._last_incident_logged_yawn = True
            else:
                monitor.is_yawning = False
                monitor._last_incident_logged_yawn = False

            if not monitor.is_drowsy and not monitor.is_yawning:
                monitor.status = "ACTIVE & ALERT"
        else:
            monitor.ear = 0.0
            monitor.mar = 0.0
            monitor.closed_duration = 0.0
            monitor.yawn_duration = 0.0
            monitor.is_drowsy = False
            monitor.is_yawning = False
            monitor.status = "NO FACE DETECTED"

        return jsonify(monitor.get_telemetry())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n========================================================")
    print("  🚀 DRIVERGUARD AI - SMART DRIVER SAFETY WEB APP        ")
    print("========================================================")
    print("  🌐 Web Dashboard: http://127.0.0.1:5000               ")
    print("========================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
