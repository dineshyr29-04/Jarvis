import io
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

from gestures import GestureRecognizer
from tracking import HandTracker


APP_TITLE = "Jarvis Holo Control"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


def make_placeholder_frame(message, subtitle="Waiting for camera..."):
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:] = (12, 14, 20)
    cv2.putText(frame, message, (48, 128), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (48, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


class HoloBackend:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.gestures_enabled = True
        self.theme_index = 0
        self.theme_names = ["Cyan", "Amber", "Mint", "Magenta"]
        self.theme_colors = [
            (255, 255, 0),
            (255, 170, 0),
            (0, 255, 170),
            (255, 0, 255),
        ]
        self.selected_shape = "HEXAGON"
        self.last_status = "System online"
        self.last_screenshot_path = None
        self.latest_frame = make_placeholder_frame("Jarvis Holographic Interface")
        self.latest_jpeg = None
        self.latest_state = {
            "fps": 0,
            "hand_count": 0,
            "gesture": "IDLE",
            "enabled": True,
            "selected_shape": self.selected_shape,
            "theme": self.theme_names[self.theme_index],
            "last_status": self.last_status,
            "hands": [],
        }

        self.screenshot_dir = Path("screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)

        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.tracker = HandTracker(max_hands=2, detection_con=0.7, tracking_con=0.7, smooth_alpha=0.5)
        self.recognizer = GestureRecognizer(pinch_threshold=45, grab_threshold=70, swipe_window=5)

        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def set_status(self, message):
        with self.lock:
            self.last_status = message

    def set_shape(self, shape):
        with self.lock:
            self.selected_shape = shape
            self.last_status = f"Shape set to {shape}"

    def set_enabled(self, enabled):
        with self.lock:
            self.gestures_enabled = enabled
            self.last_status = "Gesture engine resumed" if enabled else "Gesture engine paused"

    def cycle_theme(self):
        with self.lock:
            self.theme_index = (self.theme_index + 1) % len(self.theme_names)
            self.last_status = f"Theme switched to {self.theme_names[self.theme_index]}"

    def _draw_overlay(self, frame, hands_data, gesture_data):
        accent = self.theme_colors[self.theme_index]
        cv2.rectangle(frame, (24, 24), (frame.shape[1] - 24, 84), (14, 16, 22), cv2.FILLED)
        cv2.rectangle(frame, (24, 24), (frame.shape[1] - 24, 84), accent, 1, cv2.LINE_AA)
        cv2.putText(frame, APP_TITLE, (44, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, accent, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{gesture_data['hands'][0]['gesture'] if gesture_data['hands'] else 'IDLE'} | {'ON' if self.gestures_enabled else 'PAUSED'} | Shape: {self.selected_shape}",
            (44, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )

        for hand in hands_data:
            center = hand["action_center"]
            gesture = hand["gesture"]
            hand_id = hand["id"]
            cv2.circle(frame, center, 14, accent, 2, cv2.LINE_AA)
            cv2.circle(frame, center, 4, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
            cv2.line(frame, (center[0] - 22, center[1]), (center[0] + 22, center[1]), accent, 1, cv2.LINE_AA)
            cv2.line(frame, (center[0], center[1] - 22), (center[0], center[1] + 22), accent, 1, cv2.LINE_AA)
            cv2.putText(frame, f"H{hand_id} {gesture} ({center[0]}, {center[1]})", (center[0] + 18, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)

    def _apply_commands(self, gesture_data):
        system_command = gesture_data.get("system_command")
        if system_command == "PAUSE_GESTURES":
            self.set_enabled(False)
        elif system_command == "TAKE_SCREENSHOT" and self.gestures_enabled:
            self.capture_screenshot()
        elif system_command == "CYCLE_THEME" and self.gestures_enabled:
            self.cycle_theme()

        if not self.gestures_enabled and any(hand["gesture"] == "OPEN_PALM" for hand in gesture_data["hands"]):
            self.set_enabled(True)

    def _loop(self):
        last_tick = time.time()
        while self.running:
            success, frame = self.camera.read()
            if not success:
                frame = make_placeholder_frame("Jarvis Holographic Interface")
                with self.lock:
                    self.latest_frame = frame.copy()
                    ok, jpeg = cv2.imencode(".jpg", frame)
                    self.latest_jpeg = jpeg.tobytes() if ok else None
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            hands_data = self.tracker.process_frame(frame, draw=False)
            gesture_data = self.recognizer.analyze(hands_data)

            self._apply_commands(gesture_data)
            accent = self.theme_colors[self.theme_index]
            self._draw_overlay(frame, hands_data, gesture_data)

            fps = 1.0 / max(1e-6, time.time() - last_tick)
            last_tick = time.time()

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_state = {
                    "fps": round(fps, 1),
                    "hand_count": len(hands_data),
                    "gesture": gesture_data["hands"][0]["gesture"] if gesture_data["hands"] else "IDLE",
                    "enabled": self.gestures_enabled,
                    "selected_shape": self.selected_shape,
                    "theme": self.theme_names[self.theme_index],
                    "last_status": self.last_status,
                    "hands": [
                        {
                            "id": hand["id"],
                            "gesture": hand["gesture"],
                            "x": hand["action_center"][0],
                            "y": hand["action_center"][1],
                            "pinch": round(hand["pinch_dist"], 1),
                            "grab": round(hand["grab_dist"], 1),
                        }
                        for hand in gesture_data["hands"]
                    ],
                    "theme_color": accent,
                }
                ok, jpeg = cv2.imencode(".jpg", frame)
                self.latest_jpeg = jpeg.tobytes() if ok else None

            time.sleep(0.01)

    def capture_screenshot(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            return None

        filename = f"holo_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = self.screenshot_dir / filename
        cv2.imwrite(str(file_path), frame)
        with self.lock:
            self.last_screenshot_path = str(file_path)
            self.last_status = f"Screenshot saved: {filename}"
        ok, encoded = cv2.imencode(".png", frame)
        return encoded.tobytes() if ok else None

    def get_state(self):
        with self.lock:
            payload = dict(self.latest_state)
            payload["last_screenshot_path"] = self.last_screenshot_path
            return payload

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg


backend = HoloBackend()
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", app_title=APP_TITLE)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = backend.get_jpeg()
            if frame is None:
                time.sleep(0.03)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    return jsonify(backend.get_state())


@app.route("/api/control", methods=["POST"])
def api_control():
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action")
    value = data.get("value")

    if action == "pause":
        backend.set_enabled(False)
    elif action == "resume":
        backend.set_enabled(True)
    elif action == "toggle":
        backend.set_enabled(not backend.get_state()["enabled"])
    elif action == "theme":
        backend.cycle_theme()
    elif action == "shape" and value:
        backend.set_shape(str(value).upper())
    elif action == "capture":
        backend.capture_screenshot()
    else:
        return jsonify({"ok": False, "error": "Unknown action"}), 400

    return jsonify({"ok": True, "state": backend.get_state()})


@app.route("/api/screenshot")
def api_screenshot():
    png_bytes = backend.capture_screenshot()
    if png_bytes is None:
        return jsonify({"ok": False, "error": "No frame available"}), 503
    return send_file(io.BytesIO(png_bytes), mimetype="image/png", as_attachment=False, download_name="holo-screenshot.png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)