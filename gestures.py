import cv2
import numpy as np
import math
import time
from datetime import datetime
from pathlib import Path

from tracking import HandTracker


class GestureRecognizer:
    """
    Stage 2: Advanced Gesture Recognition & Mathematical Analysis Layer.
    """

    def __init__(self, pinch_threshold=45, grab_threshold=70, swipe_window=5):
        self.pinch_threshold = pinch_threshold
        self.grab_threshold = grab_threshold
        self.swipe_window = swipe_window
        self.history = {}
        self.initial_two_hand_dist = None
        self.base_scale = 1.0

    def analyze(self, hands_data):
        current_ids = set()
        analysis_result = {
            "hands": [],
            "two_hand_active": False,
            "two_hand_mode": None,
            "inter_hand_dist": 0.0,
            "scale_multiplier": 1.0,
            "rotation_angle": 0.0,
            "global_swipe": None,
        }

        for hand in hands_data:
            hid = hand["id"]
            current_ids.add(hid)
            lm_list = hand["lm_list"]

            thumb_tip = lm_list[4]
            index_tip = lm_list[8]
            middle_tip = lm_list[12]
            ring_tip = lm_list[16]
            pinky_tip = lm_list[20]
            palm_center = lm_list[9]

            pinch_dist = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
            is_pinching = pinch_dist < self.pinch_threshold
            pinch_center = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)

            fingertips = [index_tip, middle_tip, ring_tip, pinky_tip]
            curl_distances = [math.hypot(ft[0] - palm_center[0], ft[1] - palm_center[1]) for ft in fingertips]
            avg_curl = sum(curl_distances) / len(curl_distances)
            is_grabbing = avg_curl < self.grab_threshold

            if is_grabbing:
                gesture_type = "GRAB"
                action_center = palm_center
            elif is_pinching:
                gesture_type = "PINCH"
                action_center = pinch_center
            else:
                gesture_type = "OPEN_PALM"
                action_center = palm_center

            curr_time = time.time()
            if hid not in self.history:
                self.history[hid] = []
            self.history[hid].append((palm_center[0], palm_center[1], curr_time))

            if len(self.history[hid]) > self.swipe_window:
                self.history[hid].pop(0)

            swipe_dir = None
            if len(self.history[hid]) == self.swipe_window:
                start_x, start_y, start_t = self.history[hid][0]
                end_x, end_y, end_t = self.history[hid][-1]
                dt = end_t - start_t
                if dt > 0:
                    vx = (end_x - start_x) / dt
                    vy = (end_y - start_y) / dt
                    vel_thresh = 800.0
                    if abs(vx) > vel_thresh and abs(vx) > abs(vy):
                        swipe_dir = "SWIPE_RIGHT" if vx > 0 else "SWIPE_LEFT"
                        analysis_result["global_swipe"] = swipe_dir
                        self.history[hid].clear()
                    elif abs(vy) > vel_thresh and abs(vy) > abs(vx):
                        swipe_dir = "SWIPE_DOWN" if vy > 0 else "SWIPE_UP"
                        analysis_result["global_swipe"] = swipe_dir
                        self.history[hid].clear()

            analysis_result["hands"].append(
                {
                    "id": hid,
                    "label": hand["label"],
                    "gesture": gesture_type,
                    "action_center": action_center,
                    "pinch_dist": pinch_dist,
                    "grab_dist": avg_curl,
                    "swipe": swipe_dir,
                }
            )

        lost_ids = set(self.history.keys()) - current_ids
        for lid in lost_ids:
            del self.history[lid]

        if len(analysis_result["hands"]) == 2:
            h1 = analysis_result["hands"][0]
            h2 = analysis_result["hands"][1]
            active_gestures = ["PINCH", "GRAB"]
            if h1["gesture"] in active_gestures and h2["gesture"] in active_gestures:
                analysis_result["two_hand_active"] = True
                analysis_result["two_hand_mode"] = "SCALE_ROTATE"

                c1 = h1["action_center"]
                c2 = h2["action_center"]
                dist = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
                analysis_result["inter_hand_dist"] = dist

                if self.initial_two_hand_dist is None:
                    self.initial_two_hand_dist = dist
                else:
                    analysis_result["scale_multiplier"] = dist / max(1.0, self.initial_two_hand_dist)

                dx = c2[0] - c1[0]
                dy = c2[1] - c1[1]
                analysis_result["rotation_angle"] = math.atan2(dy, dx) * (180.0 / math.pi)
            else:
                self.initial_two_hand_dist = None
        else:
            self.initial_two_hand_dist = None

        return analysis_result


class InteractiveHologram:
    """
    Gesture-driven dashboard with shape selection and screenshot capture.
    Pinch counts as click.
    """

    def __init__(self, x=320, y=240, size=80):
        self.pos = [x, y]
        self.base_size = size
        self.current_size = size
        self.target_size = size

        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0

        self.is_grabbed = False
        self.grab_offset = [0, 0]

        self.palettes = [
            (255, 255, 0),
            (255, 150, 0),
            (0, 255, 150),
            (255, 0, 255),
        ]
        self.palette_idx = 0
        self.color = self.palettes[self.palette_idx]
        self.pulse_alpha = 0.0

        self.shape_order = ["CIRCLE", "SQUARE", "TRIANGLE", "HEXAGON"]
        self.selected_shape = "HEXAGON"
        self.screenshot_dir = Path("screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)
        self.pending_screenshot = False
        self.last_action_text = "Pinch a tile to change shape. Pinch CAPTURE to save a screenshot."
        self.last_action_until = 0.0
        self.last_screenshot_path = None
        self.capture_flash = 0.0

    def _set_status(self, message, duration=2.2):
        self.last_action_text = message
        self.last_action_until = time.time() + duration

    def _point_in_rect(self, point, rect):
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def _shape_rects(self):
        start_x = 36
        start_y = 170
        width = 164
        height = 60
        gap = 12
        rects = []
        for idx, shape in enumerate(self.shape_order):
            y = start_y + idx * (height + gap)
            rects.append((shape, (start_x, y, width, height)))
        capture_rect = (start_x, start_y + len(self.shape_order) * (height + gap) + 8, width, 66)
        return rects, capture_rect

    def _capture_screenshot(self, frame):
        file_name = datetime.now().strftime("holo_capture_%Y%m%d_%H%M%S.png")
        file_path = self.screenshot_dir / file_name
        cv2.imwrite(str(file_path), frame)
        self.last_screenshot_path = file_path
        self.capture_flash = 1.0
        self._set_status(f"Screenshot saved: {file_path.name}", duration=3.0)

    def _build_regular_polygon(self, center, radius, sides, angle_offset=0.0):
        pts = []
        for idx in range(sides):
            theta = angle_offset + (2.0 * math.pi * idx / sides)
            pts.append(
                [
                    int(center[0] + radius * math.cos(theta)),
                    int(center[1] + radius * math.sin(theta)),
                ]
            )
        return np.array(pts, np.int32)

    def _draw_glow_text(self, frame, text, origin, color, scale=0.55, thickness=1):
        x, y = origin
        cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _draw_panel_card(self, frame, rect, title, subtitle, active=False, accent=None):
        x, y, w, h = rect
        base_fill = (18, 18, 24)
        border = accent if accent is not None else self.color
        cv2.rectangle(frame, (x, y), (x + w, y + h), base_fill, cv2.FILLED)
        cv2.rectangle(frame, (x, y), (x + w, y + h), border, 2 if active else 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x + 1, y + 1), (x + w - 1, y + h - 1), (255, 255, 255), 1, cv2.LINE_AA)
        self._draw_glow_text(frame, title, (x + 14, y + 24), (245, 245, 245), scale=0.52, thickness=1)
        cv2.putText(frame, subtitle, (x + 14, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, border, 1, cv2.LINE_AA)

    def _draw_hand_overlay(self, frame, hand, panel_rect):
        x, y, w, h = panel_rect
        center = hand["action_center"]
        gesture = hand["gesture"]
        hand_id = hand["id"]
        pinch_dist = hand["pinch_dist"]
        grab_dist = hand["grab_dist"]
        label = hand["label"]

        cv2.circle(frame, center, 16, self.color, 2, cv2.LINE_AA)
        cv2.circle(frame, center, 5, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.line(frame, (center[0] - 26, center[1]), (center[0] + 26, center[1]), self.color, 1, cv2.LINE_AA)
        cv2.line(frame, (center[0], center[1] - 26), (center[0], center[1] + 26), self.color, 1, cv2.LINE_AA)

        callout_x = min(max(center[0] + 18, x + 14), x + w - 154)
        callout_y = min(max(center[1] - 38, y + 14), y + h - 68)
        cv2.rectangle(frame, (callout_x, callout_y), (callout_x + 140, callout_y + 52), (14, 14, 18), cv2.FILLED)
        cv2.rectangle(frame, (callout_x, callout_y), (callout_x + 140, callout_y + 52), self.color, 1, cv2.LINE_AA)
        self._draw_glow_text(frame, f"H{hand_id} {label[:5].upper()}", (callout_x + 10, callout_y + 18), (245, 245, 245), scale=0.42, thickness=1)
        cv2.putText(frame, f"{gesture}", (callout_x + 10, callout_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"x:{center[0]} y:{center[1]}", (callout_x + 10, callout_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)

        coords_text = f"pinch {pinch_dist:.0f}  grab {grab_dist:.0f}"
        cv2.putText(frame, coords_text, (callout_x + 10, callout_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1, cv2.LINE_AA)

    def _handle_click(self, point):
        shape_rects, capture_rect = self._shape_rects()
        for shape_name, rect in shape_rects:
            if self._point_in_rect(point, rect):
                self.selected_shape = shape_name
                self._set_status(f"Shape changed to {shape_name}", duration=2.0)
                return

        if self._point_in_rect(point, capture_rect):
            self.pending_screenshot = True
            self._set_status("Capture armed. Screenshot will be saved this frame.", duration=1.5)

    def _draw_shape_preview(self, frame, gesture_data):
        cx, cy = int(self.pos[0]), int(self.pos[1])
        s = int(self.current_size)
        glow_color = self.color
        core_color = (255, 255, 255) if self.is_grabbed else glow_color

        preview_center = (cx, cy)
        angle_radians = math.radians(self.angle_z)

        if self.selected_shape == "CIRCLE":
            cv2.circle(frame, preview_center, int(s * 1.15), core_color, 3, cv2.LINE_AA)
            cv2.circle(frame, preview_center, int(s * 0.8), glow_color, 1, cv2.LINE_AA)
        else:
            sides = {"SQUARE": 4, "TRIANGLE": 3, "HEXAGON": 6}.get(self.selected_shape, 4)
            pts = self._build_regular_polygon(preview_center, int(s * 1.1), sides, angle_offset=angle_radians)
            cv2.polylines(frame, [pts], True, core_color, 3, cv2.LINE_AA)
            cv2.polylines(frame, [pts], True, glow_color, 1, cv2.LINE_AA)

        ring_radius = int(s * 1.5)
        cv2.circle(frame, preview_center, ring_radius, glow_color, 1, cv2.LINE_AA)
        for i in range(16):
            theta = math.radians(self.angle_z + (i * 360 / 16))
            inner = ring_radius - 6
            outer = ring_radius + 5
            pt1 = (int(cx + inner * math.cos(theta)), int(cy + inner * math.sin(theta)))
            pt2 = (int(cx + outer * math.cos(theta)), int(cy + outer * math.sin(theta)))
            cv2.line(frame, pt1, pt2, glow_color, 1, cv2.LINE_AA)

        if self.capture_flash > 0:
            overlay = frame.copy()
            cv2.circle(overlay, preview_center, int(ring_radius * 1.6), glow_color, cv2.FILLED)
            cv2.addWeighted(overlay, self.capture_flash * 0.18, frame, 1.0 - (self.capture_flash * 0.18), 0, frame)

        cv2.putText(frame, f"ACTIVE SHAPE: {self.selected_shape}", (cx - 115, cy - ring_radius - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, core_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"SIZE: {s}px | ROT: {self.angle_z:.1f}deg", (cx - 112, cy + ring_radius + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, glow_color, 1, cv2.LINE_AA)

    def _draw_background(self, frame):
        h, w = frame.shape[:2]
        for y in range(h):
            mix = y / max(1, h - 1)
            base = np.array([10, 12, 18], dtype=np.float32)
            top = np.array([24, 18, 30], dtype=np.float32)
            color = (base * (1.0 - mix) + top * mix).astype(np.uint8)
            cv2.line(frame, (0, y), (w, y), tuple(int(v) for v in color), 1)

        for x in range(0, w, 48):
            cv2.line(frame, (x, 0), (x, h), (28, 28, 36), 1, cv2.LINE_AA)
        for y in range(0, h, 48):
            cv2.line(frame, (0, y), (w, y), (28, 28, 36), 1, cv2.LINE_AA)

    def _draw_ui(self, frame, gesture_data):
        h, w = frame.shape[:2]
        self._draw_background(frame)

        left_panel = (24, 24, 280, h - 48)
        right_panel = (w - 320, 24, 296, h - 48)
        center_panel = (320, 84, w - 640, h - 132)

        cv2.rectangle(frame, (left_panel[0], left_panel[1]), (left_panel[0] + left_panel[2], left_panel[1] + left_panel[3]), (16, 18, 26), cv2.FILLED)
        cv2.rectangle(frame, (right_panel[0], right_panel[1]), (right_panel[0] + right_panel[2], right_panel[1] + right_panel[3]), (16, 18, 26), cv2.FILLED)
        cv2.rectangle(frame, (center_panel[0], center_panel[1]), (center_panel[0] + center_panel[2], center_panel[1] + center_panel[3]), (10, 10, 14), cv2.FILLED)

        for panel in [left_panel, right_panel, center_panel]:
            x, y, pw, ph = panel
            cv2.rectangle(frame, (x, y), (x + pw, y + ph), self.color, 1, cv2.LINE_AA)

        self._draw_glow_text(frame, "JARVIS HOLOGRAPHIC CONTROL", (42, 60), self.color, scale=0.82, thickness=2)
        self._draw_glow_text(frame, "Pinch on a tile to select a shape. Pinch CAPTURE to save a screenshot.", (42, 92), (220, 220, 220), scale=0.45, thickness=1)

        current_mode = gesture_data["hands"][0]["gesture"] if gesture_data["hands"] else "IDLE"
        self._draw_panel_card(frame, (40, 120, 250, 70), "GESTURE INPUT", f"Mode: {current_mode}", active=True, accent=self.color)
        self._draw_panel_card(frame, (40, 202, 250, 70), "HAND PRESENCE", f"Hands: {len(gesture_data['hands'])}", accent=(255, 150, 0))
        self._draw_panel_card(frame, (40, 284, 250, 70), "SELECTED SHAPE", self.selected_shape, accent=(0, 255, 150))
        self._draw_panel_card(frame, (40, 366, 250, 70), "SCREENSHOT", "Saved to screenshots/", accent=(255, 255, 0))

        self._draw_panel_card(frame, (40, 448, 250, 112), "HAND POSITION", "See live coordinates here", accent=(255, 180, 120))
        if gesture_data["hands"]:
            y_offset = 478
            for idx, hand in enumerate(gesture_data["hands"][:2]):
                center = hand["action_center"]
                line1 = f"H{hand['id']} {hand['gesture']}"
                line2 = f"x:{center[0]} y:{center[1]}"
                line3 = f"pinch:{hand['pinch_dist']:.0f} grab:{hand['grab_dist']:.0f}"
                cv2.putText(frame, line1, (54, y_offset + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
                cv2.putText(frame, line2, (140, y_offset + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.color, 1, cv2.LINE_AA)
                cv2.putText(frame, line3, (54, y_offset + 12 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Waiting for hand...", (54, 478), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        shape_rects, capture_rect = self._shape_rects()
        self._draw_glow_text(frame, "SHAPE LIBRARY", (right_panel[0] + 26, 60), self.color, scale=0.72, thickness=2)
        self._draw_glow_text(frame, "Use pinch as click. Select one shape at a time.", (right_panel[0] + 26, 88), (220, 220, 220), scale=0.43, thickness=1)

        for shape_name, rect in shape_rects:
            active = shape_name == self.selected_shape
            accent = self.color if active else (90, 90, 110)
            self._draw_panel_card(frame, rect, shape_name, "Pinch to select", active=active, accent=accent)

            x, y, w_rect, h_rect = rect
            preview_center = (x + w_rect - 34, y + h_rect // 2)
            preview_color = self.color if active else (180, 180, 180)
            if shape_name == "CIRCLE":
                cv2.circle(frame, preview_center, 12, preview_color, 2, cv2.LINE_AA)
            else:
                sides = {"SQUARE": 4, "TRIANGLE": 3, "HEXAGON": 6}.get(shape_name, 4)
                pts = self._build_regular_polygon(preview_center, 12, sides, angle_offset=math.radians(-20))
                cv2.polylines(frame, [pts], True, preview_color, 2, cv2.LINE_AA)

        self._draw_panel_card(frame, capture_rect, "CAPTURE SCREEN", "Pinch to save PNG", active=self.pending_screenshot, accent=(255, 220, 80))
        cv2.circle(frame, (capture_rect[0] + capture_rect[2] - 28, capture_rect[1] + capture_rect[3] // 2), 12, (255, 220, 80), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (capture_rect[0] + capture_rect[2] - 34, capture_rect[1] + capture_rect[3] // 2 - 7), (capture_rect[0] + capture_rect[2] - 22, capture_rect[1] + capture_rect[3] // 2 + 5), (255, 220, 80), 2, cv2.LINE_AA)

        status_y = h - 28
        if time.time() <= self.last_action_until:
            status_text = self.last_action_text
        elif self.last_screenshot_path is not None:
            status_text = f"Last screenshot: {self.last_screenshot_path.name}"
        else:
            status_text = "Ready"
        self._draw_glow_text(frame, status_text, (42, status_y), (230, 230, 230), scale=0.52, thickness=1)

        if gesture_data["hands"]:
            for hand in gesture_data["hands"]:
                self._draw_hand_overlay(frame, hand, center_panel)

    def _update_motion(self, gesture_data):
        hands = gesture_data["hands"]

        if gesture_data["global_swipe"]:
            self.palette_idx = (self.palette_idx + 1) % len(self.palettes)
            self.color = self.palettes[self.palette_idx]
            self.pulse_alpha = 1.0
            self._set_status(f"Swipe detected: {gesture_data['global_swipe']}", duration=1.5)

        if gesture_data["two_hand_active"]:
            self.is_grabbed = True
            self.target_size = self.base_size * gesture_data["scale_multiplier"]
            self.target_size = max(40, min(300, self.target_size))
            self.current_size += (self.target_size - self.current_size) * 0.2
            self.angle_z = gesture_data["rotation_angle"]
            h1_pos = hands[0]["action_center"]
            h2_pos = hands[1]["action_center"]
            self.pos[0] = (h1_pos[0] + h2_pos[0]) // 2
            self.pos[1] = (h1_pos[1] + h2_pos[1]) // 2
        elif len(hands) == 1:
            hand = hands[0]
            if hand["gesture"] in ["PINCH", "GRAB"]:
                if not self.is_grabbed:
                    self.is_grabbed = True
                    self.grab_offset = [self.pos[0] - hand["action_center"][0], self.pos[1] - hand["action_center"][1]]
                else:
                    target_x = hand["action_center"][0] + self.grab_offset[0]
                    target_y = hand["action_center"][1] + self.grab_offset[1]
                    self.pos[0] += int((target_x - self.pos[0]) * 0.4)
                    self.pos[1] += int((target_y - self.pos[1]) * 0.4)

                if hand["gesture"] == "PINCH":
                    self._handle_click(hand["action_center"])
            else:
                self.is_grabbed = False
                self.base_size = self.current_size
        else:
            self.is_grabbed = False
            self.base_size = self.current_size

        if not self.is_grabbed:
            self.angle_x += 1.2
            self.angle_y += 0.8

        if self.pulse_alpha > 0:
            self.pulse_alpha -= 0.05

        if self.capture_flash > 0:
            self.capture_flash = max(0.0, self.capture_flash - 0.08)

    def update(self, gesture_data):
        self._update_motion(gesture_data)

    def render(self, frame, gesture_data):
        self._draw_ui(frame, gesture_data)
        self._draw_shape_preview(frame, gesture_data)

    def capture_if_requested(self, frame):
        if self.pending_screenshot:
            self.pending_screenshot = False
            self._capture_screenshot(frame)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = HandTracker(max_hands=2, detection_con=0.7, tracking_con=0.7, smooth_alpha=0.5)
    recognizer = GestureRecognizer(pinch_threshold=45, grab_threshold=70, swipe_window=5)
    hologram = InteractiveHologram(x=320, y=240, size=80)

    window_name = "Iron Man Holographic Interface - Stage 2"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("[INFO] Stage 2 Gesture Recognition & Interactive Hologram Initialized.")
    print("[INFO] Pinch a shape tile to select it. Pinch CAPTURE to save a screenshot.")
    print("[INFO] Press 'Q' or 'ESC' to exit.")

    prev_time = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        hands_data = tracker.process_frame(frame, draw=False)
        gesture_analysis = recognizer.analyze(hands_data)

        hologram.update(gesture_analysis)
        hologram.render(frame, gesture_analysis)
        hologram.capture_if_requested(frame)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "STAGE 2: GESTURE ENGINE & HOLO-MANIPULATION", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()