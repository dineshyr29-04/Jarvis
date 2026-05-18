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

    def _finger_open(self, tip, pip):
        return tip[1] < pip[1] - 8

    def _thumb_open(self, thumb_tip, thumb_ip, index_mcp):
        return thumb_tip[1] < thumb_ip[1] - 6 or abs(thumb_tip[0] - index_mcp[0]) > 20

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
            "system_command": None,
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
            index_mcp = lm_list[5]
            middle_pip = lm_list[10]
            ring_pip = lm_list[14]
            pinky_pip = lm_list[18]
            thumb_ip = lm_list[3]

            index_open = self._finger_open(index_tip, lm_list[6])
            middle_open = self._finger_open(middle_tip, middle_pip)
            ring_open = self._finger_open(ring_tip, ring_pip)
            pinky_open = self._finger_open(pinky_tip, pinky_pip)
            thumb_is_open = self._thumb_open(thumb_tip, thumb_ip, index_mcp)
            open_count = sum([thumb_is_open, index_open, middle_open, ring_open, pinky_open])

            pinch_dist = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
            is_pinching = pinch_dist < self.pinch_threshold
            pinch_center = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)

            fingertips = [index_tip, middle_tip, ring_tip, pinky_tip]
            curl_distances = [math.hypot(ft[0] - palm_center[0], ft[1] - palm_center[1]) for ft in fingertips]
            avg_curl = sum(curl_distances) / len(curl_distances)
            is_grabbing = avg_curl < self.grab_threshold
            is_fist = open_count <= 1 and avg_curl < self.grab_threshold * 0.8
            is_peace = index_open and middle_open and not ring_open and not pinky_open
            is_point = index_open and not middle_open and not ring_open and not pinky_open
            is_thumbs_up = thumb_is_open and not index_open and not middle_open and not ring_open and not pinky_open

            if is_fist:
                gesture_type = "FIST"
                action_center = palm_center
                analysis_result["system_command"] = "PAUSE_GESTURES"
            elif is_thumbs_up:
                gesture_type = "THUMBS_UP"
                action_center = palm_center
                analysis_result["system_command"] = "TAKE_SCREENSHOT"
            elif is_peace:
                gesture_type = "PEACE"
                action_center = palm_center
                analysis_result["system_command"] = "CYCLE_THEME"
            elif is_point:
                gesture_type = "POINT"
                action_center = index_tip
                analysis_result["system_command"] = "FOCUS_POINTER"
            elif is_grabbing:
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
                    "index_tip": index_tip,
                    "pinch_dist": pinch_dist,
                    "grab_dist": avg_curl,
                    "open_count": open_count,
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
        self.gestures_enabled = True
        self.pointer_focus = None
        self.hand_trails = {}

    def _set_status(self, message, duration=2.2):
        self.last_action_text = message
        self.last_action_until = time.time() + duration

    def _layout(self, frame):
        h, w = frame.shape[:2]
        margin = 22
        header_h = 72
        body_top = margin + header_h + 12
        body_h = h - body_top - margin
        left_w = 328
        right_w = 332
        center_w = max(320, w - (margin * 2) - left_w - right_w - 24)

        header = (margin, margin, w - margin * 2, header_h)
        left = (margin, body_top, left_w, body_h)
        center = (margin + left_w + 12, body_top, center_w, body_h)
        right = (w - margin - right_w, body_top, right_w, body_h)
        return header, left, center, right

    def _control_rects(self, left_panel):
        x, y, w, h = left_panel
        top = y + 126
        card_h = 56
        gap = 10
        rects = {
            "pause": (x + 18, top, w - 36, card_h),
            "resume": (x + 18, top + card_h + gap, w - 36, card_h),
            "capture": (x + 18, top + (card_h + gap) * 2, w - 36, card_h),
        }
        return rects

    def _point_in_rect(self, point, rect):
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def _shape_rects(self, right_panel):
        x, y, w, h = right_panel
        start_y = y + 126
        width = w - 36
        height = 54
        gap = 10
        rects = []
        for idx, shape in enumerate(self.shape_order):
            tile_y = start_y + idx * (height + gap)
            rects.append((shape, (x + 18, tile_y, width, height)))
        capture_rect = (x + 18, start_y + len(self.shape_order) * (height + gap) + 6, width, 58)
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

    def _draw_status_bar(self, frame, header_rect):
        x, y, w, h = header_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (14, 16, 22), cv2.FILLED)
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, 1, cv2.LINE_AA)
        self._draw_glow_text(frame, "JARVIS HOLOGRAPHIC CONTROL", (x + 18, y + 30), self.color, scale=0.92, thickness=2)
        cv2.putText(frame, "Pinch = click | Fist = stop | Open palm = continue | Peace = theme | Thumbs-up = screenshot", (x + 18, y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_trail(self, frame, hand_id, point, color):
        if hand_id not in self.hand_trails:
            self.hand_trails[hand_id] = []
        trail = self.hand_trails[hand_id]
        trail.append(point)
        if len(trail) > 14:
            trail.pop(0)
        for idx in range(1, len(trail)):
            alpha = idx / max(1, len(trail) - 1)
            thickness = max(1, int(alpha * 4))
            cv2.line(frame, trail[idx - 1], trail[idx], color, thickness, cv2.LINE_AA)

    def _draw_gesture_chip(self, frame, rect, gesture_name, caption, active=False, accent=None):
        x, y, w, h = rect
        fill = (22, 24, 32)
        border = accent if accent is not None else self.color
        cv2.rectangle(frame, (x, y), (x + w, y + h), fill, cv2.FILLED)
        cv2.rectangle(frame, (x, y), (x + w, y + h), border, 2 if active else 1, cv2.LINE_AA)
        self._draw_glow_text(frame, gesture_name, (x + 12, y + 22), (245, 245, 245), scale=0.50, thickness=1)
        cv2.putText(frame, caption, (x + 12, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, border, 1, cv2.LINE_AA)

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

    def _handle_click(self, point, left_panel=None, right_panel=None):
        if left_panel is None or right_panel is None:
            return False

        controls = self._control_rects(left_panel)
        shape_rects, capture_rect = self._shape_rects(right_panel)

        if self._point_in_rect(point, controls["pause"]):
            self.gestures_enabled = False
            self._set_status("Gesture engine paused.", duration=2.0)
            return True

        if self._point_in_rect(point, controls["resume"]):
            self.gestures_enabled = True
            self._set_status("Gesture engine resumed.", duration=2.0)
            return True

        if self._point_in_rect(point, controls["capture"]):
            self.pending_screenshot = True
            self._set_status("Screenshot armed.", duration=1.4)
            return True

        for shape_name, rect in shape_rects:
            if self._point_in_rect(point, rect):
                self.selected_shape = shape_name
                self._set_status(f"Shape changed to {shape_name}", duration=2.0)
                return True

        if self._point_in_rect(point, capture_rect):
            self.pending_screenshot = True
            self._set_status("Capture armed. Screenshot will be saved this frame.", duration=1.5)
            return True

        return False

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

        header, left_panel, center_panel, right_panel = self._layout(frame)
        header_x, header_y, header_w, header_h = header

        # Core containers.
        for panel in [left_panel, center_panel, right_panel]:
            x, y, pw, ph = panel
            cv2.rectangle(frame, (x, y), (x + pw, y + ph), (14, 16, 22), cv2.FILLED)
            cv2.rectangle(frame, (x, y), (x + pw, y + ph), self.color, 1, cv2.LINE_AA)

        self._draw_status_bar(frame, header)

        # Left column: system controls and live status.
        self._draw_panel_card(frame, (left_panel[0] + 14, left_panel[1] + 14, left_panel[2] - 28, 70), "SYSTEM STATE", "Gestures enabled" if self.gestures_enabled else "Gestures paused", active=True, accent=(0, 255, 150) if self.gestures_enabled else (255, 140, 120))

        controls = self._control_rects(left_panel)
        self._draw_gesture_chip(frame, controls["pause"], "STOP GESTURES", "Fist or pinch tile", active=not self.gestures_enabled, accent=(255, 120, 120))
        self._draw_gesture_chip(frame, controls["resume"], "CONTINUE", "Open palm or tile", active=self.gestures_enabled, accent=(0, 255, 150))
        self._draw_gesture_chip(frame, controls["capture"], "SCREENSHOT", "Thumbs-up or tile", active=self.pending_screenshot, accent=(255, 220, 80))

        self._draw_panel_card(frame, (left_panel[0] + 14, left_panel[1] + 322, left_panel[2] - 28, 74), "HAND POSITION", "Live coordinates", accent=(255, 180, 120))
        hand_status_y = left_panel[1] + 352
        if gesture_data["hands"]:
            for idx, hand in enumerate(gesture_data["hands"][:2]):
                center = hand["action_center"]
                cv2.putText(frame, f"H{hand['id']} {hand['gesture']}", (left_panel[0] + 28, hand_status_y + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
                cv2.putText(frame, f"x:{center[0]} y:{center[1]}", (left_panel[0] + 132, hand_status_y + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.color, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Waiting for hands...", (left_panel[0] + 28, hand_status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        self._draw_panel_card(frame, (left_panel[0] + 14, left_panel[1] + 412, left_panel[2] - 28, 118), "ACTIVE GESTURES", "Fist pauses | open palm continues | peace cycles theme", accent=(170, 170, 255))
        active_gesture_text = gesture_data["hands"][0]["gesture"] if gesture_data["hands"] else "IDLE"
        cv2.putText(frame, f"Current: {active_gesture_text}", (left_panel[0] + 28, left_panel[1] + 456), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(frame, "Point = follow index finger", (left_panel[0] + 28, left_panel[1] + 478), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "Thumbs-up = screenshot", (left_panel[0] + 28, left_panel[1] + 500), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        # Center column: object and live hand overlays.
        cv2.rectangle(frame, (center_panel[0] + 18, center_panel[1] + 18), (center_panel[0] + center_panel[2] - 18, center_panel[1] + center_panel[3] - 18), (10, 10, 14), cv2.FILLED)
        cv2.rectangle(frame, (center_panel[0] + 18, center_panel[1] + 18), (center_panel[0] + center_panel[2] - 18, center_panel[1] + center_panel[3] - 18), self.color, 1, cv2.LINE_AA)
        self._draw_glow_text(frame, "LIVE HAND CANVAS", (center_panel[0] + 34, center_panel[1] + 50), self.color, scale=0.74, thickness=2)
        self._draw_glow_text(frame, "Watch the markers and callouts move with your hands.", (center_panel[0] + 34, center_panel[1] + 78), (220, 220, 220), scale=0.44, thickness=1)

        if gesture_data["hands"]:
            for hand in gesture_data["hands"]:
                self._draw_trail(frame, hand["id"], hand["action_center"], self.color)
                self._draw_hand_overlay(frame, hand, center_panel)

        # Right column: shape library.
        shape_rects, capture_rect = self._shape_rects(right_panel)
        self._draw_panel_card(frame, (right_panel[0] + 14, right_panel[1] + 14, right_panel[2] - 28, 70), "SHAPE LIBRARY", self.selected_shape, active=True, accent=(255, 220, 80))
        self._draw_glow_text(frame, "Pinch a shape card to select it.", (right_panel[0] + 24, right_panel[1] + 108), (220, 220, 220), scale=0.42, thickness=1)

        for shape_name, rect in shape_rects:
            active = shape_name == self.selected_shape
            accent = self.color if active else (90, 90, 110)
            self._draw_gesture_chip(frame, rect, shape_name, "Pinch to select", active=active, accent=accent)

            x, y, w_rect, h_rect = rect
            preview_center = (x + w_rect - 30, y + h_rect // 2)
            preview_color = self.color if active else (180, 180, 180)
            if shape_name == "CIRCLE":
                cv2.circle(frame, preview_center, 12, preview_color, 2, cv2.LINE_AA)
            else:
                sides = {"SQUARE": 4, "TRIANGLE": 3, "HEXAGON": 6}.get(shape_name, 4)
                pts = self._build_regular_polygon(preview_center, 12, sides, angle_offset=math.radians(-20))
                cv2.polylines(frame, [pts], True, preview_color, 2, cv2.LINE_AA)

        self._draw_gesture_chip(frame, capture_rect, "CAPTURE SCREEN", "Pinch to save PNG", active=self.pending_screenshot, accent=(255, 220, 80))

        # Footer status.
        if time.time() <= self.last_action_until:
            status_text = self.last_action_text
        elif self.last_screenshot_path is not None:
            status_text = f"Last screenshot: {self.last_screenshot_path.name}"
        else:
            status_text = "Ready"
        self._draw_glow_text(frame, status_text, (header_x + header_w - 420, header_y + header_h - 14), (230, 230, 230), scale=0.50, thickness=1)

    def _update_motion(self, gesture_data):
        hands = gesture_data["hands"]

        if gesture_data["system_command"] == "PAUSE_GESTURES":
            self.gestures_enabled = False
            self._set_status("Gesture engine paused. Show an open palm to continue.", duration=2.0)
        elif gesture_data["system_command"] == "TAKE_SCREENSHOT" and self.gestures_enabled:
            self.pending_screenshot = True
            self._set_status("Screenshot armed from thumbs-up.", duration=1.6)
        elif gesture_data["system_command"] == "CYCLE_THEME" and self.gestures_enabled:
            self.palette_idx = (self.palette_idx + 1) % len(self.palettes)
            self.color = self.palettes[self.palette_idx]
            self.pulse_alpha = 1.0
            self._set_status("Theme cycled.", duration=1.4)

        if not self.gestures_enabled:
            if any(hand["gesture"] == "OPEN_PALM" for hand in hands):
                self.gestures_enabled = True
                self._set_status("Gesture engine resumed.", duration=2.0)

            if self.pulse_alpha > 0:
                self.pulse_alpha = max(0.0, self.pulse_alpha - 0.04)
            if self.capture_flash > 0:
                self.capture_flash = max(0.0, self.capture_flash - 0.06)
            return

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
            if hand["gesture"] in ["PINCH", "GRAB", "POINT"]:
                if not self.is_grabbed:
                    self.is_grabbed = True
                    self.grab_offset = [self.pos[0] - hand["action_center"][0], self.pos[1] - hand["action_center"][1]]
                else:
                    target_x = hand["action_center"][0] + self.grab_offset[0]
                    target_y = hand["action_center"][1] + self.grab_offset[1]
                    self.pos[0] += int((target_x - self.pos[0]) * 0.4)
                    self.pos[1] += int((target_y - self.pos[1]) * 0.4)

                if hand["gesture"] == "PINCH":
                    header, left_panel, center_panel, right_panel = self._layout(frame=np.zeros((720, 1280, 3), dtype=np.uint8))
                    self._handle_click(hand["action_center"], left_panel, right_panel)
                elif hand["gesture"] == "POINT":
                    self.pointer_focus = hand.get("index_tip", hand["action_center"])
                    self.pos[0] += int((self.pointer_focus[0] - self.pos[0]) * 0.28)
                    self.pos[1] += int((self.pointer_focus[1] - self.pos[1]) * 0.28)
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
        if key == ord("p"):
            hologram.gestures_enabled = False
            hologram._set_status("Gesture engine paused from keyboard.", duration=2.0)
        if key == ord("c"):
            hologram.gestures_enabled = True
            hologram._set_status("Gesture engine resumed from keyboard.", duration=2.0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()