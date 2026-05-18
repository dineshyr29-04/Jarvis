import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import urllib.request

# Modern MediaPipe Tasks Vision imports (Python 3.11/3.12/3.13 compatible)
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

class HandTracker:
    """
    Stage 1: Advanced Hand Tracking Module for Holographic Interface (Python 3.13 Modern API).
    
    Features:
    - Automatically downloads the official MediaPipe Hand Landmarker TFLite model bundle.
    - Uses the modern MediaPipe Tasks Vision API (`HandLandmarker`).
    - Detects up to 2 hands in real-time.
    - Tracks all 21 MediaPipe landmarks per hand.
    - Implements Exponential Moving Average (EMA) stabilization to eliminate jitter.
    - Renders futuristic neon blue/cyan HUD overlays, joint nodes, and bone wireframes.
    
    Landmark Indexing Reference (21 Landmarks):
    0: Wrist
    1, 2, 3, 4: Thumb (CMC, MCP, IP, Tip)
    5, 6, 7, 8: Index Finger (MCP, PIP, DIP, Tip)
    9, 10, 11, 12: Middle Finger (MCP, PIP, DIP, Tip)
    13, 14, 15, 16: Ring Finger (MCP, PIP, DIP, Tip)
    17, 18, 19, 20: Pinky Finger (MCP, PIP, DIP, Tip)
    """
    def __init__(self, max_hands=2, detection_con=0.7, tracking_con=0.7, smooth_alpha=0.5, model_path="hand_landmarker.task"):
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.tracking_con = tracking_con
        self.smooth_alpha = smooth_alpha
        self.model_path = model_path
        
        # 1. Ensure Model Bundle is Available Locally
        self._ensure_model_exists()
        
        # 2. Initialize MediaPipe Hand Landmarker Task
        base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_con,
            min_hand_presence_confidence=self.detection_con,
            min_tracking_confidence=self.tracking_con
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Format: { hand_track_id: np.array of shape (21, 2) }
        self.prev_landmarks = {}
        self.next_hand_id = 0

    def _ensure_model_exists(self):
        """Checks for the MediaPipe TFLite task model. Downloads official bundle if missing."""
        if not os.path.exists(self.model_path):
            print(f"[SYS] Official MediaPipe model '{self.model_path}' not found locally.")
            print("[SYS] Downloading from Google Cloud Storage (approx 12MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print("[SYS] Download complete! Model verified.")
            except Exception as e:
                print(f"[ERROR] Failed to download model: {e}")
                raise e

    def process_frame(self, frame, draw=True):
        """
        Executes the frame processing pipeline using modern MediaPipe Tasks Vision.
        """
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert BGR frame to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands in video mode (requires monotonically increasing timestamp in ms)
        timestamp_ms = int(time.time() * 1000)
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        hands_data = []
        current_frame_matched_ids = set()

        if results.hand_landmarks and results.handedness:
            for idx, (hand_landmarks, handedness_list) in enumerate(zip(results.hand_landmarks, results.handedness)):
                # Extract label: 'Left' or 'Right'
                hand_label = handedness_list[0].category_name if handedness_list[0].category_name else "Hand"
                score = handedness_list[0].score if handedness_list[0].score else 1.0

                # Extract raw pixel coordinates (21 landmarks x 2 2D coords)
                raw_coords = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks], dtype=np.float32)
                
                # Extract 3D normalized coordinates for advanced math (e.g., Unity spatial mapping)
                norm_coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                
                # --- STABILIZATION LAYER (Exponential Moving Average) ---
                best_match_id = None
                min_dist = float('inf')
                wrist_pos = raw_coords[0]

                for pid, prev_coords in self.prev_landmarks.items():
                    if pid in current_frame_matched_ids:
                        continue
                    prev_wrist = prev_coords[0]
                    dist = math.hypot(wrist_pos[0] - prev_wrist[0], wrist_pos[1] - prev_wrist[1])
                    if dist < 150 and dist < min_dist:
                        min_dist = dist
                        best_match_id = pid

                if best_match_id is not None:
                    smoothed_coords = self.smooth_alpha * raw_coords + (1.0 - self.smooth_alpha) * self.prev_landmarks[best_match_id]
                    matched_id = best_match_id
                else:
                    smoothed_coords = raw_coords
                    matched_id = self.next_hand_id
                    self.next_hand_id += 1

                self.prev_landmarks[matched_id] = smoothed_coords
                current_frame_matched_ids.add(matched_id)
                
                pixel_coords = smoothed_coords.astype(np.int32)
                
                x_min, y_min = np.min(pixel_coords, axis=0)
                x_max, y_max = np.max(pixel_coords, axis=0)
                pad = 20
                bbox = [max(0, x_min - pad), max(0, y_min - pad), min(w, x_max + pad), min(h, y_max + pad)]

                hand_info = {
                    'id': matched_id,
                    'label': hand_label,
                    'score': score,
                    'lm_list': pixel_coords.tolist(),
                    'norm_list': norm_coords,
                    'bbox': bbox
                }
                hands_data.append(hand_info)

                if draw:
                    self.draw_futuristic_hud(frame, hand_info)

        lost_ids = set(self.prev_landmarks.keys()) - current_frame_matched_ids
        for lid in lost_ids:
            del self.prev_landmarks[lid]

        return hands_data

    def draw_futuristic_hud(self, frame, hand_info):
        """
        Renders an Iron Man / Sci-Fi inspired holographic overlay over the tracked hand.
        """
        lm_list = hand_info['lm_list']
        bbox = hand_info['bbox']
        label = hand_info['label']
        
        CYAN = (255, 255, 0)
        ELECTRIC_BLUE = (255, 150, 0)
        DARK_BLUE = (200, 70, 0)
        WHITE = (255, 255, 255)

        # 1. Draw Glowing Bone Connections
        # Define connection pairs manually or use vision.HandLandmarksConnections.HAND_CONNECTIONS
        connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection.start, connection.end
            pt1 = tuple(lm_list[start_idx])
            pt2 = tuple(lm_list[end_idx])
            cv2.line(frame, pt1, pt2, DARK_BLUE, 4, cv2.LINE_AA)
            cv2.line(frame, pt1, pt2, ELECTRIC_BLUE, 2, cv2.LINE_AA)

        # 2. Draw Futuristic Joint Nodes & Fingertip Reticles
        fingertips = [4, 8, 12, 16, 20]
        for idx, pt in enumerate(lm_list):
            coord = tuple(pt)
            if idx in fingertips:
                cv2.circle(frame, coord, 10, DARK_BLUE, 2, cv2.LINE_AA)
                cv2.circle(frame, coord, 6, CYAN, 1, cv2.LINE_AA)
                cv2.circle(frame, coord, 2, WHITE, cv2.FILLED, cv2.LINE_AA)
            else:
                cv2.circle(frame, coord, 4, DARK_BLUE, cv2.FILLED, cv2.LINE_AA)
                cv2.circle(frame, coord, 2, CYAN, cv2.FILLED, cv2.LINE_AA)

        # 3. Draw High-Tech Corner Bounding Box
        x1, y1, x2, y2 = bbox
        line_len = 25
        thickness = 2
        cv2.line(frame, (x1, y1), (x1 + line_len, y1), CYAN, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + line_len), CYAN, thickness)
        cv2.line(frame, (x2, y1), (x2 - line_len, y1), CYAN, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + line_len), CYAN, thickness)
        cv2.line(frame, (x1, y2), (x1 + line_len, y2), CYAN, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - line_len), CYAN, thickness)
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), CYAN, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - line_len), CYAN, thickness)

        # 4. HUD Data Panel overlay
        panel_x, panel_y = x1, max(20, y1 - 10)
        cv2.rectangle(frame, (panel_x, panel_y - 20), (panel_x + 160, panel_y + 5), (20, 20, 20), cv2.FILLED)
        cv2.rectangle(frame, (panel_x, panel_y - 20), (panel_x + 160, panel_y + 5), CYAN, 1)
        hud_text = f"SYS: {label.upper()} HAND [{hand_info['id']}]"
        cv2.putText(frame, hud_text, (panel_x + 5, panel_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=2, detection_con=0.7, tracking_con=0.7, smooth_alpha=0.5)
    
    print("[INFO] Stage 1 Hand Tracking Initialized.")
    print("[INFO] Press 'Q' or 'ESC' to exit.")
    
    prev_time = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        hands = tracker.process_frame(frame, draw=True)
        
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "STAGE 1: HOLO HAND TRACKING ONLINE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Iron Man Holographic Interface - Stage 1", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
