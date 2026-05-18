import cv2
import mediapipe as mp
import numpy as np
import math
import time

try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw
except ImportError:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_draw

class HandTracker:
    """
    Stage 1: Advanced Hand Tracking Module for Holographic Interface.
    
    Features:
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
    def __init__(self, max_hands=2, detection_con=0.7, tracking_con=0.7, smooth_alpha=0.5):
        """
        Initializes the MediaPipe Hands pipeline and stabilization parameters.
        
        :param max_hands: Maximum number of hands to detect.
        :param detection_con: Minimum confidence threshold for hand detection (0.0 to 1.0).
        :param tracking_con: Minimum confidence threshold for landmark tracking (0.0 to 1.0).
        :param smooth_alpha: Smoothing factor for EMA filter (0.0 = max smoothing/lag, 1.0 = no smoothing/raw).
        """
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.tracking_con = tracking_con
        self.smooth_alpha = smooth_alpha
        
        # Initialize MediaPipe Hands solution
        self.mp_hands = mp_hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.tracking_con
        )
        
        # Dictionary to maintain previous frame landmark positions for stabilization
        # Format: { hand_track_id: np.array of shape (21, 2) }
        self.prev_landmarks = {}
        self.next_hand_id = 0

    def process_frame(self, frame, draw=True):
        """
        Executes the frame processing pipeline: BGR->RGB conversion, neural network inference,
        coordinate extraction, temporal smoothing, and futuristic HUD visualization.
        
        :param frame: Raw BGR image frame from webcam.
        :param draw: Boolean flag to enable/disable futuristic HUD rendering on the frame.
        :return: list of dictionaries containing detailed tracking data for each detected hand.
        """
        h, w, c = frame.shape
        # MediaPipe requires RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)
        
        hands_data = []
        current_frame_matched_ids = set()

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # Extract label: 'Left' or 'Right'
                # Note: MediaPipe assumes a mirrored camera by default, flip label if needed, but standard is fine
                hand_label = handedness.classification[0].label
                score = handedness.classification[0].score

                # Extract raw pixel coordinates (21 landmarks x 2 2D coords)
                raw_coords = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark], dtype=np.float32)
                
                # Extract 3D normalized coordinates for advanced math (e.g., Unity spatial mapping)
                norm_coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                
                # --- STABILIZATION LAYER (Exponential Moving Average) ---
                # Match current hand to previous hands to ensure smooth tracking across frames
                best_match_id = None
                min_dist = float('inf')
                wrist_pos = raw_coords[0]

                for pid, prev_coords in self.prev_landmarks.items():
                    if pid in current_frame_matched_ids:
                        continue
                    # Calculate Euclidean distance between current wrist and previous wrist
                    prev_wrist = prev_coords[0]
                    dist = math.hypot(wrist_pos[0] - prev_wrist[0], wrist_pos[1] - prev_wrist[1])
                    # Threshold for matching hand across consecutive frames (150 pixels)
                    if dist < 150 and dist < min_dist:
                        min_dist = dist
                        best_match_id = pid

                if best_match_id is not None:
                    # Apply EMA smoothing formula: P_smooth = alpha * P_current + (1 - alpha) * P_prev
                    smoothed_coords = self.smooth_alpha * raw_coords + (1.0 - self.smooth_alpha) * self.prev_landmarks[best_match_id]
                    matched_id = best_match_id
                else:
                    # New hand detected, no previous history
                    smoothed_coords = raw_coords
                    matched_id = self.next_hand_id
                    self.next_hand_id += 1

                # Update tracking state
                self.prev_landmarks[matched_id] = smoothed_coords
                current_frame_matched_ids.add(matched_id)
                
                # Convert smoothed coords back to integer pixel positions for drawing and bounding box
                pixel_coords = smoothed_coords.astype(np.int32)
                
                # Calculate bounding box [x_min, y_min, x_max, y_max]
                x_min, y_min = np.min(pixel_coords, axis=0)
                x_max, y_max = np.max(pixel_coords, axis=0)
                # Add padding to bounding box
                pad = 20
                bbox = [max(0, x_min - pad), max(0, y_min - pad), min(w, x_max + pad), min(h, y_max + pad)]

                hand_info = {
                    'id': matched_id,
                    'label': hand_label,
                    'score': score,
                    'lm_list': pixel_coords.tolist(),       # 2D Pixel coordinates [x, y]
                    'norm_list': norm_coords,               # 3D Normalized coordinates [x, y, z]
                    'bbox': bbox                            # Bounding box [xmin, ymin, xmax, ymax]
                }
                hands_data.append(hand_info)

                # --- VISUALIZATION LAYER ---
                if draw:
                    self.draw_futuristic_hud(frame, hand_info)

        # Cleanup lost hands from previous landmarks dictionary
        lost_ids = set(self.prev_landmarks.keys()) - current_frame_matched_ids
        for lid in lost_ids:
            del self.prev_landmarks[lid]

        return hands_data

    def draw_futuristic_hud(self, frame, hand_info):
        """
        Renders an Iron Man / Sci-Fi inspired holographic overlay over the tracked hand.
        Uses neon blue/cyan palettes, glowing reticles, and structured wireframes.
        
        :param frame: Image frame to draw on.
        :param hand_info: Dictionary containing smoothed hand tracking data.
        """
        lm_list = hand_info['lm_list']
        bbox = hand_info['bbox']
        label = hand_info['label']
        
        # Color Palette (BGR format for OpenCV)
        CYAN = (255, 255, 0)         # Primary holographic glow
        ELECTRIC_BLUE = (255, 150, 0) # Secondary structural lines
        DARK_BLUE = (200, 70, 0)     # Background/accent glow
        WHITE = (255, 255, 255)      # Core highlight

        # 1. Draw Glowing Bone Connections (Wireframe)
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            pt1 = tuple(lm_list[start_idx])
            pt2 = tuple(lm_list[end_idx])
            
            # Outer glow line
            cv2.line(frame, pt1, pt2, DARK_BLUE, 4, cv2.LINE_AA)
            # Inner sharp core line
            cv2.line(frame, pt1, pt2, ELECTRIC_BLUE, 2, cv2.LINE_AA)

        # 2. Draw Futuristic Joint Nodes & Fingertip Reticles
        fingertips = [4, 8, 12, 16, 20]
        for idx, pt in enumerate(lm_list):
            coord = tuple(pt)
            if idx in fingertips:
                # Fingertip Reticle: Outer rotating/static ring + inner core
                cv2.circle(frame, coord, 10, DARK_BLUE, 2, cv2.LINE_AA)
                cv2.circle(frame, coord, 6, CYAN, 1, cv2.LINE_AA)
                cv2.circle(frame, coord, 2, WHITE, cv2.FILLED, cv2.LINE_AA)
            else:
                # Standard Joint Node
                cv2.circle(frame, coord, 4, DARK_BLUE, cv2.FILLED, cv2.LINE_AA)
                cv2.circle(frame, coord, 2, CYAN, cv2.FILLED, cv2.LINE_AA)

        # 3. Draw High-Tech Corner Bounding Box
        x1, y1, x2, y2 = bbox
        line_len = 25
        thickness = 2
        
        # Top-Left corner
        cv2.line(frame, (x1, y1), (x1 + line_len, y1), CYAN, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + line_len), CYAN, thickness)
        # Top-Right corner
        cv2.line(frame, (x2, y1), (x2 - line_len, y1), CYAN, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + line_len), CYAN, thickness)
        # Bottom-Left corner
        cv2.line(frame, (x1, y2), (x1 + line_len, y2), CYAN, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - line_len), CYAN, thickness)
        # Bottom-Right corner
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), CYAN, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - line_len), CYAN, thickness)

        # 4. HUD Data Panel overlay near the hand
        panel_x, panel_y = x1, max(20, y1 - 10)
        # Draw semi-transparent background box for text
        # (Using a solid dark box for clean, fast rendering without full frame copy overhead)
        cv2.rectangle(frame, (panel_x, panel_y - 20), (panel_x + 160, panel_y + 5), (20, 20, 20), cv2.FILLED)
        cv2.rectangle(frame, (panel_x, panel_y - 20), (panel_x + 160, panel_y + 5), CYAN, 1)
        
        hud_text = f"SYS: {label.upper()} HAND [{hand_info['id']}]"
        cv2.putText(frame, hud_text, (panel_x + 5, panel_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)


def main():
    """
    Standalone test function to run Stage 1 Hand Tracking directly.
    Displays webcam feed with Iron Man holographic tracking overlays and FPS counter.
    """
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=2, detection_con=0.7, tracking_con=0.7, smooth_alpha=0.5)
    
    print("[INFO] Stage 1 Hand Tracking Initialized.")
    print("[INFO] Press 'Q' or 'ESC' to exit.")
    
    prev_time = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[WARNING] Failed to read frame from webcam.")
            break
            
        # Flip frame horizontally for intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Process frame and draw HUD
        hands = tracker.process_frame(frame, draw=True)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Global HUD Header Overlay
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "STAGE 1: HOLO HAND TRACKING ONLINE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Iron Man Holographic Interface - Stage 1", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # Q or ESC
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
