import cv2
import numpy as np
import math
import time
from tracking import HandTracker

class GestureRecognizer:
    """
    Stage 2: Advanced Gesture Recognition & Mathematical Analysis Layer.
    
    This module analyzes spatial landmark coordinates from the HandTracker subsystem
    to classify discrete gestures (Pinch, Grab, Open Palm, Swipes) and calculate continuous
    transformation metrics (Two-Hand Scaling, 3D Rotation Angles, Translation Vectors).
    
    Mathematical Foundations:
    1. Euclidean Distance: d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
       Used for Pinch detection (Thumb tip to Index tip) and Grab detection (Fingertips to Palm).
    2. Vector Mathematics & Arctangent: theta = atan2(dy, dx) * (180 / pi)
       Used for Two-Hand Rotation calculations.
    3. Linear Interpolation: np.interp(val, [in_min, in_max], [out_min, out_max])
       Used for mapping inter-hand distance to smooth holographic scaling factors.
    4. Temporal Velocity Vectors: dP = P_curr - P_prev over a sliding window.
       Used for directional Swipe detection.
    """
    def __init__(self, pinch_threshold=45, grab_threshold=70, swipe_window=5):
        """
        Initializes gesture classification thresholds and historical buffers.
        
        :param pinch_threshold: Max Euclidean distance (px) between Thumb and Index tip for a Pinch.
        :param grab_threshold: Max average distance (px) of fingertips to palm center for a Grab.
        :param swipe_window: Number of historical frames to track for velocity/swipe detection.
        """
        self.pinch_threshold = pinch_threshold
        self.grab_threshold = grab_threshold
        self.swipe_window = swipe_window
        
        # Historical buffers for swipe detection: { hand_id: [(x, y, time), ...] }
        self.history = {}
        # Store initial baseline distance for two-hand scaling
        self.initial_two_hand_dist = None
        self.base_scale = 1.0

    def analyze(self, hands_data):
        """
        Processes raw hand tracking data to extract gesture classifications and 3D manipulation metrics.
        
        :param hands_data: List of hand dictionaries from HandTracker.process_frame().
        :return: Dictionary containing comprehensive gesture analysis and multi-hand interaction states.
        """
        current_ids = set()
        analysis_result = {
            'hands': [],
            'two_hand_active': False,
            'two_hand_mode': None,       # 'SCALE_ROTATE' or None
            'inter_hand_dist': 0.0,
            'scale_multiplier': 1.0,
            'rotation_angle': 0.0,
            'global_swipe': None         # 'SWIPE_LEFT', 'SWIPE_RIGHT', 'SWIPE_UP', 'SWIPE_DOWN'
        }

        for hand in hands_data:
            hid = hand['id']
            current_ids.add(hid)
            lm_list = hand['lm_list']
            
            # Extract key anatomical landmarks
            wrist = lm_list[0]
            thumb_tip = lm_list[4]
            index_tip = lm_list[8]
            middle_tip = lm_list[12]
            ring_tip = lm_list[16]
            pinky_tip = lm_list[20]
            palm_center = lm_list[9] # Knuckle of middle finger serves as stable palm center

            # --- 1. PINCH DETECTION (Euclidean Distance) ---
            pinch_dist = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
            is_pinching = pinch_dist < self.pinch_threshold
            pinch_center = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)

            # --- 2. GRAB DETECTION (Average Fingertip-to-Palm Distance) ---
            fingertips = [index_tip, middle_tip, ring_tip, pinky_tip]
            curl_distances = [math.hypot(ft[0] - palm_center[0], ft[1] - palm_center[1]) for ft in fingertips]
            avg_curl = sum(curl_distances) / len(curl_distances)
            is_grabbing = avg_curl < self.grab_threshold

            # Classify primary discrete gesture
            if is_grabbing:
                gesture_type = "GRAB"
                action_center = palm_center
            elif is_pinching:
                gesture_type = "PINCH"
                action_center = pinch_center
            else:
                gesture_type = "OPEN_PALM"
                action_center = palm_center

            # --- 3. SWIPE DETECTION (Temporal Velocity Analysis) ---
            curr_time = time.time()
            if hid not in self.history:
                self.history[hid] = []
            self.history[hid].append((palm_center[0], palm_center[1], curr_time))
            
            # Maintain sliding window size
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
                    
                    # Velocity thresholds for swipe classification (px per second)
                    vel_thresh = 800.0
                    if abs(vx) > vel_thresh and abs(vx) > abs(vy):
                        swipe_dir = "SWIPE_RIGHT" if vx > 0 else "SWIPE_LEFT"
                        analysis_result['global_swipe'] = swipe_dir
                        self.history[hid].clear() # Reset buffer after trigger
                    elif abs(vy) > vel_thresh and abs(vy) > abs(vx):
                        swipe_dir = "SWIPE_DOWN" if vy > 0 else "SWIPE_UP"
                        analysis_result['global_swipe'] = swipe_dir
                        self.history[hid].clear()

            hand_summary = {
                'id': hid,
                'label': hand['label'],
                'gesture': gesture_type,
                'action_center': action_center,
                'pinch_dist': pinch_dist,
                'grab_dist': avg_curl,
                'swipe': swipe_dir
            }
            analysis_result['hands'].append(hand_summary)

        # Cleanup lost hand histories
        lost_ids = set(self.history.keys()) - current_ids
        for lid in lost_ids:
            del self.history[lid]

        # --- 4. TWO-HAND ADVANCED MANIPULATION (Scaling & Rotation) ---
        if len(analysis_result['hands']) == 2:
            h1 = analysis_result['hands'][0]
            h2 = analysis_result['hands'][1]
            
            # Check if both hands are actively engaged in manipulation (Pinch or Grab)
            active_gestures = ["PINCH", "GRAB"]
            if h1['gesture'] in active_gestures and h2['gesture'] in active_gestures:
                analysis_result['two_hand_active'] = True
                analysis_result['two_hand_mode'] = 'SCALE_ROTATE'
                
                c1 = h1['action_center']
                c2 = h2['action_center']
                
                # Calculate Inter-Hand Euclidean Distance
                dist = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
                analysis_result['inter_hand_dist'] = dist
                
                # Calculate Scaling Factor via Interpolation
                if self.initial_two_hand_dist is None:
                    self.initial_two_hand_dist = dist
                else:
                    # Scale multiplier relative to initial grab distance
                    # Using smooth exponential mapping
                    scale_ratio = dist / max(1.0, self.initial_two_hand_dist)
                    analysis_result['scale_multiplier'] = scale_ratio

                # Calculate 2D Rotation Angle (Arctangent of vector connecting hands)
                dx = c2[0] - c1[0]
                dy = c2[1] - c1[1]
                # math.atan2 returns angle in radians [-pi, pi], convert to degrees
                angle_deg = math.atan2(dy, dx) * (180.0 / math.pi)
                analysis_result['rotation_angle'] = angle_deg
            else:
                self.initial_two_hand_dist = None # Reset baseline when grab released
        else:
            self.initial_two_hand_dist = None

        return analysis_result


class InteractiveHologram:
    """
    Stage 2 Interactive Testing Object: 3D Holographic Wireframe Cube & HUD Rings.
    Allows the user to visually test Pinch-to-Grab, Two-Hand Scaling, 3D Rotation, and Swipes.
    """
    def __init__(self, x=320, y=240, size=80):
        self.pos = [x, y]
        self.base_size = size
        self.current_size = size
        self.target_size = size
        
        # 3D Rotation Angles (Degrees)
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0
        
        self.is_grabbed = False
        self.grab_offset = [0, 0]
        
        # Color Palettes (BGR) - Swiping changes the active palette
        self.palettes = [
            (255, 255, 0),    # Cyan / Neon Blue
            (255, 150, 0),    # Electric Blue
            (0, 255, 150),    # Emerald Green
            (255, 0, 255),    # Magenta
        ]
        self.palette_idx = 0
        self.color = self.palettes[self.palette_idx]
        self.pulse_alpha = 0.0

    def update(self, gesture_data):
        """Updates hologram physics, scale, rotation, and position based on gesture analysis."""
        hands = gesture_data['hands']
        
        # Handle Global Swipes (Cycle Palette & Trigger Pulse)
        if gesture_data['global_swipe']:
            self.palette_idx = (self.palette_idx + 1) % len(self.palettes)
            self.color = self.palettes[self.palette_idx]
            self.pulse_alpha = 1.0 # Trigger visual flash
            print(f"[HUD] Swipe Detected ({gesture_data['global_swipe']}). Palette switched.")

        # Handle Two-Hand Scaling & Rotation
        if gesture_data['two_hand_active']:
            self.is_grabbed = True
            # Update target size based on scale multiplier
            self.target_size = self.base_size * gesture_data['scale_multiplier']
            # Clamp size to prevent hologram from exploding or disappearing
            self.target_size = max(30, min(300, self.target_size))
            
            # Smooth scaling via linear interpolation (EMA)
            self.current_size += (self.target_size - self.current_size) * 0.2
            
            # Update Z-rotation directly from hand tilt angle
            self.angle_z = gesture_data['rotation_angle']
            
            # Position hologram exactly between the two hands
            h1_pos = hands[0]['action_center']
            h2_pos = hands[1]['action_center']
            self.pos[0] = (h1_pos[0] + h2_pos[0]) // 2
            self.pos[1] = (h1_pos[1] + h2_pos[1]) // 2

        # Handle Single Hand Grab / Pinch
        elif len(hands) == 1:
            hand = hands[0]
            if hand['gesture'] in ["PINCH", "GRAB"]:
                if not self.is_grabbed:
                    # Initial grab lock - calculate offset
                    self.is_grabbed = True
                    self.grab_offset = [self.pos[0] - hand['action_center'][0], self.pos[1] - hand['action_center'][1]]
                else:
                    # Follow hand smoothly
                    target_x = hand['action_center'][0] + self.grab_offset[0]
                    target_y = hand['action_center'][1] + self.grab_offset[1]
                    self.pos[0] += int((target_x - self.pos[0]) * 0.4)
                    self.pos[1] += int((target_y - self.pos[1]) * 0.4)
            else:
                self.is_grabbed = False
                # If palm open, lock new baseline size for next two-hand scale
                self.base_size = self.current_size
        else:
            self.is_grabbed = False
            self.base_size = self.current_size

        # Idle Animations when not grabbed
        if not self.is_grabbed:
            self.angle_x += 1.5
            self.angle_y += 1.0
        
        # Decay pulse effect
        if self.pulse_alpha > 0:
            self.pulse_alpha -= 0.05

    def render(self, frame):
        """Renders the 3D wireframe cube and HUD rings onto the image frame."""
        cx, cy = int(self.pos[0]), int(self.pos[1])
        s = self.current_size
        
        # Base colors
        glow_color = self.color
        core_color = (255, 255, 255) if self.is_grabbed else glow_color

        # 1. Draw Outer Rotating HUD Ring
        ring_radius = int(s * 1.4)
        cv2.circle(frame, (cx, cy), ring_radius, glow_color, 1, cv2.LINE_AA)
        
        # Draw dynamic tick marks along the ring
        num_ticks = 12
        for i in range(num_ticks):
            theta = math.radians(self.angle_z + (i * 360 / num_ticks))
            r1 = ring_radius - 6
            r2 = ring_radius + 6
            pt1 = (int(cx + r1 * math.cos(theta)), int(cy + r1 * math.sin(theta)))
            pt2 = (int(cx + r2 * math.cos(theta)), int(cy + r2 * math.sin(theta)))
            cv2.line(frame, pt1, pt2, glow_color, 2, cv2.LINE_AA)

        # 2. 3D Wireframe Cube Projection Math
        # Define 8 vertices of a 3D cube [-1, 1]
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ], dtype=np.float32)

        # Rotation Matrices
        rad_x = math.radians(self.angle_x)
        rad_y = math.radians(self.angle_y)
        rad_z = math.radians(self.angle_z)

        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(rad_x), -math.sin(rad_x)],
            [0, math.sin(rad_x), math.cos(rad_x)]
        ])
        rot_y = np.array([
            [math.cos(rad_y), 0, math.sin(rad_y)],
            [0, 1, 0],
            [-math.sin(rad_y), 0, math.cos(rad_y)]
        ])
        rot_z = np.array([
            [math.cos(rad_z), -math.sin(rad_z), 0],
            [math.sin(rad_z), math.cos(rad_z), 0],
            [0, 0, 1]
        ])

        # Apply rotations and scale
        projected_pts = []
        for v in vertices:
            # Rotate
            rotated = rot_z @ (rot_y @ (rot_x @ v))
            # Perspective projection factor (simple orthographic + scaling for HUD feel)
            px = int(cx + rotated[0] * s * 0.7)
            py = int(cy + rotated[1] * s * 0.7)
            projected_pts.append((px, py))

        # Define 12 edges connecting the 8 vertices
        edges = [
            (0,1), (1,2), (2,3), (3,0), # Back face
            (4,5), (5,6), (6,7), (7,4), # Front face
            (0,4), (1,5), (2,6), (3,7)  # Connecting struts
        ]

        # Render 3D Cube Edges
        for edge in edges:
            pt1 = projected_pts[edge[0]]
            pt2 = projected_pts[edge[1]]
            cv2.line(frame, pt1, pt2, core_color, 2, cv2.LINE_AA)

        # Render Vertex Nodes
        for pt in projected_pts:
            cv2.circle(frame, pt, 4, glow_color, cv2.FILLED, cv2.LINE_AA)

        # 3. Render Status Overlay & Interaction Mode Text
        status_text = "MODE: TWO-HAND SCALE/ROTATE" if gesture_data['two_hand_active'] else ("MODE: GRABBED" if self.is_grabbed else "MODE: IDLE (OPEN PALM)")
        cv2.putText(frame, status_text, (cx - 80, cy - ring_radius - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, core_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"SCALE: {s:.1f}px | ROT: {self.angle_z:.1f}deg", (cx - 85, cy + ring_radius + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, glow_color, 1, cv2.LINE_AA)

        # 4. Energy Pulse Flash Overlay (When Swiped)
        if self.pulse_alpha > 0:
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), int(ring_radius * 1.5), glow_color, cv2.FILLED)
            cv2.addWeighted(overlay, self.pulse_alpha * 0.4, frame, 1.0 - (self.pulse_alpha * 0.4), 0, frame)


def main():
    """
    Standalone execution for Stage 2. Launches the webcam, initializes HandTracker
    and GestureRecognizer, and displays the interactive 3D holographic wireframe object.
    """
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=2, detection_con=0.7, tracking_con=0.7, smooth_alpha=0.5)
    recognizer = GestureRecognizer(pinch_threshold=45, grab_threshold=70, swipe_window=5)
    hologram = InteractiveHologram(x=320, y=240, size=80)
    
    print("[INFO] Stage 2 Gesture Recognition & Interactive Hologram Initialized.")
    print("[INFO] Press 'Q' or 'ESC' to exit.")
    
    prev_time = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        
        # 1. Run Stage 1 Tracking Pipeline
        hands_data = tracker.process_frame(frame, draw=True)
        
        # 2. Run Stage 2 Gesture Analysis Pipeline
        gesture_analysis = recognizer.analyze(hands_data)
        
        # 3. Update & Render Interactive Holographic Object
        hologram.update(gesture_analysis)
        hologram.render(frame)
        
        # FPS Counter & Header
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "STAGE 2: GESTURE ENGINE & HOLO-MANIPULATION", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Iron Man Holographic Interface - Stage 2", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
