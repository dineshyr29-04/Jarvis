import cv2
import numpy as np
import math

# Try importing MediaPipe components individually to fix environment issues
try:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_draw
except (ImportError, AttributeError):
    try:
        from mediapipe.python.solutions import hands as mp_hands
        from mediapipe.python.solutions import drawing_utils as mp_draw
    except (ImportError, AttributeError):
        # Last ditch effort for certain installations
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils

class HolographicInterface:
    def __init__(self):
        # 1. Initialize MediaPipe Hand Tracking
        self.mp_hands = mp_hands
        self.mp_draw = mp_draw
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )


        # 2. Interface State
        self.obj_pos = [320, 240]  # Center of a 640x480 screen
        self.obj_size = 50
        self.is_selected = False
        self.base_color = (255, 191, 0)  # Cyan/Blue holographic color (BGR)
        
        # 3. Visual effects buffer
        self.angle = 0

    def get_hand_landmarks(self, frame):
        """Processes the frame and returns hand landmarks and handedness."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    h, w, c = frame.shape
                    landmarks.append((int(lm.x * w), int(lm.y * h)))
        return landmarks

    def detect_gestures(self, landmarks):
        """
        Mathematical Gesture Detection:
        - Pinch: Distance between Thumb Tip (4) and Index Tip (8).
        - Scale: Distance between Index Tip (8) and Middle Tip (12).
        - Position: Follows Index Finger Tip (8).
        """
        if not landmarks:
            return None

        # Landmarks IDs: 4=Thumb Tip, 8=Index Tip, 12=Middle Tip
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        # Calculate Euclidean Distances
        pinch_dist = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
        scale_dist = math.hypot(index_tip[0] - middle_tip[0], index_tip[1] - middle_tip[1])

        # Logic
        # 1. Pinch to select (if close enough)
        self.is_selected = pinch_dist < 40
        
        # 2. Movement (if selected, move object to index tip)
        if self.is_selected:
            self.obj_pos[0] = index_tip[0]
            self.obj_pos[1] = index_tip[1]
            
            # 3. Scaling (Map distance between index and middle to size)
            # Normal range is roughly 30 to 150 pixels
            self.obj_size = int(np.interp(scale_dist, [30, 150], [40, 150]))

    def draw_hologram(self, frame, landmarks):
        """Creates the glowing holographic effect."""
        overlay = frame.copy()
        
        # Draw hand connections with glow
        if landmarks:
            # Draw lines between specific points for a 'tech' look
            for connection in self.mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                pt1 = landmarks[start_idx]
                pt2 = landmarks[end_idx]
                
                # Layered lines for glow effect
                cv2.line(frame, pt1, pt2, self.base_color, 1)
                cv2.line(overlay, pt1, pt2, self.base_color, 4)

        # Draw the Interactive Object (Wireframe Cube/Circle)
        center = tuple(self.obj_pos)
        color = (0, 255, 255) if self.is_selected else self.base_color # Yellow when selected
        
        # Rotating Wireframe Square (simulating 3D)
        self.angle += 0.05
        size = self.obj_size
        pts = np.array([
            [center[0] + size * math.cos(self.angle), center[1] + size * math.sin(self.angle)],
            [center[0] + size * math.cos(self.angle + math.pi/2), center[1] + size * math.sin(self.angle + math.pi/2)],
            [center[0] + size * math.cos(self.angle + math.pi), center[1] + size * math.sin(self.angle + math.pi)],
            [center[0] + size * math.cos(self.angle + 3*math.pi/2), center[1] + size * math.sin(self.angle + 3*math.pi/2)]
        ], np.int32)
        
        cv2.polylines(frame, [pts], True, color, 2)
        cv2.polylines(overlay, [pts], True, color, 6) # Outer glow layer

        # Compose the final frame with transparency
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add tech-style HUD text
        cv2.putText(frame, f"STATUS: {'ACTIVE' if self.is_selected else 'IDLE'}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"SCALE: {self.obj_size}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        print("Starting Holographic Interface...")
        print("Controls: Pinch (Thumb+Index) to move. Distance (Index+Middle) to scale. 'q' to quit.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # 1. Track
            lms = self.get_hand_landmarks(frame)
            
            # 2. Detect
            self.detect_gestures(lms)
            
            # 3. Render
            self.draw_hologram(frame, lms)

            cv2.imshow("Iron Man Interface", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ui = HolographicInterface()
    ui.run()
