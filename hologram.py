import cv2
import mediapipe as mp
import numpy as np
import math
import sys

# DEBUG PRINTS
print("PYTHON VERSION:", sys.version)

try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw
    print("SUCCESS: Imported via mediapipe.python.solutions")
except ImportError:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_draw
    print("SUCCESS: Imported via mediapipe.solutions")

class HolographicInterface:
    def __init__(self): 
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.obj_pos = [320, 240]
        self.obj_size = 60
        self.is_selected = False
        self.angle = 0

    def run(self):
        cap = cv2.VideoCapture(0)
        print("System Online. Press Q to exit.")
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            landmarks = []
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                for lm in hand_lms.landmark:
                    landmarks.append((int(lm.x * w), int(lm.y * h)))
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                t, i, m = landmarks[4], landmarks[8], landmarks[12]
                if math.hypot(i[0]-t[0], i[1]-t[1]) < 45:
                    self.is_selected = True
                    self.obj_pos = [i[0], i[1]]
                    s_dist = math.hypot(i[0]-m[0], i[1]-m[1])
                    self.obj_size = int(np.interp(s_dist, [30, 180], [40, 200]))
                else: self.is_selected = False

            overlay = frame.copy()
            c, s = self.obj_pos, self.obj_size
            color = (0, 255, 255) if self.is_selected else (255, 191, 0)
            self.angle += 0.05
            pts = []
            for j in range(4):
                theta = self.angle + (j * math.pi / 2)
                pts.append([int(c[0] + s * math.cos(theta)), int(c[1] + s * math.sin(theta))])
            pts = np.array(pts, np.int32)
            cv2.polylines(frame, [pts], True, color, 2)
            cv2.polylines(overlay, [pts], True, color, 8)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, "JARVIS LIVE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Hologram UI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
                HolographicInterface().run()
