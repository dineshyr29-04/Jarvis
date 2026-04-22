import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector

class HolographicInterface:
    def __init__(self):
        self.detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)
        self.obj_pos = [320, 240]
        self.obj_size = 50
        self.is_selected = False
        self.base_color = (255, 191, 0)
        self.angle = 0

    def detect_gestures(self, frame):
        hands, frame = self.detector.findHands(frame, draw=False, flipType=False)
        landmarks = []
        if hands:
            hand = hands[0]
            landmarks = hand["lmList"]
            thumb_tip = landmarks[4][:2]
            index_tip = landmarks[8][:2]
            middle_tip = landmarks[12][:2]
            pinch_dist, _, _ = self.detector.findDistance(thumb_tip, index_tip)
            self.is_selected = pinch_dist < 40
            if self.is_selected:
                self.obj_pos[0], self.obj_pos[1] = index_tip
                scale_dist, _, _ = self.detector.findDistance(index_tip, middle_tip)
                self.obj_size = int(np.interp(scale_dist, [30, 200], [40, 250]))
        return landmarks

    def draw_hologram(self, frame, landmarks):
        overlay = frame.copy()
        if landmarks:
            connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
            for s, e in connections:
                pt1, pt2 = (landmarks[s][0], landmarks[s][1]), (landmarks[e][0], landmarks[e][1])
                cv2.line(frame, pt1, pt2, self.base_color, 1)
                cv2.line(overlay, pt1, pt2, self.base_color, 4)
        center, color = tuple(self.obj_pos), ((0,255,255) if self.is_selected else self.base_color)
        self.angle += 0.05
        size = self.obj_size
        pts = np.array([[center[0]+size*math.cos(self.angle), center[1]+size*math.sin(self.angle)], [center[0]+size*math.cos(self.angle+1.57), center[1]+size*math.sin(self.angle+1.57)], [center[0]+size*math.cos(self.angle+3.14), center[1]+size*math.sin(self.angle+3.14)], [center[0]+size*math.cos(self.angle+4.71), center[1]+size*math.sin(self.angle+4.71)]], np.int32)
        cv2.polylines(frame, [pts], True, color, 2)
        cv2.polylines(overlay, [pts], True, color, 6)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, "IRON MAN UI v2", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            lms = self.detect_gestures(frame)
            self.draw_hologram(frame, lms)
            cv2.imshow("Jarvis Interface", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ui = HolographicInterface()
    ui.run()