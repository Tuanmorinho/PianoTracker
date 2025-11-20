import cv2
import numpy as np

class KeyDetector:
    def __init__(self, target_hsv=None):
        if target_hsv is not None:
            self.set_target_color(target_hsv)
        else:
            # Default Orange (#ffaa2e -> HSV approx 18, 209, 255)
            # Range: H +/- 10, S +/- 50, V +/- 50
            self.lower_orange = np.array([8, 150, 150])
            self.upper_orange = np.array([28, 255, 255])

    def set_target_color(self, hsv):
        # hsv is a tuple or list: (H, S, V)
        # Create a range around the target
        # H: +/- 10, S: +/- 50, V: +/- 50 (clamped)
        h, s, v = hsv
        
        lower = np.array([max(0, h - 10), max(50, s - 50), max(50, v - 50)])
        upper = np.array([min(179, h + 10), 255, 255])
        
        self.lower_orange = lower
        self.upper_orange = upper
        print(f"Target Color Set: {hsv} -> Range: {lower} - {upper}")

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        return mask
