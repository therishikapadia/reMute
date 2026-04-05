import cv2
import numpy as np
import time
import os

class VisionDetector:
    def __init__(self, template_path=None, roi=(0, 0, 640, 300), threshold=0.7):
        """
        roi: (x, y, w, h) for the region to look for the logo.
        threshold: The similarity score above which we consider it a match.
        """
        self.template_path = template_path
        self.roi = roi
        self.threshold = threshold
        self.template = None
        self.template_gray = None
        self.template_size = None
        
        self.cap = None
        self.last_score = 0.0
        self.is_match = False
        
        if template_path and os.path.exists(template_path):
            self.load_template(template_path)

    def load_template(self, path):
        """Load the reference logo template."""
        self.template = cv2.imread(path)
        if self.template is not None:
            self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            self.template_size = self.template_gray.shape[::-1]  # (w, h)
            return True
        return False

    def start_camera(self, camera_index=0):
        """Open the webcam."""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return False
        return True

    def stop_camera(self):
        """Release the webcam."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def process_frame(self):
        """Capture and process a single frame."""
        if not self.cap:
            return None, 0.0, False

        ret, frame = self.cap.read()
        if not ret:
            return None, 0.0, False

        # 1. Resize frame for efficiency
        small_frame = cv2.resize(frame, (640, 480))
        
        # 2. Extract ROI
        rx, ry, rw, rh = self.roi
        # Ensure ROI is within bounds
        rx, ry = max(0, rx), max(0, ry)
        rw = min(rw, 640 - rx)
        rh = min(rh, 480 - ry)
        
        roi_frame = small_frame[ry:ry+rh, rx:rx+rw]
        
        # Draw ROI Boundary on visualization frame
        cv2.rectangle(small_frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)
        
        # 3. Convert to grayscale
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        score = 0.0
        is_match = False
        
        # 4. Template Matching
        if self.template_gray is not None:
            res = cv2.matchTemplate(gray_roi, self.template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            score = max_val
            if score >= self.threshold:
                is_match = True
                # Optional: draw rectangle on frame for debugging
                top_left = max_loc
                cv2.rectangle(roi_frame, top_left, (top_left[0] + self.template_size[0], top_left[1] + self.template_size[1]), (0, 255, 0), 2)

        self.last_score = score
        self.is_match = is_match
        
        return small_frame, score, is_match

if __name__ == "__main__":
    # Test vision detection
    detector = VisionDetector()
    if detector.start_camera():
        while True:
            frame, score, matched = detector.process_frame()
            if frame is not None:
                cv2.imshow("Detection ROI", frame)
                print(f"Score: {score:.4f}, Match: {matched}", end="\r")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        detector.stop_camera()
        cv2.destroyAllWindows()
