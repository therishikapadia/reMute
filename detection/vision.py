import cv2
import numpy as np
import os
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionDetector:
    def __init__(self, template_path=None, roi=(0, 0, 640, 300), threshold=0.7):
        """
        roi       : (x, y, w, h) within the 640x480 resized frame
        threshold : TM_CCOEFF_NORMED score above which we consider it a match
        """
        self.roi       = roi
        self.threshold = threshold

        self.template       = None
        self.template_gray  = None
        self.template_size  = None  # (w, h)

        self.cap        = None
        self._lock      = threading.Lock()
        self.last_score = 0.0
        self.is_match   = False

        if template_path and os.path.exists(template_path):
            self.load_template(template_path)

    # -------------------------------------------------------------- #

    def load_template(self, path):
        """Load and store the reference logo template."""
        tmpl = cv2.imread(path)
        if tmpl is None:
            logger.error(f"Could not load template: {path}")
            return False
        self.template      = tmpl
        self.template_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        self.template_size = (self.template_gray.shape[1], self.template_gray.shape[0])  # (w, h)
        logger.info(f"Template loaded: {path}, size={self.template_size}")
        return True

    def start_camera(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            logger.error("Could not open webcam.")
            return False
        return True

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    # -------------------------------------------------------------- #

    def process_frame(self):
        """
        Capture one frame, run template matching inside ROI.
        Returns (annotated_frame, score, is_match).
        """
        if not self.cap:
            return None, 0.0, False

        ret, frame = self.cap.read()
        if not ret:
            return None, 0.0, False

        # 1. Resize to fixed working resolution
        small_frame = cv2.resize(frame, (640, 480))

        # 2. Clamp ROI to frame bounds
        rx, ry, rw, rh = self.roi
        rx = max(0, rx)
        ry = max(0, ry)
        rw = min(rw, 640 - rx)
        rh = min(rh, 480 - ry)

        roi_frame = small_frame[ry:ry + rh, rx:rx + rw]
        cv2.rectangle(small_frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)

        # 3. Grayscale ROI
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        score    = 0.0
        is_match = False

        # 4. Template matching (only if template fits inside ROI)
        if self.template_gray is not None:
            th, tw = self.template_gray.shape[:2]
            rh_actual, rw_actual = gray_roi.shape[:2]

            if tw > rw_actual or th > rh_actual:
                logger.warning(
                    f"Template ({tw}x{th}) larger than ROI ({rw_actual}x{rh_actual}). Skipping."
                )
            else:
                res = cv2.matchTemplate(gray_roi, self.template_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                score = float(max_val)

                if score >= self.threshold:
                    is_match = True
                    tl = max_loc
                    br = (tl[0] + tw, tl[1] + th)
                    cv2.rectangle(roi_frame, tl, br, (0, 255, 0), 2)

        # 5. Overlay score text
        color = (0, 255, 0) if is_match else (0, 0, 255)
        cv2.putText(
            small_frame,
            f"Score: {score:.3f}  {'LIVE' if is_match else 'AD?'}",
            (rx, max(ry - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

        # 6. Thread-safe write
        with self._lock:
            self.last_score = score
            self.is_match   = is_match

        return small_frame, score, is_match

    def get_result(self):
        """Thread-safe read of latest result."""
        with self._lock:
            return self.last_score, self.is_match


# ------------------------------------------------------------------ #
#  Standalone test                                                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    import time

    template_path = sys.argv[1] if len(sys.argv) > 1 else None

    detector = VisionDetector(
        template_path=template_path,
        roi=(0, 0, 640, 120),
        threshold=0.65
    )

    if not detector.start_camera():
        sys.exit(1)

    print("Running... Press 'q' to quit.\n")

    try:
        while True:
            frame, score, matched = detector.process_frame()
            if frame is not None:
                cv2.imshow("Vision Detector", frame)
                print(f"Score: {score:.4f}  Match: {str(matched):<5}", end="\r")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop_camera()
        cv2.destroyAllWindows()
        print("\nStopped.")