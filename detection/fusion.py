import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionState:
    MATCH = "MATCH"
    AD = "AD"

class FusionEngine:
    def __init__(self, debounce_seconds=3.0):
        self.state = DetectionState.MATCH
        self.confidence_score = 1.0
        
        self.debounce_seconds = debounce_seconds
        self.last_transition_time = time.time()
        self.pending_state = None
        self.pending_start_time = None

    def update(self, logo_detected, logo_score, audio_high_energy, audio_rms):
        """
        Produce a fused state based on vision and audio inputs.
        
        Logic:
        - If logo is detected with high score, it's definitely a MATCH.
        - If logo is missing AND audio energy is high, it's likely an AD.
        - If logo is missing AND audio is low, we sustain current state or assume MATCH (silent transition).
        """
        
        # 1. Determine raw current observation
        if logo_detected:
            observed_state = DetectionState.MATCH
            score = logo_score
        elif audio_high_energy:
            observed_state = DetectionState.AD
            score = 0.8 # Confidence from audio only
        else:
            # Logo missing and audio is low.
            # This is likely an AD or a scene transition.
            observed_state = DetectionState.AD
            score = 0.6 # Lower confidence since it's quiet

        # 2. Debouncing logic
        if observed_state != self.state:
            if self.pending_state == observed_state:
                # Still in pending state, check if debounce time passed
                if time.time() - self.pending_start_time >= self.debounce_seconds:
                    logger.info(f"State transition: {self.state} -> {observed_state}")
                    self.state = observed_state
                    self.last_transition_time = time.time()
                    self.pending_state = None
            else:
                # Start new pending transition
                self.pending_state = observed_state
                self.pending_start_time = time.time()
        else:
            # Observe matches current state, reset pending
            self.pending_state = None
            self.pending_start_time = None

        self.confidence_score = score
        return self.state, self.confidence_score

    def get_state(self):
        return self.state, self.confidence_score