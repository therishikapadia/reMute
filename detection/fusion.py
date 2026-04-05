import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Detection States                                                    #
# ------------------------------------------------------------------ #

class DetectionState:
    LIVE = "LIVE"
    AD   = "AD"


# ------------------------------------------------------------------ #
#  Fusion Engine                                                       #
# ------------------------------------------------------------------ #

class FusionEngine:
    def __init__(self, debounce_seconds=3.0, flicker_limit=10, audio_rms_baseline=0.05):
        """
        debounce_seconds   : seconds a new observed state must persist before committing
        flicker_limit      : rapid flicker resets allowed before forcing a transition
        audio_rms_baseline : RMS value treated as normal loudness for normalisation
        """
        self.state      = DetectionState.LIVE
        self.confidence = 1.0

        self.debounce_seconds    = debounce_seconds
        self.flicker_limit       = flicker_limit
        self.audio_rms_baseline  = audio_rms_baseline

        self.pending_state         = None
        self.pending_start_time    = None
        self.pending_flicker_count = 0

        self.last_transition_time = time.time()

    # -------------------------------------------------------------- #

    def _compute_ad_score(self, logo_detected, logo_score, audio_high_energy, audio_rms):
        """
        Returns 0.0–1.0 probability that current moment is an AD.
        Higher = more likely ad.
        """
        audio_norm = min(audio_rms / max(self.audio_rms_baseline, 1e-6), 1.0)

        if logo_detected:
            # Logo present → almost certainly LIVE
            return 1.0 - logo_score

        if audio_high_energy:
            # No logo + loud audio → strong ad signal
            return 0.5 + 0.5 * audio_norm

        # No logo + quiet audio → uncertain, lean toward AD
        return 0.3 + 0.2 * audio_norm

    # -------------------------------------------------------------- #

    def update(self, logo_detected, logo_score, audio_high_energy, audio_rms):
        """
        Call on every tick with latest vision + audio values.
        Returns (state, confidence).

        logo_detected    : bool
        logo_score       : float 0.0–1.0 from VisionDetector
        audio_high_energy: bool from AudioDetector
        audio_rms        : float from AudioDetector
        """
        ad_score = self._compute_ad_score(logo_detected, logo_score, audio_high_energy, audio_rms)

        observed_state = DetectionState.AD if ad_score >= 0.5 else DetectionState.LIVE
        confidence     = ad_score if observed_state == DetectionState.AD else (1.0 - ad_score)

        # Debounce with flicker protection
        if observed_state != self.state:
            if self.pending_state == observed_state:
                # Same pending — check if debounce time passed
                if time.time() - self.pending_start_time >= self.debounce_seconds:
                    logger.info(f"State: {self.state} → {observed_state}  (confidence={confidence:.3f})")
                    self.state                 = observed_state
                    self.last_transition_time  = time.time()
                    self.pending_state         = None
                    self.pending_flicker_count = 0
            else:
                # Different pending — count as flicker
                self.pending_flicker_count += 1

                if self.pending_flicker_count >= self.flicker_limit:
                    # Sustained pressure despite flickers — force new pending
                    logger.debug(f"Flicker limit hit, forcing pending → {observed_state}")
                    self.pending_state         = observed_state
                    self.pending_start_time    = time.time()
                    self.pending_flicker_count = 0
                else:
                    self.pending_state      = observed_state
                    self.pending_start_time = time.time()
        else:
            # Observed matches committed state — clear pending
            self.pending_state         = None
            self.pending_start_time    = None
            self.pending_flicker_count = 0

        self.confidence = confidence
        return self.state, self.confidence

    def get_state(self):
        return self.state, self.confidence


# ------------------------------------------------------------------ #
#  Standalone test (simulated inputs)                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    engine = FusionEngine(debounce_seconds=2.0)

    test_cases = [
        # (logo_detected, logo_score, audio_high_energy, audio_rms, label)
        (True,  0.85, False, 0.02, "Normal play"),
        (True,  0.80, False, 0.02, "Normal play"),
        (False, 0.00, True,  0.09, "Ad starts"),
        (False, 0.00, True,  0.09, "Ad continues"),
        (False, 0.00, True,  0.09, "Ad continues"),
        (True,  0.10, False, 0.02, "Flicker back"),
        (False, 0.00, True,  0.08, "Ad again"),
        (False, 0.00, True,  0.08, "Ad again"),
        (False, 0.00, True,  0.08, "Ad again — debounce should trigger"),
    ]

    print(f"{'Label':<40} {'State':<6} {'Confidence'}")
    print("-" * 60)

    for logo_det, logo_sc, audio_high, audio_rms, label in test_cases:
        time.sleep(0.8)  # simulate tick interval
        state, conf = engine.update(logo_det, logo_sc, audio_high, audio_rms)
        print(f"{label:<40} {state:<6} {conf:.3f}")