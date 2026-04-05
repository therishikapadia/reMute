import sounddevice as sd
import numpy as np
import threading
import time
import collections


class AudioDetector:
    def __init__(self, samplerate=44100, blocksize=4096, channels=1):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels

        self._lock = threading.Lock()

        # Raw features
        self.rms = 0.0
        self.spectral_centroid = 0.0
        self.is_high_energy = False

        # Adaptive thresholds (start with sensible defaults)
        self.rms_threshold = 0.05
        self.centroid_threshold = 5000.0

        # Rolling baseline for adaptation (last 300 blocks ~ 30s at 0.1s polling)
        self._rms_history = collections.deque(maxlen=300)

        self.stream = None

    # ------------------------------------------------------------------ #
    #  Stream control                                                       #
    # ------------------------------------------------------------------ #

    def start_stream(self):
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=self.channels,
                callback=self._audio_callback
            )
            self.stream.start()
            return True
        except Exception as e:
            print(f"Error: Could not open audio stream: {e}")
            return False

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    # ------------------------------------------------------------------ #
    #  Callback (runs on sounddevice thread)                               #
    # ------------------------------------------------------------------ #

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio Status: {status}")

        audio = indata[:, 0]

        # 1. RMS (loudness)
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # 2. Spectral centroid (brightness)
        magnitude_spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(frames, d=1.0 / self.samplerate)
        sum_mag = np.sum(magnitude_spec)
        centroid = float(np.sum(freqs * magnitude_spec) / sum_mag) if sum_mag > 1e-6 else 0.0

        # 3. Adaptive threshold update
        self._rms_history.append(rms)
        if len(self._rms_history) >= 30:
            baseline = float(np.percentile(self._rms_history, 60))
            alpha = 0.01
            new_threshold = (1 - alpha) * self.rms_threshold + alpha * (baseline * 1.4)
            # Clamp: never let threshold drift below 0.01 or above 0.2
            new_threshold = max(0.01, min(0.2, new_threshold))
        else:
            new_threshold = self.rms_threshold

        is_high = (rms > new_threshold) and (centroid > self.centroid_threshold)

        with self._lock:
            self.rms = rms
            self.spectral_centroid = centroid
            self.rms_threshold = new_threshold
            self.is_high_energy = is_high

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_features(self):
        """Return (rms, spectral_centroid, is_high_energy)."""
        with self._lock:
            return self.rms, self.spectral_centroid, self.is_high_energy

    def get_ad_confidence(self):
        """
        Returns a 0.0 - 1.0 score indicating how likely the audio is an ad.
        Plug this into your combined visual + audio confidence.
        """
        with self._lock:
            rms = self.rms
            centroid = self.spectral_centroid
            threshold_rms = self.rms_threshold
            threshold_centroid = self.centroid_threshold

        rms_score = min(rms / threshold_rms, 1.0) if threshold_rms > 0 else 0.0
        centroid_score = min(centroid / threshold_centroid, 1.0) if threshold_centroid > 0 else 0.0

        return round(0.6 * rms_score + 0.4 * centroid_score, 4)


# ------------------------------------------------------------------ #
#  Quick test                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    detector = AudioDetector()

    if detector.start_stream():
        print("Listening... (Ctrl+C to stop)\n")
        try:
            while True:
                rms, centroid, high = detector.get_features()
                confidence = detector.get_ad_confidence()
                threshold = detector.rms_threshold

                print(
                    f"RMS: {rms:.4f}  "
                    f"Threshold: {threshold:.4f}  "
                    f"Centroid: {centroid:.1f} Hz  "
                    f"HighEnergy: {str(high):<5}  "
                    f"AdConfidence: {confidence:.4f}",
                    end="\r"
                )
                time.sleep(0.1)

        except KeyboardInterrupt:
            detector.stop_stream()
            print("\nStopped.")