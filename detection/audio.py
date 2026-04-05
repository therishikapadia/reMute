import sounddevice as sd
import numpy as np
import scipy.fft
import time

class AudioDetector:
    def __init__(self, samplerate=44100, blocksize=4096, channels=1):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        
        self.rms = 0.0
        self.spectral_centroid = 0.0
        self.is_high_energy = False
        
        # Adaptive thresholding
        self.rms_threshold = 0.05
        self.centroid_threshold = 5000.0
        
        self.stream = None

    def start_stream(self):
        """Open the audio input stream."""
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
        """Stop the audio input stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _audio_callback(self, indata, frames, time, status):
        """Process incoming audio data in real-time."""
        if status:
            print(f"Audio Status: {status}")
        
        # 1. Calculate RMS (Loudness)
        rms = np.sqrt(np.mean(indata**2))
        self.rms = rms
        
        # 2. Calculate Spectral Centroid (Brightness)
        # Apply FFT
        magnitude_spec = np.abs(np.fft.rfft(indata[:, 0]))
        freqs = np.fft.rfftfreq(frames, 1/self.samplerate)
        
        # Avoid division by zero
        sum_mag = np.sum(magnitude_spec)
        if sum_mag > 1e-6:
            centroid = np.sum(freqs * magnitude_spec) / sum_mag
        else:
            centroid = 0.0
        self.spectral_centroid = centroid
        
        # 3. Quick Threshold for Ad (Simple Logic)
        # Ads are usually louder and have more high-frequency content
        self.is_high_energy = (rms > self.rms_threshold)

    def get_features(self):
        """Return the latest audio features."""
        return self.rms, self.spectral_centroid, self.is_high_energy

if __name__ == "__main__":
    # Test audio detection
    detector = AudioDetector()
    if detector.start_stream():
        try:
            while True:
                rms, centroid, high = detector.get_features()
                print(f"RMS: {rms:.4f}, Centroid: {centroid:.1f}, High: {high}", end="\r")
                time.sleep(0.1)
        except KeyboardInterrupt:
            detector.stop_stream()
            print("\nAudio detection stopped.")