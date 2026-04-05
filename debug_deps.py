import sys
import os

def test_opencv():
    print("Testing OpenCV...")
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ OpenCV: Camera opened successfully.")
        cap.release()
    else:
        print("❌ OpenCV: Could not open camera.")

def test_sounddevice():
    print("Testing SoundDevice...")
    import sounddevice as sd
    print(f"✅ SoundDevice: Default devices: {sd.query_devices()}")

def test_pychromecast():
    print("Testing PyChromecast...")
    import pychromecast
    print("✅ PyChromecast: Imported successfully.")

if __name__ == "__main__":
    try:
        test_pychromecast()
        test_sounddevice()
        test_opencv()
        print("\nAll dependencies imported and basic functionality tested.")
    except Exception as e:
        print(f"\n❌ Crash detected: {e}")
        import traceback
        traceback.print_exc()
