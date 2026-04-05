import subprocess
import os
import sys
import argparse
import time
import logging

# Add project root to sys.path for CLI mode
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("reMute")

def run_cli(args):
    """Run the detection loop in the terminal (Headless Mode)."""
    from detection.vision import VisionDetector
    from detection.audio import AudioDetector
    from detection.fusion import FusionEngine, DetectionState
    from control.chromecast import ChromecastController

    logger.info("🚀 Starting reMute CLI...")
    
    # Parse ROI from CLI args if provided
    roi = tuple(args.roi) if args.roi else (400, 20, 220, 120)
    
    vision = VisionDetector(threshold=args.threshold, roi=roi)
    audio = AudioDetector()
    audio.centroid_threshold = args.centroid
    fusion = FusionEngine(debounce_seconds=args.debounce)
    chromecast = ChromecastController(friendly_name=args.chromecast)

    if not vision.start_camera():
        return
    if not audio.start_stream():
        return
    
    # Optional auto-connect
    if args.auto_connect:
        chromecast.connect()

    logger.info(f"Monitoring started. Chromecast: {args.chromecast}, Target: {args.threshold}")
    
    try:
        while True:
            # 1. Update
            frame, logo_score, logo_detected = vision.process_frame()
            rms, centroid, audio_high = audio.get_features()
            state, confidence = fusion.update(logo_detected, logo_score, audio_high, rms)
            
            # 2. Control
            if chromecast.cast:
                if state == DetectionState.AD:
                    chromecast.set_mute(True)
                else:
                    chromecast.set_mute(False)
            
            # 3. Output
            print(f"\r[ {state} ] Confidence: {confidence:.2f} | Logo: {logo_score:.3f} | RMS: {rms:.4f} | Centroid: {centroid:.0f}Hz   ", end="")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("\n👋 Stopping reMute...")
    finally:
        vision.stop_camera()
        audio.stop_stream()
        if chromecast.cast:
            chromecast.set_mute(False)

def launch_ui(python_exe):
    """Launch the Streamlit dashboard."""
    logger.info("🚀 Launching reMute Dashboard...")
    cmd = [python_exe, "-m", "streamlit", "run", "ui/app.py"]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Error launching Streamlit: {e}")

def main():
    # Determine the venv Python executable
    venv_path = os.path.join(os.path.dirname(__file__), "venv")
    python_exe = sys.executable
    
    if os.path.exists(venv_path):
        macos_linux_py = os.path.join(venv_path, "bin", "python")
        windows_py = os.path.join(venv_path, "Scripts", "python.exe")
        if os.path.exists(macos_linux_py):
            python_exe = macos_linux_py
        elif os.path.exists(windows_py):
            python_exe = windows_py

    # Auto-re-execute with venv python if we're not already using it
    if os.path.abspath(sys.executable) != os.path.abspath(python_exe):
        logger.info(f"🔄 Re-executing with venv: {python_exe}")
        os.execv(python_exe, [python_exe] + sys.argv)

    parser = argparse.ArgumentParser(description="reMute - AI Powered IPL Ad Blocker")
    parser.add_argument("--cli", action="store_true", help="Run in headless CLI mode")
    parser.add_argument("--chromecast", default="Hall TV", help="Name of your Chromecast device")
    parser.add_argument("--auto-connect", action="store_true", help="Automatically connect to Chromecast in CLI mode")
    parser.add_argument("--threshold", type=float, default=0.6, help="Vision matching threshold")
    parser.add_argument("--centroid", type=float, default=2800.0, help="Audio centroid threshold")
    parser.add_argument("--debounce", type=float, default=3.0, help="Stability control seconds")
    parser.add_argument("--roi", type=int, nargs=4, help="Region of Interest (X Y Width Height) in pixels within 640x480")
    
    args = parser.parse_args()

    if args.cli:
        run_cli(args)
    else:
        launch_ui(python_exe)

if __name__ == "__main__":
    main()
