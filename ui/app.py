import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import cv2
import numpy as np
import time
import os
import threading
import queue

from detection.vision import VisionDetector
from detection.audio import AudioDetector
from detection.fusion import FusionEngine, DetectionState
from control.chromecast import ChromecastController

# Page config
st.set_page_config(page_title="reMute - IPL Ad Blocker", layout="wide")

st.title("📺 reMute: IPL Advertisement Blocker")

# --- Initialize Session State ---
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'auto_mute' not in st.session_state:
    st.session_state.auto_mute = False
if 'chromecast_name' not in st.session_state:
    st.session_state.chromecast_name = "Hall TV"
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'template_exists' not in st.session_state:
    st.session_state.template_exists = os.path.exists("assets/logo_template.png")

# --- Shared instances (singleton-ish for the session) ---
@st.cache_resource
def get_detectors():
    # Use default values for initialization, will be updated via sidebar
    vision = VisionDetector(threshold=0.7)
    audio = AudioDetector()
    fusion = FusionEngine(debounce_seconds=3.0)
    chromecast = ChromecastController(friendly_name=st.session_state.chromecast_name)
    return vision, audio, fusion, chromecast

vision, audio, fusion, chromecast = get_detectors()

# --- Sidebar: Calibration & Settings ---
st.sidebar.header("⚙️ Settings")
st.session_state.chromecast_name = st.sidebar.text_input("Chromecast Name", value=st.session_state.chromecast_name)
st.session_state.auto_mute = st.sidebar.toggle("Auto-Mute Enabled", value=st.session_state.auto_mute)

st.sidebar.divider()
st.sidebar.header("🔍 Vision Calibration")
if st.sidebar.button("📸 Capture Logo Template"):
    # Capture current frame and save ROI as template
    if vision.cap is not None:
        ret, frame = vision.cap.read()
        if ret:
            x, y, w, h = vision.roi
            # ROI is relative to 640x480 resize in process_frame, so we should match that
            frame_resized = cv2.resize(frame, (640, 480))
            roi_img = frame_resized[y:y+h, x:x+w]
            os.makedirs("assets", exist_ok=True)
            cv2.imwrite("assets/logo_template.png", roi_img)
            vision.load_template("assets/logo_template.png")
            st.session_state.template_exists = True
            st.sidebar.success("Template captured!")
    else:
        st.sidebar.error("Start monitoring first to capture!")

threshold = st.sidebar.slider("Logo Match Threshold", 0.0, 1.0, 0.7, 0.05)
debounce = st.sidebar.slider("Debounce (seconds)", 1.0, 10.0, 3.0, 0.5)

st.sidebar.divider()
st.sidebar.header("🔊 Audio Calibration")
rms_threshold = st.sidebar.slider("Audio RMS Threshold", 0.0, 0.5, 0.05, 0.01)

# Update detector settings from sliders
vision.threshold = threshold
fusion.debounce_seconds = debounce
audio.rms_threshold = rms_threshold

if st.session_state.template_exists and vision.template is None:
    vision.load_template("assets/logo_template.png")

if not st.session_state.template_exists:
    st.warning("⚠️ No Logo Template captured. Vision detection will not work. Please align the camera and click 'Capture Logo Template' in the sidebar.")

# --- Main Dashboard ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔴 Live Feed")
    video_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    st.subheader("📊 Metrics")
    confidence_placeholder = st.empty()
    logo_score_placeholder = st.empty()
    audio_placeholder = st.empty()
    
    st.divider()
    st.subheader("📜 Event Log")
    log_placeholder = st.empty()


# --- Control Buttons ---
if not st.session_state.monitoring:
    if st.button("🚀 Start Monitoring", type="primary", use_container_width=True):
        st.session_state.monitoring = True
        st.rerun()
else:
    if st.button("🛑 Stop Monitoring", type="secondary", use_container_width=True):
        st.session_state.monitoring = False
        vision.stop_camera()
        audio.stop_stream()
        chromecast.set_mute(False) # Unmute on stop
        st.rerun()

# --- Monitoring Loop ---
if st.session_state.monitoring:
    # Initialize detectors if not started
    if vision.cap is None:
        if not vision.start_camera():
            st.error("Failed to start camera.")
            st.session_state.monitoring = False
            st.stop()
    
    if audio.stream is None:
        if not audio.start_stream():
            st.error("Failed to start audio stream.")
    
    # Try connecting to Chromecast if auto-mute is on and not connected
    if st.session_state.auto_mute and chromecast.cast is None:
        with st.spinner("Connecting to Chromecast..."):
            chromecast.connect()

    # Loop
    try:
        while st.session_state.monitoring:
            # 1. Vision Update
            frame, logo_score, logo_detected = vision.process_frame()
            
            # 2. Audio Update
            rms, centroid, audio_high = audio.get_features()
            
            # 3. Fusion Update
            state, confidence = fusion.update(logo_detected, logo_score, audio_high, rms)
            
            # 4. TV Control
            if st.session_state.auto_mute:
                if state == DetectionState.AD:
                    chromecast.set_mute(True)
                else:
                    chromecast.set_mute(False)
            
            # 5. UI Update
            if frame is not None:
                # Add overlay text
                color = (0, 0, 255) if state == DetectionState.AD else (0, 255, 0)
                cv2.putText(frame, f"STATE: {state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                video_placeholder.image(frame, channels="BGR")

            status_placeholder.markdown(f"### Current State: **{state}**")
            confidence_placeholder.metric("Confidence Score", f"{confidence:.2f}")
            logo_score_placeholder.metric("Recent Logo Match Score", f"{logo_score:.4f}", help="Red line in feed indicates ROI")
            audio_placeholder.write(f"RMS: {rms:.4f} | Centroid: {centroid:.0f}")
            
            # Logs
            if len(st.session_state.logs) == 0 or st.session_state.logs[-1].split(" - ")[1] != state:
                timestamp = time.strftime("%H:%M:%S")
                st.session_state.logs.append(f"{timestamp} - {state}")
                if len(st.session_state.logs) > 10:
                    st.session_state.logs.pop(0)
            
            log_placeholder.code("\n".join(reversed(st.session_state.logs)))
            
            time.sleep(0.1) # Frequency control
            
    except Exception as e:
        st.error(f"Error in monitor loop: {e}")
        st.session_state.monitoring = False
