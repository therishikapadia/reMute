import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import cv2
import numpy as np
import time

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
    vision = VisionDetector(threshold=0.6)
    audio = AudioDetector()
    audio.centroid_threshold = 2800.0
    fusion = FusionEngine(debounce_seconds=3.0)
    chromecast = ChromecastController(friendly_name="Hall TV")
    return vision, audio, fusion, chromecast

vision, audio, fusion, chromecast = get_detectors()

# --- Sidebar: Settings ---
st.sidebar.header("⚙️ Settings")
st.session_state.chromecast_name = st.sidebar.text_input("Chromecast Name", value=st.session_state.chromecast_name)
st.session_state.auto_mute = st.sidebar.toggle("Auto-Mute Enabled", value=st.session_state.auto_mute)

st.sidebar.divider()
st.sidebar.header("🔍 Vision Calibration")

if st.sidebar.button("📸 Capture Logo Template"):
    if vision.cap is not None and vision.cap.isOpened():
        # Capture a fresh frame and save it as the template
        ret, raw_frame = vision.cap.read()
        if ret:
            small = cv2.resize(raw_frame, (640, 480))
            rx, ry, rw, rh = vision.roi
            rx, ry = max(0, rx), max(0, ry)
            rw = min(rw, 640 - rx)
            rh = min(rh, 480 - ry)
            roi_crop = small[ry:ry + rh, rx:rx + rw]
            os.makedirs("assets", exist_ok=True)
            save_path = "assets/logo_template.png"
            cv2.imwrite(save_path, roi_crop)
            vision.load_template(save_path)
            st.session_state.template_exists = True
            st.sidebar.success("Template captured!")
        else:
            st.sidebar.warning("No frame available yet — wait a moment and try again.")
    else:
        st.sidebar.error("Start monitoring first to capture!")

threshold = st.sidebar.slider("Logo Match Threshold", 0.0, 1.0, vision.threshold, 0.05)
debounce  = st.sidebar.slider("Stability Control (Debounce)", 0.5, 10.0, fusion.debounce_seconds, 0.5)

with st.sidebar.expander("Adjust Detection Box (ROI)"):
    roi_x = st.slider("ROI X Offset", 0, 600, vision.roi[0])
    roi_y = st.slider("ROI Y Offset", 0, 440, vision.roi[1])
    roi_w = st.slider("ROI Width",    50, 640, vision.roi[2])
    roi_h = st.slider("ROI Height",   30, 480, vision.roi[3])

st.sidebar.divider()
st.sidebar.header("🔊 Audio Calibration")
rms_threshold      = st.sidebar.slider("Audio RMS Threshold", 0.0, 0.5, audio.rms_threshold, 0.01)
centroid_threshold = st.sidebar.slider("Centroid Threshold (Hz)", 1000, 8000, int(audio.centroid_threshold), 100)

# --- Apply settings live ---
vision.threshold        = threshold
vision.roi              = (roi_x, roi_y, roi_w, roi_h)
fusion.debounce_seconds = debounce
audio.rms_threshold     = rms_threshold
audio.centroid_threshold= centroid_threshold

if st.session_state.template_exists and vision.template is None:
    vision.load_template("assets/logo_template.png")

if not st.session_state.template_exists:
    st.warning("⚠️ No Logo Template captured. Vision detection will not work. "
               "Start monitoring, then click 'Capture Logo Template' in the sidebar.")

# --- Main Dashboard ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔴 Live Feed")
    video_placeholder  = st.empty()
    status_placeholder = st.empty()

with col2:
    st.subheader("📊 Metrics")
    confidence_placeholder  = st.empty()
    logo_score_placeholder  = st.empty()
    audio_placeholder       = st.empty()

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
        if chromecast.cast:
            chromecast.set_mute(False)   # Unmute on stop
        st.rerun()


# --- Monitoring Loop ---
if st.session_state.monitoring:
    # Start camera
    if vision.cap is None:
        if not vision.start_camera():
            st.error("❌ Failed to open camera. Check that no other app is using it.")
            st.session_state.monitoring = False
            st.stop()

    # Start audio stream
    if audio.stream is None:
        if not audio.start_stream():
            st.warning("⚠️ Could not start audio stream. Audio detection disabled.")

    # Connect Chromecast if auto-mute is on
    if st.session_state.auto_mute and chromecast.cast is None:
        with st.spinner("Connecting to Chromecast..."):
            chromecast.connect()

    try:
        while st.session_state.monitoring:
            # 1. Vision: capture and process a single frame
            frame, logo_score, logo_detected = vision.process_frame()

            # 2. Audio: thread-safe read (audio callback runs in sounddevice thread)
            rms, centroid, audio_high = audio.get_features()

            # 3. Fusion
            state, confidence = fusion.update(logo_detected, logo_score, audio_high, rms)

            # 4. TV Control
            if st.session_state.auto_mute and chromecast.cast:
                chromecast.set_mute(state == DetectionState.AD)

            # 5. UI Update
            if frame is not None:
                color = (0, 0, 255) if state == DetectionState.AD else (0, 255, 0)
                cv2.putText(frame, f"STATE: {state}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
                video_placeholder.image(frame, channels="BGR")

            status_placeholder.markdown(f"### Current State: **{state}**")

            m1, m2 = confidence_placeholder.columns(2)
            m1.metric("Fusion Confidence", f"{confidence:.2f}",
                      help="Confidence of the current detection state (0.0 – 1.0).")
            m2.metric("State", state)

            logo_score_placeholder.metric(
                "Logo Match Score", f"{logo_score:.4f}",
                help=f"Threshold: {threshold:.2f}"
            )
            audio_placeholder.write(f"RMS: {rms:.4f} | Centroid: {centroid:.0f} Hz")

            # Event Log
            if len(st.session_state.logs) == 0 or st.session_state.logs[-1].split(" - ")[1] != state:
                st.session_state.logs.append(f"{time.strftime('%H:%M:%S')} - {state}")
                if len(st.session_state.logs) > 10:
                    st.session_state.logs.pop(0)
            log_placeholder.code("\n".join(reversed(st.session_state.logs)))

            time.sleep(0.1)

    except Exception as e:
        st.error(f"Monitor loop error: {e}")
        st.session_state.monitoring = False
