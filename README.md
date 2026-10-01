# reMute

**Automatically mute cricket advertisements and restore audio when the match returns.**

reMute is a Python prototype that watches a TV through a webcam, checks for a calibrated broadcast logo or visual marker using OpenCV, and controls a compatible TV through Chromecast. A Streamlit dashboard makes the camera feed, detection thresholds, and state changes visible during testing.

The project connects image processing with control of a physical device: observe the screen, decide whether an advertisement is showing, and update the TV's mute state.

## Features

- Webcam-based visual detection using OpenCV template matching.
- Adjustable region of interest (ROI) and logo-match threshold.
- Capture a reference template directly from the dashboard.
- Debounced transitions between `MATCH` and `AD` to avoid reacting to brief visual changes.
- Automatic TV mute/unmute through `pychromecast`.
- Local mute-state tracking to avoid sending redundant commands.
- Microphone audio analysis with RMS loudness and spectral-centroid metrics.
- Streamlit dashboard with live video, detection scores, settings, and event logs.
- Optional terminal-only monitoring mode.

## How it works

1. **Capture:** Read webcam frames and resize them to 640 × 480 pixels.
2. **Compare:** Crop the configured ROI, convert it to grayscale, and compare it with a reference template using `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)`.
3. **Classify:** A template match above the threshold is treated as `MATCH`; a missing match is treated as `AD`.
4. **Stabilize:** Require the new observation to persist for the configured debounce interval, which defaults to 3 seconds.
5. **Control:** Mute the connected TV during `AD` and unmute it during `MATCH`.

This approach assumes the selected visual marker is present during the match and absent during advertisements. It must be calibrated for the broadcast and camera placement.

### What the audio module currently does

The microphone module calculates RMS loudness and spectral centroid. In the current decision logic, missing-logo observations produce `AD` regardless of audio level; audio changes the displayed confidence value rather than the selected state. The centroid threshold is exposed in the interface but is not used in classification.

The confidence values are heuristic scores, not calibrated probabilities. This implementation uses classical image processing and rules rather than a trained advertisement-classification model.

## Requirements

- Python and `pip`.
- A webcam aimed at the TV and access to an audio-input device.
- A Chromecast-capable TV or device supported by `pychromecast`.
- The computer and TV on the same local network, with device discovery allowed.
- Camera and microphone permissions for the terminal or Python process.

TV compatibility and whether mute affects the intended playback source need to be checked on your device.

## Installation

```bash
git clone https://github.com/therishikapadia/reMute.git
cd reMute
python3 -m venv venv
```

Activate the environment:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows PowerShell**

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If `sounddevice` reports a missing PortAudio library, install PortAudio for your operating system. On macOS with Homebrew:

```bash
brew install portaudio
```

## Run the dashboard

Run commands from the repository root:

```bash
python main.py
```

Alternatively:

```bash
python -m streamlit run ui/app.py
```

### Calibration

1. Keep **Auto-Mute Enabled** off while calibrating.
2. Aim the webcam at the TV and keep its position fixed.
3. Click **Start Monitoring**.
4. While the match is showing, adjust the ROI to tightly cover a stable logo or visual marker.
5. Click **Capture Logo Template**. The selected crop is saved to `assets/logo_template.png`.
6. Adjust the logo-match threshold while observing both match footage and advertisements.
7. Adjust the debounce interval to balance response time against temporary visual changes.
8. Enable automatic muting after confirming the detection behaviour.

The controller defaults to the device name **Hall TV**. In the current dashboard, the Chromecast Name text field does not update the controller instance. For a different TV, change the `friendly_name="Hall TV"` argument inside `get_detectors()` in `ui/app.py` and restart the application.

Normal dashboard stop and CLI shutdown paths attempt to unmute the TV. A crash or connection failure may still require manual unmuting.

## Terminal mode

```bash
python main.py --cli --auto-connect --chromecast "Hall TV" --threshold 0.6 --debounce 3 --roi 400 20 220 120
```

ROI values are `X Y Width Height` in the resized 640 × 480 frame.

**Current CLI limitation:** terminal mode does not load the saved reference template automatically. Before using it for detection, update the `VisionDetector` initialization in `run_cli()` in `main.py`:

```python
vision = VisionDetector(
    template_path="assets/logo_template.png",
    threshold=args.threshold,
    roi=roi,
)
```

Capture the template through the dashboard first, and use the same ROI dimensions in CLI mode. Without a loaded template, the detector cannot recognize the match marker.

Use `Ctrl+C` to stop terminal monitoring.

## Code layout

| File | Responsibility |
| --- | --- |
| `main.py` | Dashboard launcher, CLI arguments, and terminal monitoring loop |
| `detection/vision.py` | Webcam capture, ROI processing, and template matching |
| `detection/audio.py` | Microphone capture and audio feature extraction |
| `detection/fusion.py` | Match/ad state decisions and debouncing |
| `control/chromecast.py` | TV discovery, connection, and mute-state control |
| `ui/app.py` | Streamlit dashboard, calibration controls, and event display |
| `assets/logo_template.png` | Reference image used for visual matching |
| `requirements.txt` | Python dependencies |

## Limitations

- Camera movement, perspective, reflections, lighting, or changes in logo size can reduce template-matching reliability.
- A missing marker is treated as an advertisement, so replays, transitions, or graphics changes can cause false detections.
- No validated accuracy or latency benchmark is included in the repository.
- Audio features do not independently determine the current match/ad state.
- Chromecast connection and mute behaviour depend on the TV and local network.
- Bluetooth remote button mapping is not implemented in this repository.

## Possible next steps

- Load templates automatically in CLI mode.
- Wire the dashboard device-name field into the controller.
- Add a recorded-video evaluation workflow and measure false mutes, missed ads, and transition delay.
- Improve detection robustness with multiple templates or scale-aware matching.
- Add reconnect handling and synchronization with the TV's actual mute state.
- Evaluate whether audio features improve detection before incorporating them into the decision logic.

## Author

[Rishi Kapadia](https://github.com/therishikapadia)
