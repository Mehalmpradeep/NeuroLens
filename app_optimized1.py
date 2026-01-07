import streamlit as st
import numpy as np
import cv2
import av
import time
import datetime
import requests
import base64
import threading
from collections import deque
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ============================
# CONFIG
# ============================
st.set_page_config(page_title="NeuroLens AI", layout="wide")

BACKEND_URL = "http://localhost:8000"
CLIP_LEN = 16
STRIDE = 3  
BUFFER_MAX = CLIP_LEN * STRIDE
THRESHOLD = 0.70

mp_face_mesh = mp.solutions.face_mesh

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# ============================
# SHARED STATE
# ============================
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.buffer = deque(maxlen=BUFFER_MAX)
        self.predictions = deque(maxlen=12) 
        self.eye_dist = 0.02
        self.head_x_offset = 0.0
        self.blink_count = 0
        self.last_eye_closed = False
        self.inference_count = 0
        self.baseline_eye = 0.02
        self.calibration_samples = []
        self.is_calibrated = False

    def update_telemetry(self, eye_dist, head_x, eye_closed):
        with self.lock:
            self.eye_dist = eye_dist
            self.head_x_offset = head_x
            if not self.is_calibrated:
                self.calibration_samples.append(eye_dist)
                if len(self.calibration_samples) >= 50:
                    self.baseline_eye = np.percentile(self.calibration_samples, 75)
                    self.is_calibrated = True
            if eye_closed and not self.last_eye_closed:
                self.blink_count += 1
            self.last_eye_closed = eye_closed

@st.cache_resource
def get_state():
    return SharedState()

STATE_OBJ = get_state()

# ============================
# VIDEO PROCESSOR
# ============================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with STATE_OBJ.lock:
            STATE_OBJ.buffer.append(rgb)

        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            eye_dist = abs(lm[159].y - lm[145].y)
            head_x = abs(lm[1].x - 0.5)
            eye_closed = eye_dist < (STATE_OBJ.baseline_eye * 0.65)
            STATE_OBJ.update_telemetry(eye_dist, head_x, eye_closed)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================
# BACKGROUND INFERENCE
# ============================
def inference_loop():
    print("[SYSTEM] Background Inference Loop Started")
    while True:
        try:
            frames_to_send = []
            with STATE_OBJ.lock:
                if STATE_OBJ.is_calibrated and len(STATE_OBJ.buffer) >= BUFFER_MAX:
                    full_list = list(STATE_OBJ.buffer)
                    frames_to_send = full_list[::STRIDE][-CLIP_LEN:]

            if len(frames_to_send) == CLIP_LEN:
                encoded = [base64.b64encode(cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))[1]).decode() for f in frames_to_send]
                r = requests.post(f"{BACKEND_URL}/predict", json=encoded, timeout=5)
                
                if r.status_code == 200:
                    prob = r.json().get("fatigue_prob", 0.0)
                    with STATE_OBJ.lock:
                        STATE_OBJ.predictions.append(prob)
                        STATE_OBJ.inference_count += 1
                    print(f"[INFERENCE #{STATE_OBJ.inference_count}] Fatigue: {prob:.2f}")
        except Exception as e:
            print(f"[THREAD ERROR] {e}")
        time.sleep(1.2)

if "thread_started" not in st.session_state:
    threading.Thread(target=inference_loop, daemon=True).start()
    st.session_state.thread_started = True

# ============================
# UI LAYOUT
# ============================
st.markdown("### 🧠 NeuroLens – Real-time Fatigue Detection")

col1, col2 = st.columns([2.5, 1.5], gap="large")

with col1:
    webrtc_streamer(
        key="fatigue-stream-final-v1",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration=RTC_CONFIG,
    )

# THE FIX: This fragment refreshes the UI every 0.5s WITHOUT duplicating
@st.fragment(run_every=0.5)
def render_dashboard():
    with STATE_OBJ.lock:
        if not STATE_OBJ.is_calibrated or len(STATE_OBJ.predictions) < 3:
            level, color, status_msg, eye_val = 0, "#2ecc71", "🔄 Calibrating...", 0
            is_fatigued = False
        else:
            avg_prob = np.mean(list(STATE_OBJ.predictions))
            eye_ratio = STATE_OBJ.eye_dist / STATE_OBJ.baseline_eye
            if eye_ratio > 0.82: avg_prob *= 0.4 # Fluctuating fix (Gating)
            
            level = int(avg_prob * 100)
            is_fatigued = avg_prob >= THRESHOLD
            status_msg = "⚠️ ALERT: Fatigue Detected" if is_fatigued else "✓ Status: Normal"
            color = "#de3d51" if is_fatigued else "#2ecc71"
            eye_val = int(np.clip((STATE_OBJ.baseline_eye - STATE_OBJ.eye_dist) / (STATE_OBJ.baseline_eye * 0.5), 0, 1) * 100)

    st.markdown("#### Fatigue Detection Score")
    st.markdown(f"""
        <div style="display:flex;align-items:center;margin-bottom:20px;">
            <div style="font-size:60px;font-weight:bold;color:{color};">{level}%</div>
            <div style="margin-left:12px;font-size:18px;">Fatigue Level</div>
        </div>
    """, unsafe_allow_html=True)

    if is_fatigued: st.error(status_msg)
    else: st.success(status_msg)

    st.markdown("#### Behavioral Indicators")
    st.progress(eye_val / 100, text=f"Eye Closure: {eye_val}%")
    st.progress(min(1.0, STATE_OBJ.head_x_offset * 4), text=f"Head Movement")
    st.metric("Total Blinks Detected", STATE_OBJ.blink_count)

    st.markdown("---")
    c_a, c_b = st.columns(2)
    c_a.metric("Calibration", "Ready" if STATE_OBJ.is_calibrated else "Wait...")
    c_b.metric("Inferences", STATE_OBJ.inference_count)

with col2:
    render_dashboard()