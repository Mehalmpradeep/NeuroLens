
import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import numpy as np
import cv2
import av
import time
import requests
import base64
import threading
from collections import deque
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import plotly.graph_objects as go
from datetime import datetime, timedelta
import uuid
from fatigue_preprocessor import FatiguePreprocessor

# ============================
# CONFIGURATION
# ============================
st.set_page_config(
    page_title="NeuroLens Fatigue Detection",
    page_icon="🧠",
    layout="wide"
)

BACKEND_URL = "http://localhost:8000"
CLIP_LENGTH = 16
FPS_TARGET = 12
INFERENCE_INTERVAL = 3.0  # Increased for CPU

# Hierarchical thresholds
EYE_CLOSURE_CRITICAL = 3.0
EYE_CLOSURE_WARNING = 2.0
HEAD_PITCH_SEVERE = 0.20
HEAD_PITCH_MODERATE = 0.15

# Calibration parameters
CALIBRATION_FRAMES = 60
CALIBRATION_PERCENTILE = 60
BASELINE_ADAPT_ALPHA = 0.05
MAX_BASELINE_CHANGE = 0.20

# Asymmetric state machine parameters
FACE_LOSS_TIMEOUT = 10.0       # seconds of no face before full reset
DEESCALATION_DELAY = 5.0       # seconds of sustained improvement before de-escalating

# State severity map — higher = worse
STATE_SEVERITY = {
    "Alert":       0,
    "Calibrating": 0,
    "Uncertain":   1,
    "Warning":     2,
    "Drowsy":      3,
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

mp_face_mesh = mp.solutions.face_mesh

# ============================
# SHARED STATE (FIX: Persistent Session ID)
# ============================
class SharedState:
    """Thread-safe shared state for video processing and inference"""
    
    def __init__(self):
        self.lock = threading.Lock()
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=CLIP_LENGTH * 3)
        
        # FIX 1: Generate persistent session_id once
        self.session_id = str(uuid.uuid4())
        print(f"[SESSION] Created persistent session: {self.session_id}")
        
        # Backend predictions
        self.backend_state = "Alert"
        self.raw_probability = 0.0
        self.smoothed_probability = 0.0
        self.backend_confidence = 0.0
        self.last_inference_time = 0
        self.inference_count = 0
        
        # MediaPipe face tracking
        self.eye_aspect_ratio = 0.3
        self.head_pitch = 0.0
        self.blink_count = 0
        self.face_detected = False
        # Face presence tracking
        self.last_face_time = time.time()

        # Eye closure tracking
        self.eye_closed_duration = 0.0
        self._eye_closed_start_time = None
        self._last_eye_closed = False
        
        # Calibration state
        self.baseline_ear = 0.3
        self.calibration_samples = []
        self.is_calibrated = False
        self.calibration_progress = 0.0
        
        # Final decision
        self.final_state = "Alert"
        self.decision_reason = "Initializing"
        self.state_entry_time = time.time()

        # Asymmetric state machine tracking
        self.last_severity_level = 0
        self.deescalation_start_time = None
        self.pending_deescalation_state = None

        # Plot change detection (avoids time-based redraw)
        self.last_plot_inference_count = 0
        self.last_plot_final_state = "Alert"
        
        # Visualization data
        self.timeline_data = {
            "timestamps": deque(maxlen=300),
            "raw_prob": deque(maxlen=300),
            "smoothed_prob": deque(maxlen=300),
            "confidence": deque(maxlen=300),
            "states": deque(maxlen=300),
            "ear_ratio": deque(maxlen=300)
        }
        
        # FIX 3: Thread control
        self._inference_thread_started = False
        self._stop_thread = False
        self.last_backend_ok_time = 0.0
        self.last_backend_error = ""
        self.last_backend_response = {}
        self.prob_buffer = deque(maxlen=3)  # small, fast
        self.last_override_reason = ""




@st.cache_resource
def get_shared_state():
    return SharedState()

STATE = get_shared_state()

# ============================
# VIDEO PROCESSOR
# ============================
class FatigueVideoProcessor(VideoProcessorBase):
    """Process video frames with MediaPipe face tracking"""
    
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1
        )
        self.frame_count = 0
        self.process_interval = 4
        
        # Eye landmark indices
        self.left_eye = [33, 160, 158, 133, 153, 144]
        self.right_eye = [362, 385, 387, 263, 373, 380]

        # ── Adaptive face-crop state ──────────────────────────────────────────
        # Bounding-box history for jitter-smoothing (moving average over last 5)
        self._bbox_history: deque = deque(maxlen=5)   # each entry: (x1,y1,x2,y2) normalised [0,1]
        self._last_valid_bbox = None                   # last successfully detected bbox (normalised)

        # ── Fatigue-region spatial preprocessor ───────────────────────────────
        # Toggle individual stages via the enable_* flags when needed.
        self.preprocessor = FatiguePreprocessor(
            enable_vertical_weight=True,
            enable_eye_emphasis=True,
            enable_lighting_norm=True,
        )
    
    # ------------------------------------------------------------------
    # Adaptive face-crop helpers
    # ------------------------------------------------------------------
    def _landmarks_to_bbox(self, landmarks, img_h: int, img_w: int):
        """
        Compute a padded bounding box (pixel coords) from all 468 face
        landmarks, returned as a normalised tuple (x1n, y1n, x2n, y2n).
        """
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Add 17.5 % padding (midpoint of 15-20 %)
        pad_x = (x_max - x_min) * 0.175
        pad_y = (y_max - y_min) * 0.175
        x1 = max(0.0, x_min - pad_x)
        y1 = max(0.0, y_min - pad_y)
        x2 = min(1.0, x_max + pad_x)
        y2 = min(1.0, y_max + pad_y)
        return (x1, y1, x2, y2)

    def _smooth_bbox(self):
        """Return moving-average bbox over the history buffer (normalised)."""
        if not self._bbox_history:
            return None
        arr = np.array(self._bbox_history)          # shape (N, 4)
        return tuple(arr.mean(axis=0).tolist())     # (x1, y1, x2, y2)

    def _crop_face(self, rgb_img: np.ndarray, bbox_norm) -> np.ndarray:
        """
        Crop `rgb_img` using a normalised bbox, clamp to image bounds,
        then resize to 224×224 RGB.
        """
        h, w = rgb_img.shape[:2]
        x1n, y1n, x2n, y2n = bbox_norm
        x1 = int(np.clip(x1n * w, 0, w - 1))
        y1 = int(np.clip(y1n * h, 0, h - 1))
        x2 = int(np.clip(x2n * w, 1, w))
        y2 = int(np.clip(y2n * h, 1, h))
        crop = rgb_img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = rgb_img                           # safety fallback
        return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)

    def _get_frame_for_buffer(self, rgb_img: np.ndarray, landmarks=None) -> np.ndarray:
        """
        Return a 224×224 RGB frame for the clip buffer.

        * If landmarks are available  → compute bbox, smooth it, crop.
        * If detection failed but we have a previous bbox → reuse it.
        * If no previous bbox at all  → resize full frame.
        """
        if landmarks is not None:
            h, w = rgb_img.shape[:2]
            bbox_norm = self._landmarks_to_bbox(landmarks, h, w)
            self._bbox_history.append(bbox_norm)
            self._last_valid_bbox = self._smooth_bbox()

        if self._last_valid_bbox is not None:
            frame = self._crop_face(rgb_img, self._last_valid_bbox)
            # Apply spatial biasing toward fatigue-relevant regions.
            # Landmarks passed so eye-emphasis can locate the eye band.
            frame = self.preprocessor.process(frame, landmarks)
            return frame

        # Fallback: no bbox ever detected
        frame = cv2.resize(rgb_img, (224, 224), interpolation=cv2.INTER_LINEAR)
        # No landmarks available in this path — preprocessor gracefully skips
        # the eye-emphasis step and applies only the landmark-free stages.
        frame = self.preprocessor.process(frame, None)
        return frame

    # ------------------------------------------------------------------
    def calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR)"""
        coords = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
        
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h = np.linalg.norm(coords[0] - coords[3])
        
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        return ear
    
    def calculate_head_pitch(self, landmarks):
        """Calculate head pitch (nodding)"""
        nose_tip_y = landmarks[1].y
        chin_y = landmarks[152].y
        pitch = abs(chin_y - nose_tip_y) - 0.1
        return max(0.0, pitch)
    
    def recv(self, frame: av.VideoFrame):
        """Process incoming video frame — NO blocking operations allowed here."""
        img = frame.to_ndarray(format="bgr24")

        # Resize for CPU performance
        h, w = img.shape[:2]
        if h > 360:
            scale = 360 / h
            img = cv2.resize(img, (int(w * scale), 360))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── Adaptive face-crop + guarded frame append ─────────────────────────
        # Run face-mesh on every frame so the bbox stays current.
        # We do this BEFORE the process_interval gate so the buffer always
        # receives a cropped (224x224) frame regardless of the EAR/pitch
        # processing cadence.
        _landmarks_for_crop = None
        _crop_results = self.face_mesh.process(rgb)
        if _crop_results.multi_face_landmarks:
            _landmarks_for_crop = _crop_results.multi_face_landmarks[0].landmark

        _cropped_frame = self._get_frame_for_buffer(rgb, _landmarks_for_crop)

        with STATE.lock:
            inference_pending = len(STATE.frame_buffer) >= CLIP_LENGTH
            if not inference_pending:
                # Buffer the 224x224 face-cropped RGB frame
                STATE.frame_buffer.append(_cropped_frame)

        # Process face detection every N frames (lightweight, no I/O)
        # Re-use the results already obtained above for the crop step.
        if self.frame_count % self.process_interval == 0:
            results = _crop_results   # reuse -- avoids a second face-mesh call

            # =========================
            # NO FACE HANDLING (watchdog reset)
            # =========================
            if not results.multi_face_landmarks:
                with STATE.lock:
                    STATE.face_detected = False
                    elapsed = time.time() - STATE.last_face_time

                    if elapsed > FACE_LOSS_TIMEOUT:
                        if STATE.is_calibrated:
                            print(f"[FACE WATCHDOG] No face for >{FACE_LOSS_TIMEOUT}s → FULL RESET")

                        # --- complete state reset ---
                        STATE.is_calibrated = False
                        STATE.calibration_samples.clear()
                        STATE.calibration_progress = 0.0

                        STATE.eye_closed_duration = 0.0
                        STATE._eye_closed_start_time = None
                        STATE._last_eye_closed = False
                        STATE.blink_count = 0

                        STATE.prob_buffer.clear()
                        STATE.raw_probability = 0.0
                        STATE.smoothed_probability = 0.0
                        STATE.backend_confidence = 0.0
                        STATE.frame_buffer.clear()

                        STATE.final_state = "Alert"
                        STATE.decision_reason = "Face lost – recalibrating"
                        STATE.state_entry_time = time.time()

                        STATE.last_severity_level = 0
                        STATE.deescalation_start_time = None
                        STATE.pending_deescalation_state = None

                        print("[FACE WATCHDOG] All state cleared – waiting for face")

                self.frame_count += 1
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # =========================
            # FACE PRESENT (NORMAL FLOW)
            # =========================
            with STATE.lock:
                STATE.face_detected = True
                STATE.last_face_time = time.time()

            lm = results.multi_face_landmarks[0].landmark

            left_ear = self.calculate_ear(lm, self.left_eye)
            right_ear = self.calculate_ear(lm, self.right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            pitch = self.calculate_head_pitch(lm)

            with STATE.lock:
                STATE.eye_aspect_ratio = avg_ear
                STATE.head_pitch = pitch

                # Calibration
                if not STATE.is_calibrated:
                    if avg_ear > 0.15:
                        STATE.calibration_samples.append(avg_ear)

                    STATE.calibration_progress = len(STATE.calibration_samples) / CALIBRATION_FRAMES

                    if len(STATE.calibration_samples) >= CALIBRATION_FRAMES:
                        STATE.baseline_ear = np.percentile(
                            STATE.calibration_samples,
                            CALIBRATION_PERCENTILE
                        )
                        STATE.is_calibrated = True
                        print(f"[CALIBRATED] Baseline EAR: {STATE.baseline_ear:.4f}")

                else:
                    # Slow adaptive update
                    if avg_ear > 0.20:
                        new_baseline = (
                            (1 - BASELINE_ADAPT_ALPHA) * STATE.baseline_ear
                            + BASELINE_ADAPT_ALPHA * avg_ear
                        )

                        change_ratio = abs(new_baseline - STATE.baseline_ear) / STATE.baseline_ear
                        if change_ratio < MAX_BASELINE_CHANGE:
                            STATE.baseline_ear = new_baseline

                # Eye closure tracking
                if STATE.is_calibrated:
                    eye_closed = avg_ear < (STATE.baseline_ear * 0.65)
                    current_time = time.time()

                    if eye_closed:
                        if not STATE._last_eye_closed:
                            STATE.blink_count += 1
                            STATE._eye_closed_start_time = current_time

                        if STATE._eye_closed_start_time:
                            STATE.eye_closed_duration = current_time - STATE._eye_closed_start_time
                    else:
                        STATE.eye_closed_duration = 0.0
                        STATE._eye_closed_start_time = None

                    STATE._last_eye_closed = eye_closed

        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================
# HIERARCHICAL DECISION LOGIC
# ============================
def make_hierarchical_decision():
    """Multi-signal hierarchical decision making"""
    with STATE.lock:
        eye_duration = STATE.eye_closed_duration
        head_pitch = STATE.head_pitch
        backend_state = STATE.backend_state
        smoothed_prob = STATE.smoothed_probability
        ear = STATE.eye_aspect_ratio
        baseline = STATE.baseline_ear
        is_calibrated = STATE.is_calibrated
    
    if not is_calibrated:
        return "Calibrating", "Calibration in progress", False
    
    # Priority 1: Eye closure
    if eye_duration >= EYE_CLOSURE_CRITICAL:
        return "Drowsy", f"Eyes closed {eye_duration:.1f}s (Critical)", False
    
    if eye_duration >= EYE_CLOSURE_WARNING:
        return "Warning", f"Eyes closing {eye_duration:.1f}s", False
    
    # Priority 2: Head pitch
    if head_pitch >= HEAD_PITCH_SEVERE:
        return "Drowsy", f"Severe head nodding ({head_pitch:.2f})", False
    
    if head_pitch >= HEAD_PITCH_MODERATE:
        return "Warning", f"Moderate head nodding ({head_pitch:.2f})", True
    
    # Priority 3: Model prediction
    if backend_state in ["Drowsy", "Warning"]:
        if STATE.backend_confidence >= 0.6:
            return backend_state, f"Model: {smoothed_prob:.1%} (conf: {STATE.backend_confidence:.2f})", True
    
    # Priority 4: EAR ratio
    if baseline > 0:
        ear_ratio = ear / baseline
        if ear_ratio < 0.60:
            return "Warning", f"Low EAR ratio: {ear_ratio:.1%}", True
    
    return "Alert", "All signals normal", False

# ============================
# ASYMMETRIC STATE MACHINE
# ============================
def apply_asymmetric_state_transition(new_state: str, debounce_needed: bool) -> str:
    """
    Immediate escalation, delayed de-escalation.
    All shared-state writes are inside the lock; no sleep, no I/O.
    """
    current_time = time.time()

    with STATE.lock:
        current_state = STATE.final_state
        current_severity = STATE_SEVERITY.get(current_state, 0)
        new_severity    = STATE_SEVERITY.get(new_state, 0)

    # Critical / unconditional signal → always immediate
    if not debounce_needed:
        with STATE.lock:
            STATE.last_severity_level       = new_severity
            STATE.deescalation_start_time   = None
            STATE.pending_deescalation_state = None
        return new_state

    # Same state → clear any pending de-escalation
    if new_state == current_state:
        with STATE.lock:
            STATE.deescalation_start_time   = None
            STATE.pending_deescalation_state = None
        return current_state

    # ESCALATION — immediate
    if new_severity > current_severity:
        print(f"[STATE] ESCALATE {current_state} → {new_state} (immediate)")
        with STATE.lock:
            STATE.last_severity_level       = new_severity
            STATE.deescalation_start_time   = None
            STATE.pending_deescalation_state = None
        return new_state

    # DE-ESCALATION — must sustain for DEESCALATION_DELAY seconds
    if new_severity < current_severity:
        with STATE.lock:
            if STATE.deescalation_start_time is None:
                # Start timer
                STATE.deescalation_start_time    = current_time
                STATE.pending_deescalation_state = new_state
                print(f"[STATE] DEESCALATE PENDING {current_state} → {new_state} "
                      f"(need {DEESCALATION_DELAY}s)")
                return current_state

            elapsed = current_time - STATE.deescalation_start_time

            if STATE.pending_deescalation_state == new_state:
                if elapsed >= DEESCALATION_DELAY:
                    print(f"[STATE] DEESCALATE APPROVED {current_state} → {new_state} "
                          f"({elapsed:.1f}s)")
                    STATE.last_severity_level       = new_severity
                    STATE.deescalation_start_time   = None
                    STATE.pending_deescalation_state = None
                    return new_state
                # Still waiting
                return current_state
            else:
                # Target changed — restart timer
                STATE.deescalation_start_time    = current_time
                STATE.pending_deescalation_state = new_state
                print(f"[STATE] DEESCALATE TARGET CHANGED → {new_state} (timer reset)")
                return current_state

    return current_state
def log_override(backend_state, backend_prob, new_state, final_state, reason):
    if backend_state != final_state:
        print(
            f"[OVERRIDE] Backend={backend_state} "
            f"(p={backend_prob:.2f}) → "
            f"Final={final_state} | Reason={reason}"
        )

# ============================
# INFERENCE THREAD (FIX: Proper timeout & session handling)
# ============================
def inference_thread():
    """Background thread for backend model inference"""
    print(f"[THREAD] Inference thread started for session {STATE.session_id[:8]}")
    
    while not STATE._stop_thread:
        try:
            current_time = time.time()
            
            # Check if enough time passed
            if current_time - STATE.last_inference_time < INFERENCE_INTERVAL:
                time.sleep(0.2)
                continue
            
            # Wait for calibration
            if not STATE.is_calibrated:
                time.sleep(0.5)
                continue
            
            # Get frames for inference
            with STATE.lock:
                if len(STATE.frame_buffer) < CLIP_LENGTH:
                    time.sleep(0.2)
                    continue

                # ALWAYS take freshest frames
                frames_to_send = list(STATE.frame_buffer)[-CLIP_LENGTH:]

                # 🔥 CRITICAL: clear buffer so old video is never inferred later
                STATE.frame_buffer.clear()

                session_id = STATE.session_id
            
            # Encode frames
            encoded = []
            for frame in frames_to_send:
                resized = cv2.resize(frame, (224,224))
                _, buffer = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 50]
                )
                encoded.append(base64.b64encode(buffer).decode())
            
            # FIX 2: Increased timeout to 30s for CPU inference
            payload = {
                "frames": encoded,
                "session_id": session_id
            }
            
            print(f"[INFERENCE] Calling backend with session {session_id[:8]}...")
            
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json=payload,
                timeout=30  # INCREASED FROM 15s TO 30s for CPU
            )
            
            if response.status_code == 200:
                result = response.json()
                with STATE.lock:
                    STATE.last_backend_ok_time = time.time()
                    STATE.last_backend_error = ""
                    STATE.last_backend_response = result

                with STATE.lock:
                    # Don't update session_id - keep it persistent!
                    STATE.backend_state = result["state"]
                    raw = result["raw_probability"]
                    STATE.raw_probability = raw
                    STATE.prob_buffer.append(raw)
                    STATE.smoothed_probability = float(np.mean(STATE.prob_buffer))
                    STATE.backend_confidence = result["confidence"]
                    STATE.last_inference_time = current_time
                    STATE.inference_count += 1
                
                # Make hierarchical decision
                new_state, reason, debounce = make_hierarchical_decision()
                final_state = apply_asymmetric_state_transition(new_state, debounce)

                # LOG frontend override
                log_override(
                    backend_state=result["state"],
                    backend_prob=result["raw_probability"],
                    new_state=new_state,
                    final_state=final_state,
                    reason=reason
                )

                # Atomic state + timeline update (single lock acquire)
                with STATE.lock:
                    if result["state"] != final_state:
                        STATE.last_override_reason = (
                            f"{result['state']} → {final_state} | {reason}"
                        )

                    if final_state != STATE.final_state:
                        STATE.final_state   = final_state
                        STATE.state_entry_time = current_time

                    STATE.decision_reason = reason

                    # Timeline data
                    STATE.timeline_data["timestamps"].append(datetime.now())
                    STATE.timeline_data["raw_prob"].append(STATE.raw_probability)
                    STATE.timeline_data["smoothed_prob"].append(STATE.smoothed_probability)
                    STATE.timeline_data["confidence"].append(STATE.backend_confidence)
                    STATE.timeline_data["states"].append(final_state)
                    ear_ratio = (STATE.eye_aspect_ratio / STATE.baseline_ear
                                 if STATE.baseline_ear > 0 else 1.0)
                    STATE.timeline_data["ear_ratio"].append(ear_ratio)

                    # Sentinels for plot-change detection (used by render_dashboard)
                    STATE.last_plot_inference_count = STATE.inference_count
                    STATE.last_plot_final_state     = final_state
                
                print(f"[INFERENCE #{STATE.inference_count}] "
                      f"Final: {final_state} | "
                      f"Backend: {result['state']} | "
                      f"Raw: {result['raw_probability']:.3f} | "
                      f"Smoothed: {result['smoothed_probability']:.3f}")
            else:
                with STATE.lock:
                    STATE.last_backend_error = f"{response.status_code}: {response.text}"

        except requests.exceptions.Timeout:
            with STATE.lock:
                STATE.last_backend_error = "Timeout calling backend /predict (30s)"
        except Exception as e:
            with STATE.lock:
                STATE.last_backend_error = str(e)

        
        time.sleep(0.3)
    
    print("[THREAD] Inference thread stopped gracefully")

# FIX 3: Start thread only once per session
def ensure_inference_thread_started():
    with STATE.lock:
        if STATE._inference_thread_started:
            return
        STATE._inference_thread_started = True
        STATE._stop_thread = False

    thread = threading.Thread(target=inference_thread, daemon=True)
    thread.start()
    print("[MAIN] Inference thread initialized (singleton)")

ensure_inference_thread_started()

# ============================
# UI LAYOUT
# ============================
st.title("🧠 NeuroLens – Real-Time Fatigue Detection")
st.markdown("**Production-ready CPU-based system with temporal smoothing and personalized calibration**")

col_video, col_dashboard = st.columns([2.5, 1.5])

with col_video:
    st.markdown("### 📹 Live Video Feed")

    ctx = webrtc_streamer(
    key="live-fatigue",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=FatigueVideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 360},
            "frameRate": {"ideal": 10, "max": 10},
        },
        "audio": False,
    },
    async_processing=True,   # 🔑 VERY IMPORTANT
)

# --- FIX: Immediate reset when WebRTC stream stops ---
# This fires on manual camera stop/restart without waiting for FACE_LOSS_TIMEOUT.
if ctx.state.playing is False:
    with STATE.lock:
        was_calibrated = STATE.is_calibrated

        STATE.is_calibrated          = False
        STATE.calibration_samples.clear()
        STATE.calibration_progress   = 0.0
        STATE.baseline_ear           = 0.3

        STATE.eye_closed_duration    = 0.0
        STATE._eye_closed_start_time = None
        STATE._last_eye_closed       = False
        STATE.blink_count            = 0

        STATE.prob_buffer.clear()
        STATE.raw_probability        = 0.0
        STATE.smoothed_probability   = 0.0
        STATE.backend_confidence     = 0.0
        STATE.frame_buffer.clear()

        STATE.final_state            = "Alert"
        STATE.decision_reason        = "Stream stopped"
        STATE.state_entry_time       = time.time()

        STATE.last_severity_level        = 0
        STATE.deescalation_start_time    = None
        STATE.pending_deescalation_state = None

        STATE.timeline_data = {k: deque(maxlen=300) for k in STATE.timeline_data}

        if was_calibrated:
            print("[STREAM] WebRTC stopped → full state reset")

# ============================
# DASHBOARD (AUTO-UPDATE)
# ============================
@st.fragment(run_every=3.0)
def render_dashboard(calib_slot,prob_slot,decision_slot):
    """Real-time dashboard — stable layout, event-driven plot updates."""

    # --- Snapshot all shared state under a single lock acquire ---
    with STATE.lock:
        final_state          = STATE.final_state
        decision_reason      = STATE.decision_reason
        is_calibrated        = STATE.is_calibrated
        calibration_progress = STATE.calibration_progress
        face_detected        = STATE.face_detected

        backend_state  = STATE.backend_state
        raw_prob       = STATE.raw_probability
        smoothed_prob  = STATE.smoothed_probability
        confidence     = STATE.backend_confidence

        eye_duration = STATE.eye_closed_duration
        head_pitch   = STATE.head_pitch
        blink_count  = STATE.blink_count
        ear          = STATE.eye_aspect_ratio
        baseline_ear = STATE.baseline_ear

        timestamps    = list(STATE.timeline_data["timestamps"])
        raw_probs     = list(STATE.timeline_data["raw_prob"])
        smoothed_probs = list(STATE.timeline_data["smoothed_prob"])
        confidences   = list(STATE.timeline_data["confidence"])
        states        = list(STATE.timeline_data["states"])

        #time_in_state      = time.time() - STATE.state_entry_time
        time_in_state = int(time.time() - STATE.state_entry_time)
        session_id_display = STATE.session_id[:8]
        inference_count    = STATE.inference_count

        last_ok   = STATE.last_backend_ok_time
        last_err  = STATE.last_backend_error
        last_resp = STATE.last_backend_response

        pending_deescalate = STATE.pending_deescalation_state
        deescalate_time    = STATE.deescalation_start_time

    # --- Backend status line (always rendered, never causes layout shift) ---
    #st.caption(f"Backend last OK: {int(time.time() - last_ok)}s ago")
    last_ok_secs = (int(time.time() - last_ok) // 5) * 5
    st.caption(f"Backend last OK: {last_ok_secs}s ago")
    if last_err:
        st.error(f"Backend error: {last_err}")

    # ============================
    # CALIBRATION STATUS — fixed container
    # ============================
    #calib_slot = st.empty()
    if not is_calibrated:
        with calib_slot.container():
            st.info("🔄 **Calibrating eye baseline…** Look at the camera normally.")
            st.write("Face detected:", "✅" if face_detected else "❌")
            st.write("Samples:", len(STATE.calibration_samples))
            st.write("EAR:", float(ear))
            st.progress(calibration_progress,
                        text=f"Progress: {int(calibration_progress * 100)}%")
        return
    else:
        calib_slot.empty()   # collapse slot so it takes no space

    # ============================
    # STATUS CARD — fixed container
    # ============================
    state_colors = {
        "Alert":       "#2ecc71",
        "Warning":     "#f39c12",
        "Uncertain":   "#3498db",
        "Drowsy":      "#e74c3c",
        "Calibrating": "#95a5a6",
    }
    state_emojis = {
        "Alert":       "✅",
        "Warning":     "⚠️",
        "Uncertain":   "❓",
        "Drowsy":      "🚨",
        "Calibrating": "🔄",
    }

    color = state_colors.get(final_state, "#95a5a6")
    emoji = state_emojis.get(final_state, "⚪")

    deescalate_html = ""
    if pending_deescalate and deescalate_time:
        remaining = int(DEESCALATION_DELAY - (time.time() - deescalate_time))
        if remaining > 0:
            deescalate_html = (
                f"<div style='font-size:12px;color:rgba(255,255,255,0.65);"
                f"margin-top:4px;'>⏳ Improving → {pending_deescalate} "
                f"in {remaining:.1f}s</div>"
            )

    st.markdown(f"""
        <div style="background:{color};padding:25px;border-radius:12px;
                    text-align:center;margin-bottom:20px;">
            <div style="font-size:56px;margin-bottom:10px;">{emoji}</div>
            <div style="font-size:42px;color:white;font-weight:bold;
                        margin-bottom:8px;">{final_state.upper()}</div>
            <div style="font-size:16px;color:rgba(255,255,255,0.9);">{decision_reason}</div>
            <div style="font-size:14px;color:rgba(255,255,255,0.7);
                        #margin-top:8px;">Time in state: {time_in_state:.1f} sec</div>
                        margin-top:8px;">Time in state: {time_in_state}s</div>
            <div style="font-size:12px;color:rgba(255,255,255,0.6);
                        margin-top:4px;">Session: {session_id_display}</div>
            {deescalate_html}
        </div>
    """, unsafe_allow_html=True)

    # ============================
    # KEY METRICS — fixed 4-column row
    # ============================
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Backend State", backend_state)
    col2.metric("Model Confidence", f"{confidence:.1%}")
    col3.metric("Blinks", blink_count)
    col4.metric("Eye Closed", f"{eye_duration:.1f}s")

    # ============================
    # PLOTS — event-driven, fixed placeholders
    # Redraw only when inference_count or final_state changes.
    # ============================
    st.markdown("### 📈 Fatigue Probability Timeline")
    #prob_slot = st.empty()

    st.markdown("### 🎯 Decision Timeline")
    #decision_slot = st.empty()

    # Determine whether plots need an update this fragment run
    prev_count = st.session_state.get("_plot_inference_count", -1)
    prev_state = st.session_state.get("_plot_final_state", "")
    plots_stale = (inference_count != prev_count) or (final_state != prev_state)

    if plots_stale and len(timestamps) > 0:
        # --- Probability timeline ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps, y=raw_probs,
            name="Raw Probability",
            line=dict(color="#3498db", width=1), opacity=0.4
        ))
        fig.add_trace(go.Scatter(
            x=timestamps, y=smoothed_probs,
            name="Smoothed (Temporal)",
            line=dict(color="#e74c3c", width=3),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.1)"
        ))
        fig.add_trace(go.Scatter(
            x=timestamps, y=confidences,
            name="Confidence",
            line=dict(color="#2ecc71", width=2, dash="dot"), opacity=0.6
        ))
        fig.add_hline(y=0.4, line_dash="dash", line_color="orange",
                      opacity=0.3, annotation_text="Alert/Uncertain")
        fig.add_hline(y=0.7, line_dash="dash", line_color="red",
                      opacity=0.3, annotation_text="Drowsy")
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time",
            yaxis_title="Probability / Confidence",
            yaxis_range=[0, 1],
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            hovermode="x unified",
        )
        #prob_slot.plotly_chart(fig,width='stretch', key="prob_chart")
        prob_slot.plotly_chart(fig, use_container_width=True, key="prob_chart")

        # --- Decision timeline ---
        state_map = {"Alert": 0, "Uncertain": 1, "Warning": 2,
                     "Drowsy": 3, "Calibrating": 0}
        state_values = [state_map.get(s, 0) for s in states]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=timestamps, y=state_values,
            name="State",
            mode="lines+markers",
            line=dict(color="#9b59b6", width=3),
            marker=dict(size=8),
            fill="tozeroy", fillcolor="rgba(155,89,182,0.2)"
        ))
        fig2.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time",
            yaxis_title="State",
            yaxis=dict(
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["Alert", "Uncertain", "Warning", "Drowsy"]
            ),
            hovermode="x unified",
        )
        #decision_slot.plotly_chart(fig2,width="stretch",key="decision_chart")
        decision_slot.plotly_chart(fig2, use_container_width=True, key="decision_chart")

        # Update sentinels so we don't redraw next run unless data changes
        st.session_state["_plot_inference_count"] = inference_count
        st.session_state["_plot_final_state"]     = final_state

    elif len(timestamps) == 0:
        prob_slot.info("Collecting data…")

    # ============================
    # SIGNAL INDICATORS — always rendered
    # ============================
    st.markdown("### 🔍 Signal Indicators")
    ear_ratio       = ear / baseline_ear if baseline_ear > 0 else 1.0
    eye_closure_pct = int((1.0 - np.clip(ear_ratio, 0, 1)) * 100)

    col_a, col_b = st.columns(2)
    with col_a:
        st.progress(eye_closure_pct / 100,
                    text=f"Eye Closure: {eye_closure_pct}% (EAR: {ear:.3f})")
    with col_b:
        head_pitch_pct = min(100, int(head_pitch * 300))
        st.progress(head_pitch_pct / 100,
                    text=f"Head Pitch: {head_pitch_pct}% ({head_pitch:.3f})")

    # ============================
    # ADVANCED DIAGNOSTICS — isolated expander, never shifts layout
    # ============================
    with st.expander("🔧 Advanced Diagnostics", expanded=False):
        diag_col1, diag_col2, diag_col3 = st.columns(3)

        diag_col1.metric("Inference Count", inference_count)
        diag_col1.metric("Baseline EAR",    f"{baseline_ear:.4f}")
        diag_col1.metric("Current EAR",     f"{ear:.4f}")

        diag_col2.metric("Raw Probability",  f"{raw_prob:.3f}")
        diag_col2.metric("Smoothed Prob",    f"{smoothed_prob:.3f}")
        diag_col2.metric("Backend State",    backend_state)

        diag_col3.metric("EAR Ratio",  f"{ear_ratio:.2f}x")
        diag_col3.metric("Head Pitch", f"{head_pitch:.3f}")
        diag_col3.metric("Session ID", session_id_display + "…")

        st.markdown("---")
        st.markdown("**Decision Hierarchy:**")
        st.markdown("""
        1. **Eye Closure** (≥3 s) → Drowsy — *immediate*
        2. **Head Pitch** (severe) → Drowsy / Warning — *immediate*
        3. **Model Prediction** (confident) → Uncertain / Drowsy — *delayed de-escalation*
        4. **EAR Ratio** (<60 % baseline) → Warning — *delayed de-escalation*
        """)

        st.markdown("---")
        st.markdown("### 🧾 Backend Debug")
        st.write("Last OK:", int(time.time() - last_ok), "sec ago")
        st.metric("Last Override", STATE.last_override_reason or "—")

        if pending_deescalate:
            st.info(f"⏳ De-escalation pending: {final_state} → {pending_deescalate}")

        if last_err:
            st.error(last_err)
        st.json({
            "state":              (last_resp or {}).get("state"),
            "raw_probability":    (last_resp or {}).get("raw_probability"),
            "smoothed_probability": (last_resp or {}).get("smoothed_probability"),
            "confidence":         (last_resp or {}).get("confidence"),
            "buffer_size":        (last_resp or {}).get("buffer_size"),
        })
with col_dashboard:
    _calib_slot    = st.empty()
    _prob_slot     = st.empty()
    _decision_slot = st.empty()
    render_dashboard(_calib_slot, _prob_slot, _decision_slot)

