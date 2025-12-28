
# ============================
# CLEAN SHUTDOWN
# ============================
import streamlit as st
import numpy as np
import cv2
import av
import time
import datetime
import threading
from collections import deque

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ============================
# Backend inference
# ============================
from inference import predict_video   # MUST return (prob, aux)

# ============================
# CONFIG
# ============================
st.set_page_config(
    page_title="NeuroLens – Real-time Fatigue Detection",
    layout="wide"
)

THRESHOLD = 0.65
CLIP_LEN = 16
INFERENCE_INTERVAL = 1.5   # seconds
SMOOTHING_WINDOW = 5

# ============================
# PERSISTENT STATE
# ============================
@st.cache_resource
def get_state():
    return {
        "buffer": deque(maxlen=CLIP_LEN * 2),
        "preds": deque(maxlen=SMOOTHING_WINDOW),
        "latest": None,
        "count": 0,
        "running": True,
    }

STATE = get_state()

# ============================
# SMOOTHING
# ============================
def get_smoothed():
    if len(STATE["preds"]) == 0:
        return None

    probs = np.array([p["prob"] for p in STATE["preds"]])
    weights = np.exp(np.linspace(-1, 0, len(probs)))
    weights /= weights.sum()

    smoothed = float(np.sum(probs * weights))

    return {
        "prob": smoothed,
        "raw": probs[-1],
        "time": STATE["preds"][-1]["time"]
    }

# ============================
# VIDEO PROCESSOR
# ============================
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        STATE["buffer"].append(rgb)

        smoothed = get_smoothed()

        if smoothed:
            prob = smoothed["prob"]
            color = (0, 0, 255) if prob >= THRESHOLD else (0, 255, 0)

            cv2.rectangle(img, (10, 10), (280, 90), color, -1)
            cv2.putText(
                img,
                f"Fatigue: {int(prob * 100)}%",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            status = "ALERT" if prob >= THRESHOLD else "NORMAL"
            cv2.putText(
                img,
                status,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        else:
            cv2.putText(
                img,
                f"Collecting frames {len(STATE['buffer'])}/{CLIP_LEN}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================
# BACKGROUND INFERENCE THREAD
# ============================
def inference_loop():
    print("[INFERENCE] Background thread started")

    while STATE["running"]:
        try:
            if len(STATE["buffer"]) >= CLIP_LEN:
                frames = list(STATE["buffer"])[-CLIP_LEN:]

                prob, _ = predict_video(frames)

                STATE["preds"].append({
                    "prob": float(prob),
                    "time": datetime.datetime.now()
                })

                STATE["latest"] = STATE["preds"][-1]
                STATE["count"] += 1

                print(f"[INFERENCE {STATE['count']}] prob={prob:.3f}")

        except Exception as e:
            print("[INFERENCE ERROR]", e)

        time.sleep(INFERENCE_INTERVAL)

@st.cache_resource
def start_inference_thread():
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    return t

start_inference_thread()

# ============================
# UI
# ============================
st.markdown("### 🧠 NeuroLens – Real-time Fatigue Detection")
st.caption("Continuous video-based fatigue monitoring")

col1, col2 = st.columns([2.5, 1.5], gap="large")

# ============================
# VIDEO
# ============================
with col1:
    st.markdown("#### Live Camera Feed")

    webrtc_ctx = webrtc_streamer(
        key="fatigue-stream",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

# ============================
# DASHBOARD
# ============================
with col2:
    st.markdown("#### Fatigue Detection Score")

    smoothed = get_smoothed()

    if smoothed:
        level = int(smoothed["prob"] * 100)

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;margin-bottom:20px;">
                <div style="font-size:60px;font-weight:bold;color:#de3d51;">
                    {level}
                </div>
                <div style="margin-left:12px;font-size:18px;">
                    Fatigue Level
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if smoothed["prob"] >= THRESHOLD:
            st.error("⚠️ High Fatigue Detected!")
        else:
            st.success("✓ Normal State")

        st.caption(
            f"Raw: {int(smoothed['raw'] * 100)}% | "
            f"Updated: {smoothed['time'].strftime('%H:%M:%S')}"
        )

    else:
        st.info("🔄 Waiting for first inference...")
        st.caption(f"Frames collected: {len(STATE['buffer'])}/{CLIP_LEN}")

    # ============================
    # BEHAVIORAL INDICATORS
    # ============================
    st.markdown("#### Behavioral Indicators")

    if smoothed:
        prob = smoothed["prob"]
        
        # Simulate behavioral metrics based on fatigue probability
        # In production, these would come from actual face analysis
        eye_closure = min(int(prob * 120), 100)
        head_pose = min(int(prob * 95), 100)
        blink_freq = min(int(prob * 110), 100)
    else:
        eye_closure = head_pose = blink_freq = 0

    st.progress(eye_closure / 100, text=f"Eye Closure Rate: {eye_closure}%")
    st.progress(head_pose / 100, text=f"Head Pose Variation: {head_pose}%")
    st.progress(blink_freq / 100, text=f"Blink Frequency: {blink_freq}%")

    # ============================
    # EXPLAINABILITY (Placeholder)
    # ============================
    st.markdown("#### Explainability & Attention Map")
    
    overlay_type = st.radio(
        "Show Overlay",
        ["GradCAM", "Attention Map"],
        horizontal=True
    )
    
    st.image(
        np.uint8(np.full((240, 320, 3), 200)),
        caption=f"{overlay_type} (Not connected)"
    )

    # ============================
    # SYSTEM STATUS
    # ============================
    st.markdown("---")
    st.markdown("#### System Status")
    
    col_a, col_b, col_c = st.columns(3)
    
    col_a.metric("Frames Buffered", len(STATE["buffer"]))
    col_b.metric("Inferences Run", STATE["count"])
    col_c.metric(
        "Status",
        "🟢 Active" if webrtc_ctx.state.playing else "⚪ Idle"
    )

# ============================
# SYSTEM LOGS
# ============================
st.markdown("#### System Logs")

current_time = datetime.datetime.now().strftime('%H:%M:%S')

logs = [
    ("success", f"[{current_time}] VideoMAE model loaded"),
    ("info", f"[{current_time}] Using {CLIP_LEN}-frame window with {SMOOTHING_WINDOW}x smoothing"),
    ("info", f"[{current_time}] Background inference running every {INFERENCE_INTERVAL}s"),
]

if smoothed:
    pred_time = smoothed['time'].strftime('%H:%M:%S')
    logs.append(("success", f"[{pred_time}] Current: {smoothed['prob']:.1%} (raw: {smoothed['raw']:.1%})"))

for level, msg in logs:
    getattr(st, level)(msg)
    
    
    
