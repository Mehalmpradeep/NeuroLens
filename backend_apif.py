"""
Production FastAPI Backend for VideoMAE Fatigue Detection
- Stateless inference with per-session state
- Temporal smoothing with weighted moving average
- Three-zone classification (Alert/Uncertain/Drowsy)
- CPU-optimized sliding window inference
"""
import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import base64
import time
from typing import List, Dict, Optional
from pydantic import BaseModel
from collections import deque
import uuid

# Import inference module
from inferencef import predict_fatigue, warmup, device
#from neurolens_production_metrics import MetricsMiddleware

# ============================
# CONFIGURATION
# ============================
ALERT_THRESHOLD = 0.5
DROWSY_THRESHOLD = 0.8
TEMPORAL_WINDOW = 5
SUSTAINED_DROWSY_COUNT = 3  # ≥3 of last 5 must be Drowsy
UNCERTAIN_ESCALATION_TIME = 15.0  # seconds

# ============================
# FASTAPI APP
# ============================
app = FastAPI(
    title="NeuroLens Fatigue Detection API",
    version="2.0.0",
    description="Production VideoMAE fatigue classifier with temporal smoothing"
)
#app.add_middleware(MetricsMiddleware, output_dir="./metrics_logs")
#from neurolens_production_metrics import MetricsMiddleware
#app.add_middleware(MetricsMiddleware, output_dir="./metrics_logs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ============================
# SESSION STATE MANAGEMENT
# ============================
'''class SessionState:
    """Per-session temporal state for smoothing and tracking"""
    
    def __init__(self):
        # Temporal buffer for smoothing
        self.prediction_buffer = deque(maxlen=TEMPORAL_WINDOW)
        self.timestamp_buffer = deque(maxlen=TEMPORAL_WINDOW)
        
        # State tracking
        self.current_state = "Alert"
        self.state_entry_time = time.time()
        self.last_inference_time = 0
        
        # Uncertain tracking
        self.uncertain_start_time = None
        self.uncertain_duration = 0.0
        
        # Statistics
        self.inference_count = 0
        self.state_transitions = []
    
    def add_prediction(self, probability: float):
        """Add new prediction with timestamp"""
        current_time = time.time()
        self.prediction_buffer.append(probability)
        self.timestamp_buffer.append(current_time)
        self.last_inference_time = current_time
        self.inference_count += 1
    
    def get_smoothed_probability(self) -> float:
        """
        Calculate weighted moving average
        Recent predictions weighted more heavily
        """
        if not self.prediction_buffer:
            return 0.0
        
        # Exponential weights: more recent = higher weight
        weights = np.exp(np.linspace(-1, 0, len(self.prediction_buffer)))
        weights /= weights.sum()
        
        smoothed = np.average(self.prediction_buffer, weights=weights)
        return float(smoothed)
    
    def classify_state(self, smoothed_prob: float) -> str:
        """
        Three-zone classification with temporal logic
        """
        # Count Drowsy predictions in buffer
        drowsy_count = sum(
            1 for p in self.prediction_buffer 
            if p >= DROWSY_THRESHOLD
        )
        
        # Sustained drowsiness check (≥3 of last 5)
        if len(self.prediction_buffer) >= 3 and drowsy_count >= SUSTAINED_DROWSY_COUNT:
            return "Drowsy"
        
        # Zone-based classification for non-sustained cases
        if smoothed_prob >= DROWSY_THRESHOLD:
            return "Uncertain"  # Not sustained yet
        elif smoothed_prob >= ALERT_THRESHOLD:
            return "Uncertain"
        else:
            return "Alert"
    
    def update_state(self, new_state: str):
        """Update state with transition tracking"""
        if new_state != self.current_state:
            transition = {
                "from": self.current_state,
                "to": new_state,
                "timestamp": time.time(),
                "duration": time.time() - self.state_entry_time
            }
            self.state_transitions.append(transition)
            
            self.current_state = new_state
            self.state_entry_time = time.time()
            
            # Reset uncertain tracking on state change
            if new_state != "Uncertain":
                self.uncertain_start_time = None
                self.uncertain_duration = 0.0
    
    def handle_uncertain_escalation(self) -> str:
        """
        Escalate Uncertain to Warning if persists too long
        """
        if self.current_state == "Uncertain":
            if self.uncertain_start_time is None:
                self.uncertain_start_time = time.time()
            
            self.uncertain_duration = time.time() - self.uncertain_start_time
            
            if self.uncertain_duration >= UNCERTAIN_ESCALATION_TIME:
                return "Warning"
        else:
            self.uncertain_start_time = None
            self.uncertain_duration = 0.0
        
        return self.current_state
    
    def get_confidence(self) -> float:
        """
        Calculate confidence based on buffer consistency
        High confidence = predictions agree with each other
        """
        if len(self.prediction_buffer) < 2:
            return 0.5
        
        # Calculate standard deviation (lower = higher confidence)
        std = np.std(self.prediction_buffer)
        # Convert to confidence score (0-1)
        confidence = 1.0 / (1.0 + std * 2)
        return float(np.clip(confidence, 0.0, 1.0))

# Session storage
sessions: Dict[str, SessionState] = {}

def get_or_create_session(session_id: str = None) -> tuple[str, SessionState]:
    """Get existing session or create new one"""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    
    # Create new session
    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = SessionState()
    return new_id, sessions[new_id]
'''
# ============================
# REQUEST/RESPONSE MODELS
# ============================
class PredictionRequest(BaseModel):
    frames: List[str]
    session_id: Optional[str] = None  # Properly optional

class PredictionResponse(BaseModel):
    # Core outputs
    state: str  # Alert / Uncertain / Drowsy / Warning
    raw_probability: float
    smoothed_probability: float
    confidence: float
    
    # Session info
    session_id: str
    time_in_state: float
    
    # Debug info
    buffer_size: int
    inference_count: int
    uncertain_duration: float

class HealthResponse(BaseModel):
    status: str
    device: str
    model: str
    active_sessions: int

# ============================
# HELPER FUNCTIONS
# ============================
def decode_base64_frame(frame_b64: str) -> np.ndarray:
    """Decode base64 string to RGB numpy array"""
    try:
        img_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame decode error: {str(e)}")

# ============================
# ENDPOINTS
# ============================
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Fatigue prediction with temporal smoothing
    
    Input: 
        - frames: List of 16 base64-encoded frames
        - session_id: Optional session identifier
    
    Output: 
        - Three-zone state (Alert/Uncertain/Drowsy/Warning)
        - Raw and smoothed probabilities
        - Confidence score
    """
    try:
        print(f"[DEBUG] Received request with {len(request.frames)} frames, session_id: {request.session_id}")
        
        # Validate input
        if len(request.frames) < 16:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 16 frames, got {len(request.frames)}"
            )
        
        # Get or create session
        frames = request.frames[-16:]

        decoded_frames = [decode_base64_frame(f) for f in frames]

        raw_probability, _, debug = predict_fatigue(decoded_frames)

        # Stateless classification
        if raw_probability >= DROWSY_THRESHOLD:
            state = "Drowsy"
        elif raw_probability >= ALERT_THRESHOLD:
            state = "Uncertain"
        else:
            state = "Alert"
        print(
            f"[API] State={state} "
            f"Raw={raw_probability:.3f}"
        )


        return PredictionResponse(
            state=state,
            raw_probability=raw_probability,
            smoothed_probability=raw_probability,  # frontend handles smoothing
            confidence=0.5,  # frontend controls this
            session_id=request.session_id or "stateless",
            time_in_state=0.0,
            buffer_size=0,
            inference_count=0,
            uncertain_duration=0.0
        )

       
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        device=device.type,
        model="Neurolens/NeuroLens-VideoMAE-Fatigue-v1",
        active_sessions=0
    )

'''@app.post("/reset")
async def reset_session(session_id: str = None):
    """
    Reset session state
    If session_id provided, reset that session
    If not provided, clear all sessions
    """
    if session_id:
        if session_id in sessions:
            del sessions[session_id]
            return {
                "status": "reset",
                "message": f"Session {session_id} cleared"
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        count = len(sessions)
        sessions.clear()
        return {
            "status": "reset",
            "message": f"All {count} sessions cleared"
        }

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get detailed session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return {
        "session_id": session_id,
        "current_state": session.current_state,
        "time_in_state": time.time() - session.state_entry_time,
        "inference_count": session.inference_count,
        "buffer": list(session.prediction_buffer),
        "transitions": session.state_transitions[-10:],  # Last 10
        "confidence": session.get_confidence()
    }'''

# ============================
# STARTUP
# ============================
@app.on_event("startup")
async def startup_event():
    """Warmup model on startup"""
    print("\n" + "="*70)
    print("NEUROLENS FATIGUE DETECTION API v2.0")
    print("="*70)
    print(f"Device: {device.type}")
    print(f"Model: Neurolens/NeuroLens-VideoMAE-Fatigue-v1")
    
    warmup(runs=3)
    
    print("\n[API] Ready to accept requests")
    print(f"  Predict: POST http://localhost:8000/predict")
    print(f"  Health: GET http://localhost:8000/health")
    print(f"  Reset: POST http://localhost:8000/reset")
    print(f"  Docs: http://localhost:8000/docs\n")

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
