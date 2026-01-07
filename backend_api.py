"""
FastAPI Backend for Real-time Fatigue Detection
Handles all heavy inference operations
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import base64
from typing import List, Dict
import asyncio
from collections import deque
import time

# Your inference module
from inference import predict_video

app = FastAPI(title="NeuroLens Backend")

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
frame_buffer = deque(maxlen=32)
prediction_history = deque(maxlen=10)
CLIP_LENGTH = 16

@app.post("/predict")
async def predict_fatigue(frames: List[str]):
    """
    Receive base64 frames, run inference
    """
    try:
        # Decode frames
        decoded_frames = []
        for frame_b64 in frames[-CLIP_LENGTH:]:
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            decoded_frames.append(img_rgb)
        
        if len(decoded_frames) < CLIP_LENGTH:
            return {
                "status": "insufficient_frames",
                "required": CLIP_LENGTH,
                "received": len(decoded_frames)
            }
        
        # Run inference (offloaded from frontend)
        start = time.time()
        prob, aux = predict_video(decoded_frames)
        inference_time = time.time() - start
        
        result = {
            "status": "success",
            "fatigue_prob": float(prob),
            "inference_time_ms": int(inference_time * 1000),
            "timestamp": time.time()
        }
        
        prediction_history.append(result)
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "predictions_made": len(prediction_history),
        "last_prediction": prediction_history[-1] if prediction_history else None
    }

@app.get("/history")
async def get_history():
    return {"predictions": list(prediction_history)}

if __name__ == "__main__":
    print("🚀 Starting NeuroLens Backend on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")