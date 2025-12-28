# dashboard/inference.py
import os
import torch
import cv2
from dotenv import load_dotenv
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN not found. Check your .env file.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFERENCE] Using device: {device}")

MODEL_ID = "gowbs425/NeuroLens-VideoMAE"

# Load processor and model using token
print("[INFERENCE] Loading VideoMAE model...")
processor = VideoMAEImageProcessor.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN
)

model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    output_attentions=False  # Disable attentions for CPU speed
).to(device)

model.eval()

# CPU Optimizations
if device == "cpu":
    print("[INFERENCE] Applying CPU optimizations...")
    # Set number of threads for CPU inference
    torch.set_num_threads(4)  # Adjust based on your CPU cores
    print(f"[INFERENCE] Using {torch.get_num_threads()} CPU threads")
else:
    # GPU optimizations
    model = model.half()  # Use FP16 for faster inference
    print("[INFERENCE] Using FP16 precision (GPU)")

print("[INFERENCE] Model loaded successfully!")

@torch.inference_mode()  # Faster than no_grad
def predict_video(frames):
    """
    Run inference on video frames (CPU optimized)
    
    Args:
        frames: List of RGB numpy arrays [H, W, 3]
        
    Returns:
        prob: Float probability of fatigue (0-1)
        attentions: None (disabled for speed)
    """
    try:
        # Resize frames to smaller size for CPU processing
        # Using 224x224 as it's the standard size
        frames_resized = []
        for frame in frames:
            resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            frames_resized.append(resized)
        
        # Process frames
        inputs = processor(frames_resized, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        # Convert to FP16 only if using GPU
        if device == "cuda":
            pixel_values = pixel_values.half()
        
        # Run model inference
        outputs = model(pixel_values)
        
        # Get prediction probability
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        fatigue_prob = probs[0, 1].item()  # Index 1 is fatigue class
        
        print(f"[INFERENCE] Prediction: {fatigue_prob:.3f}")
        
        # Return None for attentions (disabled for speed)
        return fatigue_prob, None
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, None
