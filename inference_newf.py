# dashboard/inference.py
import os
import torch
import cv2
from dotenv import load_dotenv
from transformers import VideoMAEImageProcessor
from neurolens_model import VideoMAEBinaryFatigueDetector

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN not found. Check your .env file.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFERENCE] Using device: {device}")

MODEL_ID = "NeuroLens/NeuroLens-VideoMAE-Robust"
CHECKPOINT_PATH = "D:/NeuroLens/checkpoints_new/best_model.pth" # Update path as needed

# Load processor
print("[INFERENCE] Loading VideoMAE processor...")
processor = VideoMAEImageProcessor.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN
)

# Load model with custom architecture
print("[INFERENCE] Loading custom VideoMAE model...")
model = VideoMAEBinaryFatigueDetector(model_name=MODEL_ID)

# Load trained weights
print(f"[INFERENCE] Loading weights from {CHECKPOINT_PATH}...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Handle different checkpoint formats
if isinstance(checkpoint, dict):
    # Check for common wrapper keys
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # Your checkpoint has this structure!
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Load the state dict
model.load_state_dict(state_dict, strict=True)
print("[INFERENCE] Weights loaded successfully!")

model = model.to(device)
model.eval()

# CPU Optimizations

if device == "cpu":
    print("[INFERENCE] Applying CPU optimizations...")
    torch.set_num_threads(4)  # Adjust based on your CPU cores
    print(f"[INFERENCE] Using {torch.get_num_threads()} CPU threads")
else:
    # GPU optimizations
    model = model.half()  # Use FP16 for faster inference
    print("[INFERENCE] Using FP16 precision (GPU)")

print("[INFERENCE] Model loaded and ready!")

@torch.inference_mode()  # Faster than no_grad
def predict_video(frames):
    """
    Run inference on video frames (CPU/GPU optimized)
    
    Args:
        frames: List of RGB numpy arrays [H, W, 3]
        
    Returns:
        prob: Float probability of fatigue (0-1)
        attentions: None (disabled for speed)
    """
    try:
        # Resize frames to 224x224 for processing
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
        
        # Run model inference - get raw logits
        logits = model(pixel_values)
        
        # Apply sigmoid to convert logit to probability (binary classification)
        fatigue_prob = torch.sigmoid(logits).item()
        
        print(f"[INFERENCE] Prediction: {fatigue_prob:.3f}")
        
        # Return None for attentions (disabled for speed)
        return fatigue_prob, None
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, None