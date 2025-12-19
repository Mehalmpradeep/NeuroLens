import torch
import numpy as np
from decord import VideoReader, cpu
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# ===============================
# CONFIG
# ===============================
CHECKPOINT_DIR = "D:/NeuroLens/checkpoints"
NUM_FRAMES = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# LOAD MODEL + PROCESSOR
# ===============================
print("Loading model...")
processor = VideoMAEImageProcessor.from_pretrained(CHECKPOINT_DIR)
model = VideoMAEForVideoClassification.from_pretrained(CHECKPOINT_DIR).to(device)
model.eval()

print("Model ready!")

# ===============================
# VIDEO LOADING FUNCTION
# ===============================
def load_video(path, num_frames=16):
    try:
        vr = VideoReader(path, ctx=cpu(0))
        if len(vr) < num_frames:
            raise ValueError("Video too short")

        indices = np.linspace(0, len(vr)-1, num_frames).astype(int)
        frames = vr.get_batch(indices).asnumpy()
        return frames
    except Exception as e:
        print(f"Error loading video: {e}")
        return None

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict(video_path):
    frames = load_video(video_path, NUM_FRAMES)
    if frames is None:
        return None

    # Preprocess frames
    inputs = processor(list(frames), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    label_map = {0: "NOT DROWSY", 1: "DROWSY"}
    return label_map[pred_class], confidence

# ===============================
# TEST
# ===============================
if __name__ == "__main__":
    test_video = "D:/NeuroLens/datasets/uta-rldd-clips/test/notdrowsy/004_utaclip00000.mp4"

    prediction, score = predict(test_video)
    print("\n======================")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {score:.4f}")
    print("======================")
