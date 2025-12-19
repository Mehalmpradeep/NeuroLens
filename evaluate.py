import os
import torch
import pandas as pd
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# =========================
# CONFIG
# =========================
CHECKPOINT_DIR = "D:/NeuroLens/checkpoints"
VAL_CSV = "D:/NeuroLens/csv/all_val.csv"
DATASET_ROOT = "D:/NeuroLens/datasets"
VIDEO_COL = "path"
LABEL_COL = "label"

NUM_FRAMES = 16
BATCH_SIZE = 16   # You can increase if GPU allows
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# LOAD MODEL
# =========================
print("Loading model...")
processor = VideoMAEImageProcessor.from_pretrained(CHECKPOINT_DIR)
model = VideoMAEForVideoClassification.from_pretrained(CHECKPOINT_DIR).to(device)
model.eval()
print("Model ready!")


# =========================
# VIDEO LOADING FUNCTION
# =========================
def load_video(path):
    try:
        vr = VideoReader(path, ctx=cpu(0))
        if len(vr) < NUM_FRAMES:
            return None
        idx = np.linspace(0, len(vr)-1, NUM_FRAMES).astype(int)
        frames = vr.get_batch(idx).asnumpy()
        return frames
    except:
        return None


# =========================
# BATCH EVALUATION
# =========================
df = pd.read_csv(VAL_CSV)
true_labels = []
pred_labels = []

videos = df[VIDEO_COL].tolist()
labels = df[LABEL_COL].tolist()

print(f"Total validation videos: {len(videos)}")

for start in tqdm(range(0, len(videos), BATCH_SIZE), desc="Evaluating batches"):
    end = min(start + BATCH_SIZE, len(videos))

    batch_frames = []
    batch_true = []

    # Load batch videos
    for i in range(start, end):
        rel_path = videos[i]
        label = int(labels[i])

        path = rel_path if os.path.isabs(rel_path) else os.path.join(DATASET_ROOT, rel_path)
        frames = load_video(path)

        if frames is not None:
            batch_frames.append(frames)
            batch_true.append(label)

    if len(batch_frames) == 0:
        continue

    # Preprocess entire batch at once
    inputs = processor([list(f) for f in batch_frames], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

    batch_pred = np.argmax(probs, axis=1)

    true_labels.extend(batch_true)
    pred_labels.extend(batch_pred)


# =========================
# METRICS
# =========================
acc = accuracy_score(true_labels, pred_labels)
cm = confusion_matrix(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, target_names=["NOT DROWSY", "DROWSY"])

print("\n=======================")
print(f"Final Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
print("=======================\n")
