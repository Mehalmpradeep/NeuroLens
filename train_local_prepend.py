# ===============================
# 1. GPU CHECK
# ===============================
import torch

print("🚀 GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🚀 GPU name:", torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 2. IMPORTS
# ===============================
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from decord import VideoReader, cpu
from tqdm import tqdm

# ===============================
# 3. PATHS (CHANGE ONLY THIS IF NEEDED)
# ===============================
BASE_DIR = "D:/NeuroLens"

# CSVs are OUTSIDE datasets folder
TRAIN_CSV = f"{BASE_DIR}/csv/all_train.csv"
VAL_CSV   = f"{BASE_DIR}/csv/all_val.csv"

# All videos are inside this folder
DATASET_ROOT = f"{BASE_DIR}/datasets"

VIDEO_COL = "path"
LABEL_COL = "label"

CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===============================
# 4. LOAD CSVs
# ===============================
print("📄 Loading CSVs...")
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

print(f"✅ Train samples: {len(train_df)}")
print(f"✅ Val samples  : {len(val_df)}")

# ===============================
# 5. DATASET CLASS (WITH PREPENDING)
# ===============================
class NeuroLensDataset(Dataset):
    def __init__(self, csv_file, processor, num_frames=16):
        self.data = pd.read_csv(csv_file)
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def load_video(self, path):
        try:
            vr = VideoReader(path, ctx=cpu(0))
            if len(vr) < self.num_frames:
                return None

            indices = np.linspace(0, len(vr) - 1, self.num_frames).astype(int)
            return vr.get_batch(indices).asnumpy()
        except Exception:
            return None

    def __getitem__(self, idx):
        while True:
            row = self.data.iloc[idx]
            video_path = row[VIDEO_COL]

            # 🔑 PREPEND DATASET ROOT IF PATH IS RELATIVE
            if not os.path.isabs(video_path):
                video_path = os.path.join(DATASET_ROOT, video_path)

            frames = self.load_video(video_path)

            if frames is not None:
                inputs = self.processor(list(frames), return_tensors="pt")
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["labels"] = torch.tensor(int(row[LABEL_COL]))
                return inputs

            idx = (idx + 1) % len(self.data)

print("✅ Dataset class ready")

# ===============================
# 6. MODEL & PROCESSOR
# ===============================
print("🤖 Loading VideoMAE...")
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base",
    num_labels=2,
    ignore_mismatched_sizes=True
).to(device)

print("✅ Model loaded")

# ===============================
# 7. DATASETS & DATALOADERS
# ===============================
train_dataset = NeuroLensDataset(TRAIN_CSV, processor)
val_dataset   = NeuroLensDataset(VAL_CSV, processor)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,      # reduce to 1 if CUDA OOM
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("✅ DataLoaders ready")

# ===============================
# 8. OPTIMIZER
# ===============================
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ===============================
# 9. TRAIN & EVALUATION
# ===============================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()

        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

# ===============================
# 10. TRAIN LOOP
# ===============================
EPOCHS = 5
best_val_acc = 0

print("🔥 Training started")

for epoch in range(EPOCHS):
    print(f"\n🟢 Epoch {epoch+1}/{EPOCHS}")

    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_acc = evaluate(model, val_loader)

    print(f"📉 Train Loss: {train_loss:.4f}")
    print(f"📊 Val Loss  : {val_loss:.4f}")
    print(f"🎯 Val Acc   : {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(CHECKPOINT_DIR)
        processor.save_pretrained(CHECKPOINT_DIR)
        print("💾 Best model saved")

print("🏁 Training complete")
