# ===============================
# 1. GPU CHECK
# ===============================
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from decord import VideoReader, cpu
from tqdm import tqdm

print("🚀 GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🚀 GPU name:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 2. CONFIG
# ===============================
BASE_DIR = "D:/NeuroLens"

TRAIN_CSV = f"{BASE_DIR}/csv/all_train.csv"
VAL_CSV   = f"{BASE_DIR}/csv/all_val.csv"
DATASET_ROOT = f"{BASE_DIR}/datasets"

VIDEO_COL = "path"
LABEL_COL = "label"

CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RESUME_CHECKPOINT = f"{CHECKPOINT_DIR}/last_checkpoint.pt"

NUM_FRAMES = 16
BATCH_SIZE = 2
EPOCHS = 10
LR = 2e-5
NUM_WORKERS = 0

# 🔹 EARLY STOPPING
PATIENCE = 3
no_improve_epochs = 0

# ===============================
# 3. LOAD CSVs
# ===============================
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

print(f"✅ Train samples: {len(train_df)}")
print(f"✅ Val samples  : {len(val_df)}")

# ===============================
# 4. DATASET CLASS
# ===============================
class NeuroLensDataset(Dataset):
    def __init__(self, csv_file, processor, num_frames=16, max_retries=5):
        self.data = pd.read_csv(csv_file)
        self.processor = processor
        self.num_frames = num_frames
        self.max_retries = max_retries

    def __len__(self):
        return len(self.data)

    def load_video(self, path):
        try:
            vr = VideoReader(path, ctx=cpu(0))
            if len(vr) < self.num_frames:
                return None
            idx = np.linspace(0, len(vr) - 1, self.num_frames).astype(int)
            return vr.get_batch(idx).asnumpy()
        except:
            return None

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            row = self.data.iloc[idx]
            video_path = row[VIDEO_COL]

            if not os.path.isabs(video_path):
                video_path = os.path.join(DATASET_ROOT, video_path)

            frames = self.load_video(video_path)
            if frames is not None:
                inputs = self.processor(list(frames), return_tensors="pt")
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["labels"] = torch.tensor(int(row[LABEL_COL]))
                return inputs

            idx = (idx + 1) % len(self.data)

        raise RuntimeError("Too many corrupted videos")

print("✅ Dataset class ready")

# ===============================
# 5. MODEL
# ===============================
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base",
    num_labels=2,
    ignore_mismatched_sizes=True
).to(device)

print("✅ Model loaded")

# ===============================
# 6. DATALOADERS
# ===============================
train_loader = DataLoader(
    NeuroLensDataset(TRAIN_CSV, processor, NUM_FRAMES),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    NeuroLensDataset(VAL_CSV, processor, NUM_FRAMES),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("✅ DataLoaders ready")

# ===============================
# 7. OPTIMIZER & AMP
# ===============================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# ===============================
# 8. TRAIN & EVAL
# ===============================
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(pixel_values=pixel_values, labels=labels)

        total_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

# ===============================
# 9. RESUME LOGIC
# ===============================
start_epoch = 0
best_val_acc = 0.0

if os.path.exists(RESUME_CHECKPOINT):
    print("🔄 Resuming from checkpoint")
    ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_val_acc = ckpt["best_val_acc"]

# ===============================
# 10. TRAIN LOOP (WITH EARLY STOPPING)
# ===============================
print("🔥 Training started")

for epoch in range(start_epoch, EPOCHS):
    print(f"\n🟢 Epoch {epoch + 1}/{EPOCHS}")

    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    print(f"📉 Train Loss: {train_loss:.4f}")
    print(f"📊 Val Loss  : {val_loss:.4f}")
    print(f"🎯 Val Acc   : {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve_epochs = 0
        model.save_pretrained(CHECKPOINT_DIR)
        processor.save_pretrained(CHECKPOINT_DIR)
        print("💾 Best model saved")
    else:
        no_improve_epochs += 1
        print(f"⚠️ No improvement for {no_improve_epochs} epoch(s)")

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_acc": best_val_acc
    }, RESUME_CHECKPOINT)

    print("💾 Checkpoint saved")

    if no_improve_epochs >= PATIENCE:
        print("⏹️ Early stopping triggered")
        break

print("🏁 Training complete")
