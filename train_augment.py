# ===============================
# 1. IMPORTS & GPU CHECK
# ===============================
import torch
import os
import cv2
import random
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from decord import VideoReader, cpu
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# ===============================
# 2. CONFIG
# ===============================
BASE_DIR = "D:/NeuroLens"

TRAIN_CSV = f"{BASE_DIR}/csv/all_train.csv"
VAL_CSV   = f"{BASE_DIR}/csv/all_val.csv"
DATASET_ROOT = f"{BASE_DIR}/datasets"

ROBUST_CKPT_DIR = f"{BASE_DIR}/checkpoints_robust"
os.makedirs(ROBUST_CKPT_DIR, exist_ok=True)

HF_MODEL_ID = "Neurolens/NeuroLens-VideoMAE"

RESUME_CKPT = f"{ROBUST_CKPT_DIR}/resume_state.pt"

NUM_FRAMES = 16
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-6
PATIENCE = 2
NUM_WORKERS = 0

# ===============================
# 3. LIGHT ROBUSTNESS AUGMENTATION
# ===============================
def apply_light_robustness_aug(frame):
    if random.random() < 0.6:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-15, 15)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    if random.random() < 0.3:
        gamma = random.uniform(0.9, 1.2)
        inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
        frame = cv2.LUT(frame, table)

    if random.random() < 0.2:
        noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame

# ===============================
# 4. DATASET  (FIXED – SKIPS BAD VIDEOS)
# ===============================
class NeuroLensDataset(Dataset):
    def __init__(self, csv_file, processor, augment=False):
        self.data = pd.read_csv(csv_file)
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def load_video(self, path):
        try:
            vr = VideoReader(path, ctx=cpu(0))
            if len(vr) < NUM_FRAMES:
                return None
            idx = np.linspace(0, len(vr) - 1, NUM_FRAMES).astype(int)
            return vr.get_batch(idx).asnumpy()
        except:
            return None

    def __getitem__(self, idx):
        max_retries = 5

        for _ in range(max_retries):
            row = self.data.iloc[idx]
            path = row["path"]
            if not os.path.isabs(path):
                path = os.path.join(DATASET_ROOT, path)

            frames = self.load_video(path)
            if frames is not None:
                processed = []
                for f in frames:
                    if self.augment:
                        f = apply_light_robustness_aug(f)
                    processed.append(f)

                inputs = self.processor(processed, return_tensors="pt")
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["labels"] = torch.tensor(int(row["label"]))
                return inputs

            # move to next index if video is bad
            idx = (idx + 1) % len(self.data)

        # extremely unlikely unless dataset is badly corrupted
        raise RuntimeError("Too many invalid videos encountered")

# ===============================
# 5. LOAD TRAINED FATIGUE MODEL
# ===============================
processor = VideoMAEImageProcessor.from_pretrained(HF_MODEL_ID)

model = VideoMAEForVideoClassification.from_pretrained(
    HF_MODEL_ID
).to(device)

# ===============================
# 6. FREEZE BACKBONE (CRITICAL)
# ===============================
for param in model.videomae.parameters():
    param.requires_grad = False

for param in model.videomae.encoder.layer[-1].parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🧠 Trainable parameters: {trainable:,} / {total:,}")

# ===============================
# 7. DATALOADERS
# ===============================
train_loader = DataLoader(
    NeuroLensDataset(TRAIN_CSV, processor, augment=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    NeuroLensDataset(VAL_CSV, processor, augment=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# ===============================
# 8. OPTIMIZER & AMP
# ===============================
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
scaler = torch.cuda.amp.GradScaler()

# ===============================
# 8.1 RESUME LOGIC
# ===============================
start_epoch = 0
best_val_acc = 0
no_improve = 0

if os.path.exists(RESUME_CKPT):
    print("🔄 Resuming training from checkpoint")
    ckpt = torch.load(RESUME_CKPT, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt["epoch"] + 1
    best_val_acc = ckpt["best_val_acc"]
    no_improve = ckpt["no_improve"]

# ===============================
# 9. TRAINING LOOP
# ===============================
for epoch in range(start_epoch, EPOCHS):
    print(f"\n🟢 Epoch {epoch+1}/{EPOCHS}")
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            out = model(
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device)
            )

        scaler.scale(out.loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += out.loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device)
            )
            preds = out.logits.argmax(dim=1)
            correct += (preds == batch["labels"].to(device)).sum().item()
            total += batch["labels"].size(0)

    val_acc = correct / total
    print(f"📉 Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"🎯 Val Acc   : {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        model.save_pretrained(ROBUST_CKPT_DIR)
        processor.save_pretrained(ROBUST_CKPT_DIR)
        print("💾 Best robustness model saved")
    else:
        no_improve += 1

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_acc": best_val_acc,
        "no_improve": no_improve
    }, RESUME_CKPT)

    if no_improve >= PATIENCE:
        print("⏹️ Early stopping")
        break

print("🏁 Robustness fine-tuning complete")
