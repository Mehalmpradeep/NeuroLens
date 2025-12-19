import torch
ckpt = torch.load("D:/NeuroLens/checkpoints/last_checkpoint.pt", map_location="cpu")
print("Best validation accuracy:", ckpt["best_val_acc"])
print("Stopped at epoch:", ckpt["epoch"])
