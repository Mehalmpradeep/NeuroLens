'''# dashboard/xai.py
import torch
import numpy as np

def attention_rollout(attentions):
    """
    attentions: tuple of tensors from VideoMAE
    Returns: aggregated attention rollout tensor
    """
    rollout = None
    for layer in attentions:
        attn = layer.mean(dim=1)  # average heads
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn if rollout is None else attn @ rollout
    return rollout'''
    # dashboard/xai.py
import torch
import numpy as np
import cv2


def attention_rollout(attentions):
    """
    attentions: tuple of (B, num_heads, N, N)
    returns: (N, N) attention rollout
    """
    rollout = None

    for layer in attentions:
        attn = layer.mean(dim=1)  # average heads → (B, N, N)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        rollout = attn if rollout is None else attn @ rollout

    return rollout


def explain_video(frames, attentions, patch_size=16):
    """
    frames: list of RGB frames (len = 16)
    attentions: VideoMAE attention tuple
    returns: overlay image (H, W, 3)
    """

    # ---- rollout ----
    rollout = attention_rollout(attentions)  # (1, N, N)
    rollout = rollout[0]

    # Remove CLS token
    rollout = rollout[1:, 1:]

    num_patches = int(np.sqrt(rollout.shape[0]))
    attn_map = rollout.mean(dim=0)
    attn_map = attn_map.reshape(num_patches, num_patches)
    attn_map = attn_map.detach().cpu().numpy()

    # Normalize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() + 1e-6)

    # Pick middle frame
    frame = frames[len(frames) // 2]
    h, w, _ = frame.shape

    attn_map = cv2.resize(attn_map, (w, h))
    heatmap = np.uint8(255 * attn_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    return overlay

