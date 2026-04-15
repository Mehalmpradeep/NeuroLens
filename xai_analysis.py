"""
xai_analysis.py — NeuroLens XAI Research Module  (fully debugged)
==================================================================
Standalone explainability analysis for VideoMAE-based fatigue detection.
Implements Grad-CAM and Attention Rollout for spatiotemporal interpretation.

All 10 issues from the diagnostic report are fixed in this file.
Every fix is marked with a  # FIX: <issue number>  comment.

Usage:
    # Single video file
    python xai_analysis.py --checkpoint best_model.pth --video_path clip.mp4

    # Directory of frame images for one clip
    python xai_analysis.py --checkpoint best_model.pth --frames_dir ./clip1/

    # Batch — entire folder of videos
    python xai_analysis.py --checkpoint best_model.pth --video_dir ./clips/

    # Self-test with a random tensor (no real clip needed)
    python xai_analysis.py --checkpoint best_model.pth --self_test
"""

import os
import sys
import json
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurolens_model1 import VideoMAEBinaryFatigueDetector

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("xai_analysis")

# =============================================================================
# CONSTANTS
# =============================================================================
MODEL_ID       = "Neurolens/NeuroLens-VideoMAE-Robust"
FATIGUE_THRESH = 0.70
NUM_FRAMES     = 16
PATCH_SIZE     = 16
TUBELET_SIZE   = 2
IMAGE_SIZE     = 224

T_PATCHES           = NUM_FRAMES  // TUBELET_SIZE   # 8
H_PATCHES           = IMAGE_SIZE  // PATCH_SIZE     # 14
W_PATCHES           = IMAGE_SIZE  // PATCH_SIZE     # 14
NUM_SPATIAL_PATCHES = H_PATCHES * W_PATCHES          # 196
NUM_PATCHES         = T_PATCHES * NUM_SPATIAL_PATCHES  # 1568  (no CLS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"XAI module using device: {device}")


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(
    checkpoint_path: str,
) -> Tuple[VideoMAEBinaryFatigueDetector, VideoMAEImageProcessor]:
    """Load a trained NeuroLens checkpoint and the matching HuggingFace processor."""
    log.info(f"Loading model from: {checkpoint_path}")
    processor = VideoMAEImageProcessor.from_pretrained(
        MODEL_ID, token=os.getenv("HF_TOKEN")
    )
    model = VideoMAEBinaryFatigueDetector(model_name=MODEL_ID).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    log.info("Checkpoint loaded successfully.")
    return model, processor


# =============================================================================
# SHARED HELPERS
# =============================================================================

def upsample_cam_to_frame(
    cam_hw: np.ndarray,
    target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
    renorm: bool = False,
) -> np.ndarray:
    """
    Bilinearly upsample a patch-resolution CAM to pixel resolution.

    Args:
        cam_hw:      [H_patches, W_patches] float32 array.
        target_size: (H, W) output pixel resolution.
        renorm:      If True, re-normalise the upsampled map to [0,1].
                     Leave False when global normalisation has already been
                     applied (Grad-CAM path); set True for visualisation-only
                     use cases where per-frame contrast is acceptable.
    """
    cam_up = cv2.resize(
        cam_hw, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
    )
    if renorm:
        lo, hi = cam_up.min(), cam_up.max()
        cam_up = (cam_up - lo) / (hi - lo + 1e-8)
    return cam_up.astype(np.float32)


def overlay_heatmap(
    frame_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a [0,1] float CAM heatmap over an RGB uint8 frame."""
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(
        frame_rgb.astype(np.float32), 1 - alpha,
        heatmap_rgb.astype(np.float32), alpha, 0,
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def compute_spatial_entropy(cam: np.ndarray) -> float:
    """Normalised Shannon entropy of the activation distribution (0=focused, 1=diffuse)."""
    flat  = np.clip(cam.flatten().astype(np.float64), 0, None)
    total = flat.sum()
    if total < 1e-10:
        return 0.0
    probs   = flat / total
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy / math.log(len(probs) + 1e-10))


def compute_metrics(
    cam: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, float]:
    """Compute a standard suite of spatial + face-region metrics for one CAM."""
    metrics: Dict[str, float] = {
        "mean_activation":         float(cam.mean()),
        "max_activation":          float(cam.max()),
        "std_activation":          float(cam.std()),
        "spatial_entropy":         compute_spatial_entropy(cam),
        "high_activation_sparsity": float((cam >= 0.5 * cam.max()).mean()),
    }
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(cam.shape[1], x2), min(cam.shape[0], y2)
        face_sum  = cam[y1:y2, x1:x2].sum()
        total_sum = cam.sum()
        metrics["face_activation_pct"] = float(face_sum / (total_sum + 1e-8))
    else:
        metrics["face_activation_pct"] = -1.0
    return metrics


# =============================================================================
# GRAD-CAM
# =============================================================================

class GradCAM:
    """
    Class-discriminative Grad-CAM for VideoMAEForVideoClassification.

    KEY FIXES applied vs. the original implementation
    --------------------------------------------------
    FIX 1 — No CLS token (issue #1 & #2):
        Your checkpoint produces N=1568 tokens — exactly T×H×W with NO CLS.
        The old code asserted N_plus_1 == N_expected+1, which crashed.
        Fixed by auto-detecting CLS presence: if N == T×H×W, skip the slice;
        if N == T×H×W + 1, remove index 0.

    FIX 2 — .detach() inside the forward hook killed the gradient tape.
        Stored the live tensor; .detach() called only after backward().

    FIX 3 — Tensor-level gradient hook instead of module backward hook.
        register_hook() on the activation tensor itself is robust across
        all HuggingFace tuple-wrapping variants.

    FIX 4 — torch.enable_grad() wraps the forward+backward pass so the
        method works even when the caller is inside model.eval().

    FIX 5 — Correct Grad-CAM formula: element-wise (grad * act).sum(dim=-1)
        instead of mean(grad) * sum(act), which lost channel correspondence.

    FIX 6 — Global normalisation over the full [T,H,W] volume instead of
        per-frame normalisation, preserving temporal comparability.

    FIX 7 — T, H, W, tubelet_size read from model.config at runtime,
        not from module-level global constants.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._hooks: List = []
        self._register_hooks()

    # ── hooks ─────────────────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        last_block = self._last_encoder_block()

        def _fwd(module, _inp, output):
            # FIX 2: store live tensor — no .detach() here
            hidden = output[0] if isinstance(output, tuple) else output
            self._activations = hidden

        self._hooks.append(last_block.register_forward_hook(_fwd))

    def _last_encoder_block(self) -> nn.Module:
        try:
            return self.model.videomae.videomae.encoder.layer[-1]
        except AttributeError:
            pass
        for _, mod in reversed(list(self.model.named_modules())):
            if isinstance(mod, nn.ModuleList):
                return mod[-1]
        raise RuntimeError("Cannot locate last encoder block.")

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._activations = None

    # ── compute ───────────────────────────────────────────────────────────────

    def compute(
        self, pixel_values: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Returns:
            per_frame_cams   : float32 [NUM_FRAMES, H, W]  globally normalised
            temporal_avg_cam : float32 [H, W]
            fatigue_prob     : float
        """
        # FIX 4: guarantee autograd is active regardless of calling context
        with torch.enable_grad():
            return self._compute_inner(pixel_values)

    def _compute_inner(
        self, pixel_values: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, float]:

        dev = next(self.model.parameters()).device
        pixel_values = pixel_values.detach().to(dev)

        # ── forward ──────────────────────────────────────────────────────────
        self.model.zero_grad()
        logits = self.model(pixel_values)           # [B, 1]
        prob   = float(torch.sigmoid(logits[0, 0]).item())

        assert self._activations is not None, (
            "Forward hook did not fire. Check _last_encoder_block()."
        )

        # ── register tensor-level gradient hook (FIX 3) ──────────────────────
        grad_bucket: List[Optional[torch.Tensor]] = [None]

        def _grad_hook(g: torch.Tensor) -> None:
            grad_bucket[0] = g.detach().clone()

        h = self._activations.register_hook(_grad_hook)

        # ── backward ─────────────────────────────────────────────────────────
        logits[0, 0].backward()
        h.remove()

        acts  = self._activations.detach().clone()  # FIX 2: detach after backward
        grads = grad_bucket[0]

        assert grads is not None, (
            "Gradient hook did not fire. Ensure the model is not wrapped in "
            "torch.inference_mode() at the call site."
        )

        grad_norm = grads.abs().max().item()
        if grad_norm < 1e-8:
            log.warning(
                f"Gradient near-zero ({grad_norm:.2e}). CAM may be blank. "
                "Possible causes: saturated sigmoid, or detached activations "
                "inside a sub-module."
            )

        # ── CLS-token handling (FIX 1) ────────────────────────────────────────
        # Detect architecture: with-CLS or without-CLS.
        cfg        = self.model.videomae.config
        T_p        = cfg.num_frames   // cfg.tubelet_size   # FIX 7
        H_p        = cfg.image_size   // cfg.patch_size     # FIX 7
        W_p        = cfg.image_size   // cfg.patch_size     # FIX 7
        N_expected = T_p * H_p * W_p                        # 1568

        B, N_actual, D = acts.shape

        if N_actual == N_expected:
            # ── No CLS token (your checkpoint) ── FIX 1
            acts_patch  = acts[0]         # [1568, D]
            grads_patch = grads[0]        # [1568, D]
            log.debug("GradCAM: no-CLS architecture detected (%d tokens).", N_actual)

        elif N_actual == N_expected + 1:
            # ── CLS token present at index 0 ──
            acts_patch  = acts[0,  1:, :]  # [1568, D]
            grads_patch = grads[0, 1:, :]  # [1568, D]
            log.debug("GradCAM: CLS architecture detected (%d tokens).", N_actual)

        else:
            raise ValueError(
                f"Unexpected token count {N_actual}. "
                f"Expected {N_expected} (no CLS) or {N_expected+1} (with CLS). "
                f"Config: T={T_p}, H={H_p}, W={W_p}."
            )

        # ── Grad-CAM weighting (FIX 5) ────────────────────────────────────────
        # Correct formula: element-wise dot-product per patch token.
        #   cam_p = ReLU( Σ_d  grad_{p,d} · act_{p,d} )
        cam_tokens = F.relu(
            (grads_patch * acts_patch).sum(dim=-1)   # [N_patches]
        )

        # ── Reshape to spatiotemporal grid (FIX 7) ────────────────────────────
        cam_3d = cam_tokens.view(T_p, H_p, W_p).cpu().float()   # [T, H, W]

        # ── Global normalisation (FIX 6) ──────────────────────────────────────
        cam_min, cam_max = cam_3d.min(), cam_3d.max()
        if (cam_max - cam_min) < 1e-8:
            log.warning("CAM is all-zero after ReLU (no positive contributions).")
            cam_3d_norm = torch.zeros_like(cam_3d)
        else:
            cam_3d_norm = (cam_3d - cam_min) / (cam_max - cam_min + 1e-8)

        # ── Expand tubelets → one CAM per input frame (FIX 7) ────────────────
        tubelet = cfg.tubelet_size
        per_frame_cams: List[np.ndarray] = []
        for t in range(T_p):
            cam_up = upsample_cam_to_frame(
                cam_3d_norm[t].numpy(), (cfg.image_size, cfg.image_size)
            )
            for _ in range(tubelet):
                per_frame_cams.append(cam_up)

        stacked          = np.stack(per_frame_cams, axis=0)   # [16, H, W]
        temporal_avg_cam = stacked.mean(axis=0)                # [H, W]
        return stacked, temporal_avg_cam, prob


# =============================================================================
# ATTENTION ROLLOUT
# =============================================================================

class AttentionRollout:
    """
    Attention Rollout (Abnar & Zuidema, 2020) for VideoMAE.

    KEY FIXES applied vs. the original implementation
    --------------------------------------------------
    FIX 1 — No CLS token:
        Standard rollout reads the CLS row (row 0) of the rolled-out matrix.
        When there is no CLS, we average all rows instead, which gives the
        mean information flow from every patch to every other patch — a
        reasonable proxy for patch importance without a global aggregator.

    FIX 8 — Hook guards against None attention outputs.
        Some HuggingFace attention modules return (hidden, None) when
        output_attentions is not propagated correctly. The hook now logs
        a warning and skips None tensors instead of crashing.

    FIX 9 — @torch.inference_mode() removed from compute().
        inference_mode prevents gradient computation and also prevents
        tensor hooks from firing in some PyTorch versions. Replaced with
        torch.no_grad() which is safe for a gradient-free method.
    """

    def __init__(self, model: VideoMAEBinaryFatigueDetector, discard_ratio: float = 0.9):
        self.model         = model
        self.discard_ratio = discard_ratio
        self._attn_weights: List[torch.Tensor] = []
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        encoder_layers = self.model.videomae.videomae.encoder.layer

        def make_hook(idx: int):
            def hook(module, _inp, output):
                # FIX 8: guard against None attention weights
                if not (isinstance(output, tuple) and len(output) > 1):
                    return
                attn = output[1]
                if attn is None:
                    log.debug("AttentionRollout: layer %d returned None attn weights.", idx)
                    return
                self._attn_weights.append(attn.detach().cpu())
            return hook

        for i, layer in enumerate(encoder_layers):
            attn_mod = layer.attention.attention
            self._hooks.append(attn_mod.register_forward_hook(make_hook(i)))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # FIX 9: use torch.no_grad() instead of @torch.inference_mode()
    def compute(
        self, pixel_values: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Returns:
            per_frame_cams   : float32 [NUM_FRAMES, H, W]
            temporal_avg_cam : float32 [H, W]
            fatigue_prob     : float
        """
        with torch.no_grad():
            return self._compute_inner(pixel_values)

    def _compute_inner(
        self, pixel_values: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, float]:

        self._attn_weights.clear()
        dev          = next(self.model.parameters()).device
        pixel_values = pixel_values.to(dev)

        # Temporarily enable attention output on the inner model
        orig_flag = self.model.videomae.config.output_attentions
        self.model.videomae.config.output_attentions = True

        outputs = self.model.videomae(pixel_values=pixel_values, output_attentions=True)
        prob    = float(torch.sigmoid(outputs.logits[0, 0]).item())

        self.model.videomae.config.output_attentions = orig_flag

        if not self._attn_weights:
            log.warning(
                "AttentionRollout: no attention weights captured — "
                "falling back to uniform saliency map. "
                "Check that output_attentions=True is propagated through all "
                "encoder layers in this HuggingFace model version."
            )
            uniform = np.ones((NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            return uniform, uniform[0], prob

        # ── Rollout ───────────────────────────────────────────────────────────
        N_seq = self._attn_weights[0].shape[-1]   # N or N+1 depending on arch
        result: Optional[torch.Tensor] = None

        for attn in self._attn_weights:
            attn_avg = attn[0].mean(dim=0)   # [N_seq, N_seq]

            if self.discard_ratio > 0:
                cutoff   = torch.quantile(attn_avg.flatten(), self.discard_ratio)
                attn_avg = torch.where(attn_avg >= cutoff, attn_avg,
                                       torch.zeros_like(attn_avg))

            identity = torch.eye(N_seq)
            attn_hat = 0.5 * attn_avg + 0.5 * identity
            attn_hat = attn_hat / (attn_hat.sum(dim=-1, keepdim=True) + 1e-8)

            result = attn_hat if result is None else torch.matmul(attn_hat, result)

        # ── Extract patch importance (FIX 1) ──────────────────────────────────
        cfg        = self.model.videomae.config
        T_p        = cfg.num_frames  // cfg.tubelet_size
        H_p        = cfg.image_size  // cfg.patch_size
        W_p        = cfg.image_size  // cfg.patch_size
        N_expected = T_p * H_p * W_p

        if N_seq == N_expected + 1:
            # CLS architecture: read the CLS row → attention to each patch
            patch_importance = result[0, 1:]             # [N_patches]
        elif N_seq == N_expected:
            # No-CLS architecture (FIX 1): average all rows
            patch_importance = result.mean(dim=0)        # [N_patches]
        else:
            # Unexpected: trim and use mean
            log.warning(
                "AttentionRollout: N_seq=%d does not match expected %d or %d. "
                "Trimming and averaging.", N_seq, N_expected, N_expected + 1,
            )
            patch_importance = result.mean(dim=0)[:N_expected]

        # ── Reshape and upsample ──────────────────────────────────────────────
        usable = T_p * H_p * W_p
        cam_3d = patch_importance[:usable].view(T_p, H_p, W_p).numpy()

        per_frame_cams: List[np.ndarray] = []
        for t in range(T_p):
            cam_up = upsample_cam_to_frame(
                cam_3d[t], (cfg.image_size, cfg.image_size), renorm=True
            )
            for _ in range(cfg.tubelet_size):
                per_frame_cams.append(cam_up)

        stacked          = np.stack(per_frame_cams, axis=0)
        temporal_avg_cam = stacked.mean(axis=0)
        return stacked, temporal_avg_cam, prob


# =============================================================================
# FRAME / VIDEO LOADERS
# =============================================================================

def load_frames_from_dir(frames_dir: str) -> List[np.ndarray]:
    exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in Path(frames_dir).iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No image files in {frames_dir}")
    indices = np.linspace(0, len(paths) - 1, NUM_FRAMES, dtype=int)
    frames  = []
    for i in indices:
        bgr = cv2.imread(str(paths[i]))
        if bgr is None:
            raise IOError(f"Cannot read: {paths[i]}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def load_frames_from_video(video_path: str) -> List[np.ndarray]:
    # FIX 3 (path errors): resolve to absolute path before opening
    video_path = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(
            f"Cannot open video: {video_path}\n"
            "Check that the path is correct and the file exists."
        )
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Video appears empty: {video_path}")

    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    index_set = set(indices.tolist())
    frames_dict: Dict[int, np.ndarray] = {}
    current = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current in index_set:
            frames_dict[current] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current += 1
        if current > int(indices[-1]):
            break

    cap.release()
    result = [frames_dict[i] for i in indices if i in frames_dict]
    while len(result) < NUM_FRAMES:
        result.append(result[-1])
    return result[:NUM_FRAMES]


def preprocess_clip(
    frames: List[np.ndarray],
    processor: VideoMAEImageProcessor,
) -> torch.Tensor:
    resized = [
        cv2.resize(f, (IMAGE_SIZE, IMAGE_SIZE)) if f.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE) else f
        for f in frames
    ]
    inputs = processor(resized, return_tensors="pt")
    return inputs["pixel_values"].to(device)


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def save_frame_overlays(
    frames: List[np.ndarray],
    per_frame_cams: np.ndarray,
    output_dir: Path,
    prefix: str,
    method: str,
) -> List[str]:
    saved = []
    out   = output_dir / method / prefix
    out.mkdir(parents=True, exist_ok=True)
    for i, (frame, cam) in enumerate(zip(frames, per_frame_cams)):
        img   = overlay_heatmap(cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE)), cam)
        fname = out / f"frame_{i:02d}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        saved.append(str(fname))
    return saved


def save_temporal_average(
    avg_cam: np.ndarray,
    frames: List[np.ndarray],
    output_dir: Path,
    prefix: str,
    method: str,
) -> str:
    out   = output_dir / method / prefix
    out.mkdir(parents=True, exist_ok=True)
    mid   = cv2.resize(frames[NUM_FRAMES // 2], (IMAGE_SIZE, IMAGE_SIZE))
    img   = overlay_heatmap(mid, avg_cam)
    fname = out / "temporal_average.png"
    cv2.imwrite(str(fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return str(fname)


# =============================================================================
# SINGLE-CLIP EVALUATION
# =============================================================================

def evaluate_clip(
    clip_id: str,
    frames: List[np.ndarray],
    model: VideoMAEBinaryFatigueDetector,
    processor: VideoMAEImageProcessor,
    gradcam: GradCAM,
    rollout: AttentionRollout,
    output_dir: Path,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Dict:
    """Run full Grad-CAM + Rollout XAI evaluation on one clip."""
    log.info(f"[{clip_id}] Preprocessing {len(frames)} frames …")
    pixel_values = preprocess_clip(frames, processor)
    result       = {"clip_id": clip_id}

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    log.info(f"[{clip_id}] Running Grad-CAM …")
    gc_frame_cams, gc_avg_cam, gc_prob = gradcam.compute(pixel_values)

    result["fatigue_probability_gradcam"] = round(gc_prob, 5)
    result["fatigue_class_gradcam"]       = "Drowsy" if gc_prob >= FATIGUE_THRESH else "Alert"
    result["gradcam_metrics"] = {
        k: round(v, 6) for k, v in compute_metrics(gc_avg_cam, face_bbox).items()
    }
    result["gradcam_outputs"] = {
        "frame_overlays":   save_frame_overlays(frames, gc_frame_cams, output_dir, clip_id, "gradcam"),
        "temporal_average": save_temporal_average(gc_avg_cam, frames, output_dir, clip_id, "gradcam"),
    }

    # ── Attention Rollout ──────────────────────────────────────────────────────
    log.info(f"[{clip_id}] Running Attention Rollout …")
    ro_frame_cams, ro_avg_cam, ro_prob = rollout.compute(pixel_values)

    result["fatigue_probability_rollout"] = round(ro_prob, 5)
    result["fatigue_class_rollout"]       = "Drowsy" if ro_prob >= FATIGUE_THRESH else "Alert"
    result["rollout_metrics"] = {
        k: round(v, 6) for k, v in compute_metrics(ro_avg_cam, face_bbox).items()
    }
    result["rollout_outputs"] = {
        "frame_overlays":   save_frame_overlays(frames, ro_frame_cams, output_dir, clip_id, "rollout"),
        "temporal_average": save_temporal_average(ro_avg_cam, frames, output_dir, clip_id, "rollout"),
    }

    gc_e = result["gradcam_metrics"]["spatial_entropy"]
    ro_e = result["rollout_metrics"]["spatial_entropy"]
    result["entropy_delta"] = round(ro_e - gc_e, 6)

    log.info(
        f"[{clip_id}] Done. Prob={gc_prob:.3f}  "
        f"GC-entropy={gc_e:.3f}  RO-entropy={ro_e:.3f}"
    )
    return result


# =============================================================================
# BATCH EVALUATION
# =============================================================================

def batch_evaluate(
    checkpoint_path: str,
    sources: List[Dict],
    output_dir: str = "xai_results",
) -> None:
    """
    Run XAI evaluation over a list of clip sources.

    Each source dict must contain:
        "clip_id"                          (str)
        "frames_dir" OR "video_path"       (str)
        "face_bbox"                        (optional list [x1,y1,x2,y2])
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(checkpoint_path)
    gradcam = GradCAM(model)
    rollout = AttentionRollout(model)

    all_results = []

    for source in sources:
        clip_id = source["clip_id"]
        bbox    = tuple(source["face_bbox"]) if "face_bbox" in source else None

        try:
            if "frames_dir" in source:
                frames = load_frames_from_dir(source["frames_dir"])
            elif "video_path" in source:
                frames = load_frames_from_video(source["video_path"])
            else:
                log.error(f"[{clip_id}] No 'frames_dir' or 'video_path'. Skipping.")
                continue

            all_results.append(
                evaluate_clip(clip_id, frames, model, processor,
                              gradcam, rollout, out_path, bbox)
            )
        except Exception as exc:
            log.error(f"[{clip_id}] Failed: {exc}", exc_info=True)
            all_results.append({"clip_id": clip_id, "error": str(exc)})

    summary_path = out_path / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"All results saved to: {summary_path}")

    gradcam.remove_hooks()
    rollout.remove_hooks()
    _print_summary(all_results)


def _print_summary(results: List[Dict]) -> None:
    print("\n" + "═" * 72)
    print("  NEUROLENS XAI — BATCH SUMMARY")
    print("═" * 72)
    for r in results:
        if "error" in r:
            print(f"  ✗  {r['clip_id']}: ERROR — {r['error']}")
            continue
        prob   = r.get("fatigue_probability_gradcam", float("nan"))
        state  = r.get("fatigue_class_gradcam", "?")
        gc_e   = r.get("gradcam_metrics", {}).get("spatial_entropy", float("nan"))
        ro_e   = r.get("rollout_metrics", {}).get("spatial_entropy", float("nan"))
        face   = r.get("gradcam_metrics", {}).get("face_activation_pct", -1)
        face_s = f"{face*100:.1f}%" if isinstance(face, float) and face >= 0 else "N/A"
        print(f"  ✓  {r['clip_id']}")
        print(f"       Prob(fatigue)={prob:.3f}  State={state}")
        print(f"       GC-entropy={gc_e:.3f}  RO-entropy={ro_e:.3f}  Face={face_s}")
        print()
    print("═" * 72)


# =============================================================================
# SELF-TEST
# =============================================================================

def _self_test(checkpoint_path: str) -> None:
    """
    Smoke test with a random tensor.
    Verifies that GradCAM produces a non-zero map for this specific checkpoint.
    """
    log.info("=== SELF-TEST START ===")
    model, processor = load_model(checkpoint_path)

    dev   = next(model.parameters()).device
    dummy = torch.randn(1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE, device=dev)

    gc = GradCAM(model)
    per_frame_cams, avg_cam, prob = gc.compute(dummy)
    gc.remove_hooks()

    assert per_frame_cams.shape == (NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE), \
        f"Wrong shape: {per_frame_cams.shape}"
    assert per_frame_cams.max() > 1e-4, \
        "CAM is all-zero — gradient flow failed."

    log.info(
        f"SELF-TEST PASSED ✓  prob={prob:.3f}  "
        f"cam_max={per_frame_cams.max():.4f}  cam_mean={per_frame_cams.mean():.4f}"
    )

    ro = AttentionRollout(model)
    ro_cams, _, ro_prob = ro.compute(dummy)
    ro.remove_hooks()
    assert ro_cams.shape == (NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE), \
        f"Rollout wrong shape: {ro_cams.shape}"
    log.info(f"ROLLOUT SELF-TEST PASSED ✓  prob={ro_prob:.3f}")
    log.info("=== SELF-TEST COMPLETE ===")


# =============================================================================
# CLI  — FIX 4 (argument mismatch) & FIX 5 (script confusion)
# =============================================================================
# This is now the ONLY entry-point script. xai_analysis.py handles:
#   --self_test        smoke test with random tensor
#   --video_path       single video file
#   --frames_dir       single clip as a folder of images
#   --video_dir        batch mode over a folder of videos
# All arguments were missing or wrong in the original — rebuilt from scratch.
# =============================================================================
# DATASET CSV LOADER  (NEW)
# =============================================================================

def load_dataset_csv(csv_path: str) -> List[Dict]:
    """
    CSV format:
        relative_path,label

    Example:
        clips/test/drowsy/001_clip00000.mp4,1

    Converts to absolute paths using:
        D:/NeuroLens/datasets/  (auto-prepended)
    """

    BASE_DATASET_DIR = Path("D:/NeuroLens/datasets").resolve()

    sources: List[Dict] = []

    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rel_path, label = line.split(",")
            except ValueError:
                log.warning(f"Skipping malformed row: {line}")
                continue

            full_path = BASE_DATASET_DIR / rel_path

            if not full_path.exists():
                log.warning(f"File not found: {full_path}")
                continue

            sources.append({
                "clip_id": full_path.stem,
                "video_path": str(full_path.resolve()),
            })

    log.info(f"Loaded {len(sources)} clips from dataset CSV.")
    return sources

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NeuroLens XAI — Grad-CAM & Attention Rollout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (Windows PowerShell — use backtick ` for line continuation):
  python xai_analysis.py --checkpoint best_model.pth --self_test

  python xai_analysis.py --checkpoint best_model.pth `
      --video_path C:/data/clips/clip001.mp4 `
      --output_dir xai_results

  python xai_analysis.py --checkpoint best_model.pth `
      --video_dir C:/data/clips/ `
      --output_dir xai_results

  python xai_analysis.py --checkpoint best_model.pth `
      --frames_dir C:/data/frames/clip001/ `
      --output_dir xai_results

Examples (Linux/macOS — use backslash for line continuation):
  python xai_analysis.py --checkpoint best_model.pth \\
      --video_dir ./clips/ --output_dir xai_results
        """,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to best_model.pth")
    p.add_argument("--video_path",
                   help="Single video file (.mp4 / .avi / .mov / .mkv)")
    p.add_argument("--frames_dir",
                   help="Directory of frame images for a single clip")
    p.add_argument("--video_dir",
                   help="Directory of video files (batch mode)")
    p.add_argument("--output_dir", default="xai_results",
                   help="Root output directory (default: xai_results/)")
    p.add_argument("--face_bbox", nargs=4, type=int,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="Face bounding box in 224×224 coords (optional)")
    p.add_argument("--self_test", action="store_true",
                   help="Run smoke-test with a random tensor and exit")
    p.add_argument("--dataset_csv",
               help="CSV file containing relative video paths and labels")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # FIX 6 (self-test vs real evaluation confusion):
    # --self_test is explicit and separate from real evaluation.
    if args.self_test:
        _self_test(args.checkpoint)
        sys.exit(0)
    
        # ──────────────────────────────────────────────────────────────
    # DATASET CSV MODE (NEW)
    # ──────────────────────────────────────────────────────────────
    if args.dataset_csv:

        csv_path = str(Path(args.dataset_csv).resolve())

        if not Path(csv_path).exists():
            log.error(f"Dataset CSV not found: {csv_path}")
            sys.exit(1)

        sources = load_dataset_csv(csv_path)

        if not sources:
            log.error("No valid clips found in CSV.")
            sys.exit(1)

        batch_evaluate(
            checkpoint_path=args.checkpoint,
            sources=sources,
            output_dir=args.output_dir,
        )

        sys.exit(0)

    bbox    = tuple(args.face_bbox) if args.face_bbox else None
    sources: List[Dict] = []

    if args.frames_dir:
        # FIX 3: resolve path before using it
        frames_dir = str(Path(args.frames_dir).resolve())
        sources.append({
            "clip_id":    Path(frames_dir).name,
            "frames_dir": frames_dir,
            **({"face_bbox": bbox} if bbox else {}),
        })

    elif args.video_path:
        # FIX 3: resolve path
        video_path = str(Path(args.video_path).resolve())
        sources.append({
            "clip_id":    Path(video_path).stem,
            "video_path": video_path,
            **({"face_bbox": bbox} if bbox else {}),
        })

    elif args.video_dir:
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        video_dir  = Path(args.video_dir).resolve()
        for vp in sorted(video_dir.iterdir()):
            if vp.suffix.lower() in video_exts:
                sources.append({
                    "clip_id":    vp.stem,
                    "video_path": str(vp),          # FIX 3: use resolved absolute path
                    **({"face_bbox": bbox} if bbox else {}),
                })
        if not sources:
            log.error(f"No video files found in: {video_dir}")
            sys.exit(1)
        log.info(f"Found {len(sources)} video(s) in {video_dir}")

    else:
        log.error(
            "Provide one of: --video_path, --frames_dir, --video_dir, or --self_test"
        )
        sys.exit(1)

    batch_evaluate(
        checkpoint_path=args.checkpoint,
        sources=sources,
        output_dir=args.output_dir,
    )