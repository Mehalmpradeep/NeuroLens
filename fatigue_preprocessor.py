
"""
FatiguePreprocessor — Modular spatial biasing for fatigue-relevant regions.

Design contract
---------------
- Input : 224×224 RGB numpy uint8 array  +  optional MediaPipe landmark list
- Output: 224×224 RGB numpy uint8 array  (same dtype, shape, channel order)
- Deterministic per frame (no randomness)
- All blend alphas are bounded to ≤ 0.15 to stay subtle
- If landmarks are missing every stage that needs them is silently skipped
- No blocking I/O, no heavy loops, vectorised NumPy / OpenCV only
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe 468-point face mesh)
# ---------------------------------------------------------------------------
# Outer corners of left eye (from viewer's perspective)
_LEFT_EYE_OUTER  = 33
_LEFT_EYE_INNER  = 133
# Outer corners of right eye
_RIGHT_EYE_OUTER = 263
_RIGHT_EYE_INNER = 362
# Top / bottom of each eye (vertical extent)
_LEFT_EYE_TOP    = 159
_LEFT_EYE_BOT    = 145
_RIGHT_EYE_TOP   = 386
_RIGHT_EYE_BOT   = 374


class FatiguePreprocessor:
    """
    Applies three optional, toggleable preprocessing steps to a cropped
    224×224 face frame before it enters the VideoMAE clip buffer.

    Steps
    -----
    A. Vertical spatial weighting — softly de-emphasises the lower chin region.
    B. Eye-region Gaussian emphasis — mildly brightens the eye band.
    C. Lighting normalisation — very mild CLAHE on the luminance channel.

    All operations are bounded so that visual appearance barely changes;
    the intent is to nudge the model's spatial attention, not alter colour.
    """

    # ── tuneable constants ──────────────────────────────────────────────────
    # (A) Vertical weighting
    CHIN_BLEND_ALPHA   : float = 0.12   # fraction of frame height darkened at chin; max 0.15
    CHIN_FADE_FRACTION : float = 0.20   # bottom 20 % of the frame is affected

    # (B) Eye emphasis
    EYE_BLEND_ALPHA    : float = 0.10   # highlight strength; max 0.15
    EYE_PADDING_PX     : int   = 14     # extra pixels added above/below eye bbox
    EYE_SIGMA_RATIO    : float = 0.25   # Gaussian sigma as fraction of eye-band height

    # (C) CLAHE
    CLAHE_CLIP_LIMIT   : float = 1.5    # very gentle; default OpenCV is 2.0
    CLAHE_TILE         : int   = 8      # tile grid size

    # ── constructor ─────────────────────────────────────────────────────────
    def __init__(
        self,
        enable_vertical_weight : bool = True,
        enable_eye_emphasis     : bool = True,
        enable_lighting_norm    : bool = True,
    ) -> None:
        self.enable_vertical_weight = enable_vertical_weight
        self.enable_eye_emphasis    = enable_eye_emphasis
        self.enable_lighting_norm   = enable_lighting_norm

        # Pre-build the CLAHE object once (thread-safe read-only after init)
        self._clahe = cv2.createCLAHE(
            clipLimit   = self.CLAHE_CLIP_LIMIT,
            tileGridSize= (self.CLAHE_TILE, self.CLAHE_TILE),
        )

        # Pre-compute static vertical weight map (shape: 224×224×1, float32)
        self._vert_weight_map: np.ndarray = self._build_vertical_weight_map(224)

    # ── public API ──────────────────────────────────────────────────────────
    def process(
        self,
        frame    : np.ndarray,
        landmarks: Optional[Sequence] = None,
    ) -> np.ndarray:
        """
        Apply all enabled preprocessing steps.

        Parameters
        ----------
        frame     : 224×224 uint8 RGB array
        landmarks : MediaPipe face landmark list or None

        Returns
        -------
        224×224 uint8 RGB array
        """
        if frame is None or frame.shape != (224, 224, 3):
            return frame  # safety: pass through anything unexpected

        # Work in float32 to accumulate without repeated clipping
        out = frame.astype(np.float32)

        # ── Step A: vertical spatial weighting ──────────────────────────────
        if self.enable_vertical_weight:
            out = self._apply_vertical_weight(out)

        # ── Step B: eye-region emphasis ─────────────────────────────────────
        if self.enable_eye_emphasis and landmarks is not None:
            out = self._apply_eye_emphasis(out, landmarks)

        # ── Step C: lighting normalisation ──────────────────────────────────
        if self.enable_lighting_norm:
            out = self._apply_lighting_norm(out)

        return np.clip(out, 0, 255).astype(np.uint8)

    # ── private helpers ─────────────────────────────────────────────────────

    # --- (A) Vertical weighting -------------------------------------------

    def _build_vertical_weight_map(self, size: int) -> np.ndarray:
        """
        Build a (size, size, 1) float32 map in [1-alpha, 1].
        The top (1-CHIN_FADE_FRACTION) rows stay at weight 1.0.
        The bottom CHIN_FADE_FRACTION rows linearly fade toward (1-alpha).
        """
        weights = np.ones((size, 1), dtype=np.float32)
        fade_rows = int(size * self.CHIN_FADE_FRACTION)
        if fade_rows > 0:
            fade_start = size - fade_rows
            t = np.linspace(0.0, 1.0, fade_rows, dtype=np.float32).reshape(-1, 1)
            weights[fade_start:] = 1.0 - t * self.CHIN_BLEND_ALPHA
        # Broadcast to full width
        weight_map = np.tile(weights, (1, size)).reshape(size, size, 1)  # (H, W, 1)
        return weight_map

    def _apply_vertical_weight(self, frame_f: np.ndarray) -> np.ndarray:
        """Multiply frame by the static vertical weight map (broadcast over channels)."""
        return frame_f * self._vert_weight_map  # broadcasts (H,W,1) → (H,W,3)

    # --- (B) Eye emphasis ------------------------------------------------

    def _get_eye_band(self, landmarks: Sequence) -> tuple[int, int] | None:
        """
        Return (y_top, y_bot) pixel rows for a horizontal band enclosing
        both eyes, with padding.  Returns None if landmarks are too sparse.
        """
        try:
            ys = [
                landmarks[_LEFT_EYE_TOP ].y,
                landmarks[_LEFT_EYE_BOT ].y,
                landmarks[_RIGHT_EYE_TOP].y,
                landmarks[_RIGHT_EYE_BOT].y,
            ]
        except (IndexError, AttributeError):
            return None

        y_top = int(min(ys) * 224) - self.EYE_PADDING_PX
        y_bot = int(max(ys) * 224) + self.EYE_PADDING_PX
        y_top = max(0, y_top)
        y_bot = min(223, y_bot)

        if y_bot <= y_top:
            return None
        return y_top, y_bot

    def _apply_eye_emphasis(
        self,
        frame_f  : np.ndarray,
        landmarks: Sequence,
    ) -> np.ndarray:
        """
        Mildly brighten the eye band using a smooth Gaussian envelope
        blended at EYE_BLEND_ALPHA.  No hard edges anywhere.
        """
        band = self._get_eye_band(landmarks)
        if band is None:
            return frame_f

        y_top, y_bot = band
        band_h = y_bot - y_top

        # 1-D Gaussian profile along the vertical axis of the band
        sigma = max(1.0, band_h * self.EYE_SIGMA_RATIO)
        ys    = np.arange(band_h, dtype=np.float32) - band_h / 2.0
        gauss = np.exp(-0.5 * (ys / sigma) ** 2).reshape(-1, 1, 1)  # (band_h,1,1)

        # The highlight is a scaled version of the band itself, not a flat white
        band_slice = frame_f[y_top:y_bot]                # (band_h, 224, 3)
        highlight  = band_slice * gauss * self.EYE_BLEND_ALPHA

        out = frame_f.copy()
        out[y_top:y_bot] = band_slice + highlight
        return out

    # --- (C) Lighting normalisation --------------------------------------

    def _apply_lighting_norm(self, frame_f: np.ndarray) -> np.ndarray:
        """
        Apply gentle CLAHE to the L channel of LAB colour space.
        Converts float32 → uint8 → LAB → CLAHE → back.
        Blend at a fixed 0.15 to prevent flicker.
        """
        BLEND = 0.15  # never exceed this

        uint8_in = np.clip(frame_f, 0, 255).astype(np.uint8)

        lab = cv2.cvtColor(uint8_in, cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(lab)

        L_eq  = self._clahe.apply(L)
        lab_eq = cv2.merge([L_eq, a, b])
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

        # Soft blend: mostly original, tiny bit equalised → avoids flicker
        blended = (1.0 - BLEND) * frame_f + BLEND * rgb_eq.astype(np.float32)
        return blended