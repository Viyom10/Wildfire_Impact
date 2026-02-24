"""
MHEIDS — Drought Detection Module

Wraps existing drought change-detection inference and adds ΔNDVI / NDII
index-based analysis conforming to the HazardModule interface.
"""
import numpy as np
import torch
from typing import Optional

from hazard_base import HazardModule, HazardResult


# Sentinel-2 20 m band mapping (0-indexed in the 9-band stack)
_BAND = {"B02": 0, "B03": 1, "B04": 2, "B05": 3, "B06": 4,
         "B07": 5, "B11": 6, "B12": 7, "B8A": 8}

# Drought severity thresholds (ΔNDVI-based)
# "None" cutoff must match binary mask threshold (-0.10)
_SEVERITY = [
    (-0.10, "None"),      # ≥ -0.10 → no drought
    (-0.20, "Mild"),      # -0.10 to -0.20
    (-0.30, "Moderate"),  # -0.20 to -0.30
    (-0.45, "Severe"),    # -0.30 to -0.45
    (float("-inf"), "Extreme"),  # < -0.45
]


def _compute_ndvi(img: np.ndarray) -> np.ndarray:
    """NDVI = (NIR – Red) / (NIR + Red)."""
    nir = img[_BAND["B8A"]].astype(np.float64)
    red = img[_BAND["B04"]].astype(np.float64)
    denom = nir + red
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom != 0, (nir - red) / denom, 0.0).astype(np.float32)


def _compute_ndii(img: np.ndarray) -> np.ndarray:
    """NDII = (NIR – SWIR) / (NIR + SWIR)  (canopy water content)."""
    nir  = img[_BAND["B8A"]].astype(np.float64)
    swir = img[_BAND["B11"]].astype(np.float64)
    denom = nir + swir
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom != 0, (nir - swir) / denom, 0.0).astype(np.float32)


def _drought_severity(delta_ndvi: np.ndarray) -> np.ndarray:
    """Map ΔNDVI to severity codes 0–4.
    0=None, 1=Mild, 2=Moderate, 3=Severe, 4=Extreme
    """
    sev = np.zeros_like(delta_ndvi, dtype=np.uint8)
    # Apply from most severe → least severe so milder levels don't overwrite
    sev[delta_ndvi < -0.45] = 4  # Extreme
    sev[(delta_ndvi >= -0.45) & (delta_ndvi < -0.30)] = 3  # Severe
    sev[(delta_ndvi >= -0.30) & (delta_ndvi < -0.20)] = 2  # Moderate
    sev[(delta_ndvi >= -0.20) & (delta_ndvi < -0.10)] = 1  # Mild
    # sev stays 0 for delta_ndvi >= -0.10 (None)
    return sev


class DroughtModule(HazardModule):
    """Drought / vegetation-stress detection."""

    name = "drought"

    def __init__(self, model=None, device: Optional[str] = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- interface --------------------------------------------------

    def validate_inputs(self, before, after, **kw):
        if before.ndim != 3 or after.ndim != 3:
            raise ValueError("Expected 3-D arrays (C, H, W).")
        if before.shape != after.shape:
            raise ValueError(f"Shape mismatch: {before.shape} vs {after.shape}")
        if before.shape[0] < 3:
            raise ValueError("Need at least 3 bands.")
        return True

    def analyze(self, before, after, **kw) -> HazardResult:
        self.validate_inputs(before, after)

        if self.model is not None and before.shape[0] == 9:
            return self._analyze_ml(before, after)
        else:
            return self._analyze_index(before, after)

    # ---- private ----------------------------------------------------

    def _analyze_ml(self, before, after) -> HazardResult:
        dev = self.device
        b = torch.from_numpy(before).unsqueeze(0).float().to(dev)
        a = torch.from_numpy(after).unsqueeze(0).float().to(dev)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(b, a)
            probs  = torch.softmax(logits, dim=1)
            prob_map  = probs[0, 1].cpu().numpy()
            pred_mask = (prob_map >= 0.5).astype(np.uint8)

        conf = float(np.mean(np.maximum(prob_map, 1 - prob_map)))

        # Compute index-based metadata anyway
        ndvi_change = _compute_ndvi(after) - _compute_ndvi(before)
        severity = _drought_severity(ndvi_change)

        return HazardResult(
            binary_mask=pred_mask,
            probability_map=prob_map,
            confidence=conf,
            affected_area_pct=self._percentage(pred_mask),
            metadata={
                "method": "ml_siamese",
                "ndvi_change": ndvi_change,
                "stress_severity": severity,
            },
        )

    def _analyze_index(self, before, after) -> HazardResult:
        """Fallback: ΔNDVI thresholding (or RGB greenness proxy)."""
        if before.shape[0] >= 9:
            ndvi_pre  = _compute_ndvi(before)
            ndvi_post = _compute_ndvi(after)
            ndii_pre  = _compute_ndii(before)
            ndii_post = _compute_ndii(after)
            delta_ndvi = ndvi_post - ndvi_pre
            method = "index_ndvi"
            confidence = 0.55
        else:
            # ── RGB fallback: Excess Green Index + brightness analysis ──
            # ExG = 2*G - R - B  (normalised) — captures vegetation greenness
            def _exg(img):
                total = img.sum(axis=0).astype(np.float64) + 1e-8
                r = img[0].astype(np.float64) / total
                g = img[1].astype(np.float64) / total
                b = img[2].astype(np.float64) / total
                return (2.0 * g - r - b).astype(np.float32)

            exg_pre  = _exg(before)
            exg_post = _exg(after)
            delta_exg = exg_post - exg_pre          # negative = greening loss

            # Brightness change (drought → land dries → brighter)
            bright_pre  = before.mean(axis=0).astype(np.float32)
            bright_post = after.mean(axis=0).astype(np.float32)
            bright_max  = np.maximum(bright_pre.max(), bright_post.max()) + 1e-8
            delta_bright = (bright_post - bright_pre) / bright_max  # positive = brighter

            # Combine: loss of green + brightening → drought signal
            # Scale delta_exg to roughly match NDVI range for thresholds
            delta_ndvi = (delta_exg * 2.5 - delta_bright * 0.5).astype(np.float32)
            ndvi_pre = exg_pre    # store for metadata
            ndvi_post = exg_post
            ndii_pre = ndii_post = None
            method = "index_rgb_greenness"
            confidence = 0.45     # lower confidence for RGB proxy

        # Detection threshold aligned with severity "None" cutoff
        # Only flag pixels that have meaningful greenness loss
        threshold = -0.10   # matches _SEVERITY[0] = "None" cutoff
        pred_mask = (delta_ndvi < threshold).astype(np.uint8)
        # Probability: ramp from threshold to -0.40 (scaled 0→1)
        prob_map  = np.clip((-delta_ndvi - 0.10) / 0.30, 0, 1).astype(np.float32)
        prob_map  *= pred_mask  # zero out sub-threshold pixels
        severity  = _drought_severity(delta_ndvi)

        meta = {"method": method, "ndvi_change": delta_ndvi,
                "stress_severity": severity}
        if before.shape[0] >= 9:
            meta["ndii_change"] = ndii_post - ndii_pre

        return HazardResult(
            binary_mask=pred_mask,
            probability_map=prob_map,
            confidence=confidence,
            affected_area_pct=self._percentage(pred_mask),
            metadata=meta,
        )
