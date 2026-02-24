"""
MHEIDS — Wildfire Detection Module

Wraps the existing BAM-CD / siamese change-detection inference into the
standardised HazardModule interface.  When no trained model is available
the module falls back to dNBR thresholding.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from hazard_base import HazardModule, HazardResult


# Sentinel-2 20 m band mapping (0-indexed within the 9-band stack)
_BAND = {"B02": 0, "B03": 1, "B04": 2, "B05": 3, "B06": 4,
         "B07": 5, "B11": 6, "B12": 7, "B8A": 8}

# dNBR severity thresholds (USGS Key & Benson 2006)
# Aligned with binary mask threshold (0.10)
_SEVERITY = [
    (0.10, "Unburned"),     # < 0.10 → no burn
    (0.27, "Low"),          # 0.10 to 0.27
    (0.44, "Moderate-Low"), # 0.27 to 0.44
    (0.66, "Moderate-High"),# 0.44 to 0.66
    (float("inf"), "High"), # >= 0.66
]


def _compute_nbr(img: np.ndarray) -> np.ndarray:
    """NBR = (NIR – SWIR) / (NIR + SWIR)  using B8A and B12."""
    nir  = img[_BAND["B8A"]].astype(np.float64)
    swir = img[_BAND["B12"]].astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((nir + swir) != 0,
                        (nir - swir) / (nir + swir),
                        0.0).astype(np.float32)


def _dnbr_severity(dnbr: np.ndarray) -> np.ndarray:
    """Classify each pixel into a severity label (as int 0–4).
    0=Unburned, 1=Low, 2=Moderate-Low, 3=Moderate-High, 4=High
    """
    sev = np.zeros_like(dnbr, dtype=np.uint8)
    sev[(dnbr >= 0.10) & (dnbr < 0.27)] = 1  # Low
    sev[(dnbr >= 0.27) & (dnbr < 0.44)] = 2  # Moderate-Low
    sev[(dnbr >= 0.44) & (dnbr < 0.66)] = 3  # Moderate-High
    sev[dnbr >= 0.66] = 4                     # High
    # sev stays 0 for dnbr < 0.10 (Unburned)
    return sev


class WildfireModule(HazardModule):
    """Wildfire / burnt-area detection."""

    name = "wildfire"

    def __init__(self, model=None, device: Optional[str] = None):
        """
        Parameters
        ----------
        model  : a loaded PyTorch change-detection model  (optional)
        device : 'cuda' or 'cpu'
        """
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
        """Run siamese model inference."""
        dev = self.device
        b = torch.from_numpy(before).unsqueeze(0).float().to(dev)
        a = torch.from_numpy(after).unsqueeze(0).float().to(dev)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(b, a)
            probs = torch.softmax(logits, dim=1)
            prob_map = probs[0, 1].cpu().numpy()
            pred_mask = (prob_map >= 0.5).astype(np.uint8)

        conf = float(np.mean(np.maximum(prob_map, 1 - prob_map)))

        # Also compute dNBR for severity
        if before.shape[0] >= 9:
            nbr_pre = _compute_nbr(before)
            nbr_post = _compute_nbr(after)
            dnbr = nbr_pre - nbr_post
            severity = _dnbr_severity(dnbr)
        else:
            dnbr = None
            severity = None

        return HazardResult(
            binary_mask=pred_mask,
            probability_map=prob_map,
            confidence=conf,
            affected_area_pct=self._percentage(pred_mask),
            metadata={
                "method": "ml_siamese",
                "dnbr": dnbr,
                "burn_severity": severity,
            },
        )

    def _analyze_index(self, before, after) -> HazardResult:
        """Fallback: pure dNBR thresholding (or RGB char-index proxy)."""
        if before.shape[0] >= 9:
            nbr_pre  = _compute_nbr(before)
            nbr_post = _compute_nbr(after)
            dnbr = nbr_pre - nbr_post
            method = "index_dnbr"
            confidence = 0.60
        else:
            # ── RGB fallback: Char Index + redness + saturation analysis ──
            b_pre = before.astype(np.float64)
            b_aft = after.astype(np.float64)

            # Char Index: burned areas get darker; CI = (brightness_pre - brightness_post) / brightness_pre
            bright_pre  = b_pre.mean(axis=0) + 1e-8
            bright_post = b_aft.mean(axis=0) + 1e-8
            char_idx = ((bright_pre - bright_post) / bright_pre).astype(np.float32)

            # Redness increase: fire scars often have elevated red relative to green/blue
            def _redness(img):
                total = img.sum(axis=0) + 1e-8
                return (img[0] / total).astype(np.float32)   # R / (R+G+B)
            red_shift = _redness(b_aft) - _redness(b_pre)

            # Saturation loss: burned areas lose colour saturation
            def _saturation(img):
                mx = img.max(axis=0).astype(np.float64)
                mn = img.min(axis=0).astype(np.float64)
                return np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0.0).astype(np.float32)
            sat_loss = _saturation(b_pre) - _saturation(b_aft)

            # Combine signals → dNBR-like composite (positive = burn)
            dnbr = (char_idx * 0.5 + red_shift * 1.5 + sat_loss * 0.5).astype(np.float32)
            method = "index_rgb_char"
            confidence = 0.45

        # Detection threshold aligned with severity "Unburned" cutoff
        threshold = 0.10  # matches _SEVERITY[0] = "Unburned" cutoff
        pred_mask = (dnbr >= threshold).astype(np.uint8)
        # Probability: ramp from threshold to 0.50 (scaled 0→1)
        prob_map  = np.clip((dnbr - 0.10) / 0.40, 0, 1).astype(np.float32)
        prob_map  *= pred_mask  # zero out sub-threshold pixels
        severity  = _dnbr_severity(dnbr)

        return HazardResult(
            binary_mask=pred_mask,
            probability_map=prob_map,
            confidence=confidence,
            affected_area_pct=self._percentage(pred_mask),
            metadata={
                "method": method,
                "dnbr": dnbr,
                "burn_severity": severity,
            },
        )
