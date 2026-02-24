"""
MHEIDS — Rainfall Impact Analysis Module

Non-ML module that uses NDWI / MNDWI spectral indices to detect
water-extent changes and cross-validates with optional precipitation data.
"""
import numpy as np
from typing import Optional

from hazard_base import HazardModule, HazardResult


# Sentinel-2 20 m band mapping
_BAND = {"B02": 0, "B03": 1, "B04": 2, "B05": 3, "B06": 4,
         "B07": 5, "B11": 6, "B12": 7, "B8A": 8}


def _compute_ndwi(img: np.ndarray) -> np.ndarray:
    """NDWI = (Green – NIR) / (Green + NIR)."""
    green = img[_BAND["B03"]].astype(np.float64)
    nir   = img[_BAND["B8A"]].astype(np.float64)
    denom = green + nir
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom != 0, (green - nir) / denom, 0.0).astype(np.float32)


def _compute_mndwi(img: np.ndarray) -> np.ndarray:
    """MNDWI = (Green – SWIR) / (Green + SWIR)."""
    green = img[_BAND["B03"]].astype(np.float64)
    swir  = img[_BAND["B11"]].astype(np.float64)
    denom = green + swir
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom != 0, (green - swir) / denom, 0.0).astype(np.float32)


class RainfallModule(HazardModule):
    """Rainfall-impact detection via spectral water indices."""

    name = "rainfall"

    # -- interface ----------------------------------------------------

    def validate_inputs(self, before, after, **kw):
        if before.ndim != 3 or after.ndim != 3:
            raise ValueError("Expected 3-D arrays (C, H, W).")
        if before.shape != after.shape:
            raise ValueError(f"Shape mismatch: {before.shape} vs {after.shape}")
        if before.shape[0] < 3:
            raise ValueError("Need at least 3 bands.")
        return True

    def analyze(self, before, after, **kw) -> HazardResult:
        """
        Parameters (kwargs)
        ----------
        rainfall_grid         : np.ndarray (H, W) — accumulated precip (mm)
        historical_avg_rainfall : float           — long-term average (mm)
        """
        self.validate_inputs(before, after)

        rainfall_grid = kw.get("rainfall_grid", None)
        historical_avg = kw.get("historical_avg_rainfall", None)

        have_full_bands = before.shape[0] >= 9

        # 1. Spectral index computation
        if have_full_bands:
            ndwi_pre  = _compute_ndwi(before)
            ndwi_post = _compute_ndwi(after)
            mndwi_pre  = _compute_mndwi(before)
            mndwi_post = _compute_mndwi(after)
        else:
            # rough proxy using available bands
            ndwi_pre = self._safe_divide(
                before[1].astype(np.float32) - before[-1].astype(np.float32),
                before[1].astype(np.float32) + before[-1].astype(np.float32))
            ndwi_post = self._safe_divide(
                after[1].astype(np.float32) - after[-1].astype(np.float32),
                after[1].astype(np.float32) + after[-1].astype(np.float32))
            mndwi_pre = mndwi_post = None

        delta_ndwi = ndwi_post - ndwi_pre

        # 2. Classification
        # +1 = new water / flooding,  0 = no change,  -1 = water loss
        water_change = np.zeros_like(delta_ndwi, dtype=np.int8)
        water_change[delta_ndwi > 0.3]  =  1
        water_change[delta_ndwi < -0.3] = -1

        # Binary mask (any significant impact)
        threshold = 0.15
        pred_mask = (np.abs(delta_ndwi) > threshold).astype(np.uint8)

        # Probability map: ramp from threshold to 0.50 (scaled 0→1)
        prob_map = np.clip((np.abs(delta_ndwi) - threshold) / 0.35, 0, 1).astype(np.float32)
        prob_map *= pred_mask  # zero out sub-threshold pixels

        # 3. Confidence — base is moderate; boosted by rainfall cross-validation
        base_confidence = 0.55 if have_full_bands else 0.35

        if rainfall_grid is not None and historical_avg is not None and historical_avg > 0:
            anomaly = rainfall_grid / historical_avg
            mean_anomaly = float(np.mean(anomaly))
            # If anomaly confirms spectral signal, raise confidence
            if (mean_anomaly > 1.5 and float(delta_ndwi.mean()) > 0) or \
               (mean_anomaly < 0.5 and float(delta_ndwi.mean()) < 0):
                base_confidence = min(base_confidence + 0.25, 0.95)
            elif (mean_anomaly > 1.5 and float(delta_ndwi.mean()) < 0) or \
                 (mean_anomaly < 0.5 and float(delta_ndwi.mean()) > 0):
                # Conflicting signals
                base_confidence *= 0.6
        elif rainfall_grid is None:
            base_confidence *= 0.9   # no aux data → small penalty

        # 4. Impact categories
        impact_cat = np.full(delta_ndwi.shape, "Normal", dtype=object)
        impact_cat[delta_ndwi < -0.15] = "Deficit"
        impact_cat[delta_ndwi > 0.15]  = "Slight excess"
        impact_cat[delta_ndwi > 0.35]  = "Flood risk"

        return HazardResult(
            binary_mask=pred_mask,
            probability_map=prob_map,
            confidence=float(base_confidence),
            affected_area_pct=self._percentage(pred_mask),
            metadata={
                "method": "index_ndwi",
                "ndwi_change": delta_ndwi,
                "mndwi_change": (mndwi_post - mndwi_pre) if mndwi_post is not None else None,
                "water_change_mask": water_change,
                "impact_category": impact_cat,
            },
        )
