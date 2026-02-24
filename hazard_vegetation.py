"""
MHEIDS — Vegetation Health Monitoring Module

Hybrid module: computes NDVI, EVI, SAVI temporal differences and
classifies vegetation health into five categories.
"""
import numpy as np
from hazard_base import HazardModule, HazardResult


# Sentinel-2 20 m band mapping
_BAND = {"B02": 0, "B03": 1, "B04": 2, "B05": 3, "B06": 4,
         "B07": 5, "B11": 6, "B12": 7, "B8A": 8}

# Health classification thresholds (ΔNDVI)
_HEALTH = [
    (-0.25, "Severe degradation"),
    (-0.10, "Degradation"),
    ( 0.10, "Stable"),
    ( 0.25, "Recovery"),
    (float("inf"), "Strong recovery"),
]


def _ndvi(img: np.ndarray) -> np.ndarray:
    nir = img[_BAND["B8A"]].astype(np.float64)
    red = img[_BAND["B04"]].astype(np.float64)
    d = nir + red
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(d != 0, (nir - red) / d, 0.0).astype(np.float32)


def _evi(img: np.ndarray) -> np.ndarray:
    """EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)."""
    nir  = img[_BAND["B8A"]].astype(np.float64)
    red  = img[_BAND["B04"]].astype(np.float64)
    blue = img[_BAND["B02"]].astype(np.float64)
    d = nir + 6.0 * red - 7.5 * blue + 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(d != 0, 2.5 * (nir - red) / d, 0.0).astype(np.float32)


def _savi(img: np.ndarray, L: float = 0.5) -> np.ndarray:
    """SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)."""
    nir = img[_BAND["B8A"]].astype(np.float64)
    red = img[_BAND["B04"]].astype(np.float64)
    d = nir + red + L
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(d != 0, (nir - red) / d * (1 + L), 0.0).astype(np.float32)


def _health_category(delta_ndvi: np.ndarray) -> np.ndarray:
    """Classify ΔNDVI into 5 health categories (codes 0–4)."""
    cat = np.full(delta_ndvi.shape, 2, dtype=np.uint8)   # default = Stable
    cat[delta_ndvi < -0.25] = 0  # Severe degradation
    cat[(delta_ndvi >= -0.25) & (delta_ndvi < -0.10)] = 1  # Degradation
    cat[(delta_ndvi >= 0.10)  & (delta_ndvi < 0.25)]  = 3  # Recovery
    cat[delta_ndvi >= 0.25]  = 4  # Strong recovery
    return cat


class VegetationModule(HazardModule):
    """Vegetation health monitoring via spectral index differencing."""

    name = "vegetation"

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
        self.validate_inputs(before, after)

        have_full = before.shape[0] >= 9

        # ---- compute indices ----------------------------------------
        if have_full:
            ndvi_pre, ndvi_post = _ndvi(before), _ndvi(after)
            evi_pre,  evi_post  = _evi(before),  _evi(after)
            savi_pre, savi_post = _savi(before), _savi(after)
        else:
            # minimal proxy (R=0, G=1, B=2 as bands 4, 3, 2)
            ndvi_pre = self._safe_divide(
                before[-1].astype(np.float32) - before[0].astype(np.float32),
                before[-1].astype(np.float32) + before[0].astype(np.float32))
            ndvi_post = self._safe_divide(
                after[-1].astype(np.float32) - after[0].astype(np.float32),
                after[-1].astype(np.float32) + after[0].astype(np.float32))
            evi_pre = evi_post = None
            savi_pre = savi_post = None

        delta_ndvi = ndvi_post - ndvi_pre

        # ---- classification -----------------------------------------
        health_cat = _health_category(delta_ndvi)

        # Binary mask: degradation = 1 (threshold aligned with health categories)
        threshold = -0.10  # matches "Degradation" cutoff in _HEALTH
        pred_mask = (delta_ndvi < threshold).astype(np.uint8)

        # Probability map: ramp from threshold to -0.45 (scaled 0→1)
        prob_map = np.clip((-delta_ndvi - 0.10) / 0.35, 0, 1).astype(np.float32)
        prob_map *= pred_mask  # zero out sub-threshold pixels

        # Confidence: higher if EVI/SAVI corroborate NDVI
        conf = 0.60 if have_full else 0.40
        if evi_pre is not None:
            delta_evi = evi_post - evi_pre
            agreement = np.mean(np.sign(delta_ndvi) == np.sign(delta_evi))
            conf = min(conf + 0.2 * agreement, 0.95)

        meta = {
            "method": "index_ndvi_evi_savi",
            "ndvi_before": ndvi_pre,
            "ndvi_after": ndvi_post,
            "delta_ndvi": delta_ndvi,
            "health_category": health_cat,
        }
        if evi_pre is not None:
            meta["evi_change"]  = evi_post  - evi_pre
        if savi_pre is not None:
            meta["savi_change"] = savi_post - savi_pre

        return HazardResult(
            binary_mask=pred_mask,
            probability_map=prob_map,
            confidence=float(conf),
            affected_area_pct=self._percentage(pred_mask),
            metadata=meta,
        )
