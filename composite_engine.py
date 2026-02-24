"""
MHEIDS — Multi-Factor Composite Impact Evaluation Engine

Combines results from multiple hazard modules into a single Composite
Impact Score (CIS) with confidence adjustment and risk categorisation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from hazard_base import HazardResult


# Default weights (must sum to 1.0)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "wildfire":   0.35,
    "drought":    0.25,
    "rainfall":   0.20,
    "vegetation": 0.20,
}

# Risk-level thresholds
RISK_LEVELS = [
    (0.20, "Low"),
    (0.45, "Medium"),
    (0.70, "High"),
    (1.01, "Critical"),
]

RISK_COLORS = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}


@dataclass
class CompositeResult:
    """Output of the composite evaluation engine."""
    pixel_cis:         np.ndarray          # (H, W) [0, 1]
    pixel_risk:        np.ndarray          # (H, W) codes 0–3
    region_cis:        float               # mean CIS for entire region
    region_risk:       str                 # dominant risk label
    risk_distribution: Dict[str, float]    # {Low: %, Medium: %, …}
    dominant_hazard:   str                 # name of highest-contributing hazard
    hazard_breakdown:  Dict[str, float]    # per-hazard weighted contribution
    confidence:        float               # composite confidence
    metadata:          Dict = field(default_factory=dict)


def _normalise_weights(weights: Dict[str, float],
                       active: List[str]) -> Dict[str, float]:
    """Keep only active hazards and normalise weights to sum to 1."""
    filtered = {k: weights.get(k, 0.0) for k in active}
    total = sum(filtered.values())
    if total == 0:
        equal = 1.0 / len(active) if active else 0.0
        return {k: equal for k in active}
    return {k: v / total for k, v in filtered.items()}


def _risk_label(cis: float) -> str:
    for threshold, label in RISK_LEVELS:
        if cis < threshold:
            return label
    return "Critical"


def _risk_code(cis_map: np.ndarray) -> np.ndarray:
    codes = np.zeros_like(cis_map, dtype=np.uint8)
    for i, (threshold, _) in enumerate(RISK_LEVELS):
        codes[cis_map >= threshold] = min(i + 1, 3)
    # fix: code 0 for < 0.20
    codes[cis_map < 0.20] = 0
    return codes


class CompositeEngine:
    """
    Combines HazardResult objects into a Composite Impact Score (CIS).

    Parameters
    ----------
    weights : dict mapping hazard name → float weight  (default: DEFAULT_WEIGHTS)
    use_confidence : if True, apply confidence-adjusted CIS formula
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 use_confidence: bool = True):
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.use_confidence = use_confidence

    def evaluate(self, results: Dict[str, HazardResult]) -> CompositeResult:
        """
        Compute composite impact score from one or more hazard results.

        Parameters
        ----------
        results : dict mapping hazard name (str) → HazardResult

        Returns
        -------
        CompositeResult
        """
        if not results:
            raise ValueError("At least one HazardResult is required.")

        # Determine spatial shape from the first result
        first = next(iter(results.values()))
        H, W = first.probability_map.shape

        # Normalise weights to active hazards only
        active = list(results.keys())
        w = _normalise_weights(self.weights, active)

        # Build pixel-level CIS
        if self.use_confidence:
            numerator   = np.zeros((H, W), dtype=np.float64)
            denominator = np.zeros((H, W), dtype=np.float64)
            for name, res in results.items():
                wc = w[name] * res.confidence
                numerator   += wc * res.probability_map.astype(np.float64)
                denominator += wc
            pixel_cis = np.where(denominator != 0,
                                 numerator / denominator,
                                 0.0).astype(np.float32)
        else:
            pixel_cis = np.zeros((H, W), dtype=np.float64)
            for name, res in results.items():
                pixel_cis += w[name] * res.probability_map.astype(np.float64)
            pixel_cis = pixel_cis.astype(np.float32)

        pixel_cis = np.clip(pixel_cis, 0.0, 1.0)

        # Risk map
        pixel_risk = _risk_code(pixel_cis)

        # Region-level stats
        region_cis = float(np.mean(pixel_cis))
        region_risk = _risk_label(region_cis)

        N = H * W
        risk_dist = {
            "Low":      float(np.sum(pixel_risk == 0)) / N * 100,
            "Medium":   float(np.sum(pixel_risk == 1)) / N * 100,
            "High":     float(np.sum(pixel_risk == 2)) / N * 100,
            "Critical": float(np.sum(pixel_risk == 3)) / N * 100,
        }

        # Per-hazard weighted contribution
        breakdown: Dict[str, float] = {}
        for name, res in results.items():
            breakdown[name] = float(w[name] * np.mean(res.probability_map))

        dominant = max(breakdown, key=breakdown.get) if breakdown else ""

        # Composite confidence = weighted mean of per-module confidence
        comp_conf = sum(w[n] * r.confidence for n, r in results.items())

        return CompositeResult(
            pixel_cis=pixel_cis,
            pixel_risk=pixel_risk,
            region_cis=region_cis,
            region_risk=region_risk,
            risk_distribution=risk_dist,
            dominant_hazard=dominant,
            hazard_breakdown=breakdown,
            confidence=float(comp_conf),
            metadata={"weights": w, "active_hazards": active,
                      "use_confidence": self.use_confidence},
        )
