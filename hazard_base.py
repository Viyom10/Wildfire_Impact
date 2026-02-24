"""
MHEIDS — Hazard Module Base Interface

Defines the abstract base class and result dataclass that all hazard modules must implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class HazardResult:
    """Standardised output from every hazard module."""
    binary_mask: np.ndarray           # (H, W) — 0/1 classification
    probability_map: np.ndarray       # (H, W) — [0, 1] continuous
    confidence: float                 # module-level confidence  [0, 1]
    affected_area_pct: float          # percentage of pixels classified as positive
    metadata: Dict = field(default_factory=dict)  # indices, thresholds, etc.


class HazardModule(ABC):
    """Abstract interface that all hazard modules must implement."""

    name: str = "base"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def validate_inputs(self, before: np.ndarray, after: np.ndarray, **kwargs) -> bool:
        """
        Check whether the input data is suitable for analysis.

        Parameters
        ----------
        before : np.ndarray  (C, H, W) — pre-event image
        after  : np.ndarray  (C, H, W) — post-event image

        Returns True if valid, raises ValueError otherwise.
        """
        ...

    @abstractmethod
    def analyze(self, before: np.ndarray, after: np.ndarray, **kwargs) -> HazardResult:
        """
        Run hazard-specific analysis.

        Parameters
        ----------
        before : np.ndarray  (C, H, W)
        after  : np.ndarray  (C, H, W)
        kwargs : auxiliary data (rainfall grid, land cover mask, etc.)

        Returns
        -------
        HazardResult
        """
        ...

    def get_confidence(self, result: HazardResult) -> float:
        """Return the confidence score from the most recent analysis."""
        return result.confidence

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Element-wise a/b with zero-division protection."""
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(b != 0, a / b, fill)
        return np.nan_to_num(result, nan=fill)

    @staticmethod
    def _percentage(mask: np.ndarray) -> float:
        """Percentage of True/1 pixels in a binary mask."""
        return float(mask.sum()) / mask.size * 100.0 if mask.size > 0 else 0.0
