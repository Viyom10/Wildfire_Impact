"""
MHEIDS Test Suite — Composite Evaluation Engine Tests
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from hazard_base import HazardResult
from composite_engine import CompositeEngine, CompositeResult, DEFAULT_WEIGHTS


# ── helpers ──────────────────────────────────────────────────────────

def _dummy_result(prob: float = 0.5, conf: float = 0.9,
                  shape: tuple = (32, 32)) -> HazardResult:
    """Create a uniform-probability HazardResult."""
    return HazardResult(
        binary_mask=(np.ones(shape, dtype=np.uint8) if prob >= 0.5
                     else np.zeros(shape, dtype=np.uint8)),
        probability_map=np.full(shape, prob, dtype=np.float32),
        confidence=conf,
        affected_area_pct=100.0 if prob >= 0.5 else 0.0,
    )


# ██ Tests ██

class TestCompositeEngine:

    def test_single_hazard(self):
        engine = CompositeEngine()
        res = engine.evaluate({"wildfire": _dummy_result(0.8, 0.95)})
        assert isinstance(res, CompositeResult)
        assert 0 <= res.region_cis <= 1
        assert res.dominant_hazard == "wildfire"

    def test_all_hazards(self):
        engine = CompositeEngine()
        results = {
            "wildfire":   _dummy_result(0.9, 0.95),
            "drought":    _dummy_result(0.6, 0.80),
            "rainfall":   _dummy_result(0.1, 0.60),
            "vegetation": _dummy_result(0.8, 0.90),
        }
        comp = engine.evaluate(results)
        assert comp.pixel_cis.shape == (32, 32)
        assert comp.pixel_risk.shape == (32, 32)
        # All four hazard names present in breakdown
        assert set(comp.hazard_breakdown.keys()) == set(results.keys())

    def test_risk_distribution_sums_to_100(self):
        engine = CompositeEngine()
        results = {
            "wildfire":   _dummy_result(0.9, 0.95),
            "drought":    _dummy_result(0.6, 0.80),
        }
        comp = engine.evaluate(results)
        total = sum(comp.risk_distribution.values())
        assert abs(total - 100.0) < 0.1

    def test_confidence_adjustment(self):
        """Low-confidence module should reduce its influence."""
        engine_adj = CompositeEngine(use_confidence=True)
        engine_no  = CompositeEngine(use_confidence=False)

        results = {
            "wildfire":   _dummy_result(0.9, 0.95),
            "rainfall":   _dummy_result(0.1, 0.10),  # very low confidence
        }
        comp_adj = engine_adj.evaluate(results)
        comp_no  = engine_no.evaluate(results)

        # With confidence adjustment, low-conf rainfall is de-weighted,
        # so CIS should be closer to the wildfire score
        assert comp_adj.region_cis > comp_no.region_cis

    def test_zero_probability_all(self):
        """All modules at 0 → CIS should be 0."""
        engine = CompositeEngine()
        results = {
            "wildfire": _dummy_result(0.0, 0.9),
            "drought":  _dummy_result(0.0, 0.9),
        }
        comp = engine.evaluate(results)
        assert comp.region_cis < 0.01
        assert comp.region_risk == "Low"

    def test_maximum_risk(self):
        """All modules at 1.0 → CIS should be ~1.0, risk = Critical."""
        engine = CompositeEngine()
        results = {
            "wildfire": _dummy_result(1.0, 1.0),
            "drought":  _dummy_result(1.0, 1.0),
        }
        comp = engine.evaluate(results)
        assert comp.region_cis > 0.95
        assert comp.region_risk == "Critical"

    def test_custom_weights(self):
        engine = CompositeEngine(weights={"wildfire": 1.0, "drought": 0.0})
        results = {
            "wildfire": _dummy_result(0.8, 0.9),
            "drought":  _dummy_result(0.2, 0.9),
        }
        comp = engine.evaluate(results)
        # drought weight = 0 → CIS ≈ wildfire prob
        assert abs(comp.region_cis - 0.8) < 0.05

    def test_empty_input_raises(self):
        engine = CompositeEngine()
        with pytest.raises(ValueError):
            engine.evaluate({})

    def test_pixel_risk_codes(self):
        engine = CompositeEngine()
        results = {"wildfire": _dummy_result(0.8, 1.0)}
        comp = engine.evaluate(results)
        # all pixels have same prob → same risk code
        unique = np.unique(comp.pixel_risk)
        assert len(unique) == 1

    def test_metadata_present(self):
        engine = CompositeEngine()
        results = {"wildfire": _dummy_result(0.5, 0.8)}
        comp = engine.evaluate(results)
        assert "weights" in comp.metadata
        assert "active_hazards" in comp.metadata

    def test_dominant_hazard_correct(self):
        engine = CompositeEngine(use_confidence=False)
        results = {
            "wildfire":   _dummy_result(0.3, 0.9),
            "vegetation": _dummy_result(0.9, 0.9),
        }
        comp = engine.evaluate(results)
        assert comp.dominant_hazard == "vegetation"
