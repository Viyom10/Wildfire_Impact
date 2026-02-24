"""
MHEIDS Test Suite — Robustness & Edge Case Tests

Tests system behaviour under degraded inputs, noise, and boundary conditions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from hazard_wildfire import WildfireModule
from hazard_drought import DroughtModule
from hazard_rainfall import RainfallModule
from hazard_vegetation import VegetationModule
from composite_engine import CompositeEngine


# ── helpers ──────────────────────────────────────────────────────────

def _clean_pair(C=9, H=64, W=64):
    rng = np.random.RandomState(42)
    before = rng.randint(100, 10000, (C, H, W)).astype(np.float32)
    after  = before.copy()
    after[:, 20:50, 20:50] *= 0.4
    return before, after


# ██ Noise robustness ██

class TestNoiseRobustness:

    def test_gaussian_noise(self):
        """Moderate Gaussian noise should not crash the pipeline."""
        before, after = _clean_pair()
        noise = np.random.normal(0, 500, after.shape).astype(np.float32)
        after_noisy = after + noise

        for Mod in [WildfireModule, DroughtModule, RainfallModule, VegetationModule]:
            res = Mod().analyze(before, after_noisy)
            assert np.all(np.isfinite(res.probability_map))

    def test_heavy_noise(self):
        """Even heavy noise should produce finite outputs."""
        before, after = _clean_pair()
        noise = np.random.normal(0, 5000, after.shape).astype(np.float32)
        after_noisy = after + noise

        for Mod in [WildfireModule, DroughtModule, RainfallModule, VegetationModule]:
            res = Mod().analyze(before, after_noisy)
            assert np.all(np.isfinite(res.probability_map))


# ██ Missing bands ██

class TestMissingBands:

    def test_two_bands_zero(self):
        """Zeroing out 2 of 9 bands should still produce results."""
        before, after = _clean_pair()
        after[6:8] = 0  # zero SWIR bands

        for Mod in [WildfireModule, DroughtModule, RainfallModule, VegetationModule]:
            res = Mod().analyze(before, after)
            assert res.binary_mask.shape == (64, 64)

    def test_minimum_3_bands(self):
        """3-band input should work via fallback paths."""
        rng = np.random.RandomState(42)
        before = rng.rand(3, 64, 64).astype(np.float32) * 255
        after  = before.copy()
        after[:, 20:50, 20:50] *= 0.3

        for Mod in [WildfireModule, DroughtModule, RainfallModule, VegetationModule]:
            res = Mod().analyze(before, after)
            assert res.binary_mask.shape == (64, 64)


# ██ Edge cases ██

class TestEdgeCases:

    def test_identical_images(self):
        """No change between before/after → minimal or zero detection."""
        rng = np.random.RandomState(42)
        img = rng.randint(500, 5000, (9, 64, 64)).astype(np.float32)

        for Mod in [WildfireModule, DroughtModule, RainfallModule, VegetationModule]:
            res = Mod().analyze(img.copy(), img.copy())
            assert res.affected_area_pct < 10.0, f"{Mod.__name__} false-positive on identical"

    def test_all_zero(self):
        """All-zero images should not crash."""
        z = np.zeros((9, 64, 64), dtype=np.float32)
        for Mod in [WildfireModule, DroughtModule, RainfallModule, VegetationModule]:
            res = Mod().analyze(z.copy(), z.copy())
            assert res.binary_mask.shape == (64, 64)

    def test_single_pixel_anomaly(self):
        """A single hot pixel should not cause large detections."""
        rng = np.random.RandomState(42)
        before = rng.randint(500, 5000, (9, 64, 64)).astype(np.float32)
        after  = before.copy()
        after[:, 32, 32] = 65535  # single anomalous pixel

        mod = WildfireModule()
        res = mod.analyze(before, after)
        assert res.affected_area_pct < 5.0

    def test_very_small_region(self):
        """5×5 pixel change area should be detectable or not crash."""
        before, after = _clean_pair()
        after = before.copy()
        after[:, 30:35, 30:35] *= 0.1

        for Mod in [WildfireModule, VegetationModule]:
            res = Mod().analyze(before, after)
            assert res.binary_mask.shape == (64, 64)

    def test_saturated_image(self):
        """Saturated image (all 65535) should not crash."""
        sat = np.full((9, 64, 64), 65535, dtype=np.float32)
        norm = np.full((9, 64, 64), 5000, dtype=np.float32)

        for Mod in [WildfireModule, DroughtModule, RainfallModule, VegetationModule]:
            res = Mod().analyze(norm, sat)
            assert np.all(np.isfinite(res.probability_map))


# ██ Composite under degraded conditions ██

class TestCompositeRobustness:

    def test_composite_with_mixed_confidence(self):
        """Composite should handle modules with wildly different confidences."""
        from hazard_base import HazardResult

        results = {
            "wildfire":   HazardResult(
                binary_mask=np.ones((32, 32), dtype=np.uint8),
                probability_map=np.full((32, 32), 0.9, dtype=np.float32),
                confidence=0.99, affected_area_pct=100.0),
            "rainfall":   HazardResult(
                binary_mask=np.zeros((32, 32), dtype=np.uint8),
                probability_map=np.full((32, 32), 0.1, dtype=np.float32),
                confidence=0.05, affected_area_pct=0.0),
        }
        comp = CompositeEngine().evaluate(results)
        # High-confidence wildfire should dominate
        assert comp.region_cis > 0.5

    def test_composite_with_all_zero_probs(self):
        from hazard_base import HazardResult
        results = {
            "wildfire": HazardResult(
                binary_mask=np.zeros((32, 32), dtype=np.uint8),
                probability_map=np.zeros((32, 32), dtype=np.float32),
                confidence=0.9, affected_area_pct=0.0),
        }
        comp = CompositeEngine().evaluate(results)
        assert comp.region_cis < 0.01
