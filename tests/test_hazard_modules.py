"""
MHEIDS Test Suite — Hazard Module Tests

Tests for all four hazard modules: wildfire, drought, rainfall, vegetation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from hazard_base import HazardModule, HazardResult
from hazard_wildfire import WildfireModule
from hazard_drought import DroughtModule
from hazard_rainfall import RainfallModule
from hazard_vegetation import VegetationModule


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def img_9band():
    """Random 9-band 64×64 image pair."""
    rng = np.random.RandomState(42)
    before = rng.randint(0, 10000, (9, 64, 64)).astype(np.float32)
    after  = before.copy()
    # introduce change in a region
    after[:, 16:48, 16:48] *= 0.5
    return before, after

@pytest.fixture
def img_3band():
    """RGB 3-band image pair."""
    rng = np.random.RandomState(42)
    before = rng.randint(0, 255, (3, 64, 64)).astype(np.float32)
    after  = before.copy()
    after[:, 16:48, 16:48] *= 0.5
    return before, after

@pytest.fixture
def img_identical():
    """Two identical images (no change)."""
    rng = np.random.RandomState(42)
    img = rng.randint(100, 5000, (9, 64, 64)).astype(np.float32)
    return img.copy(), img.copy()

@pytest.fixture
def img_zeros():
    """All-zero images."""
    z = np.zeros((9, 64, 64), dtype=np.float32)
    return z.copy(), z.copy()


# ██ HazardResult dataclass ██

class TestHazardResult:
    def test_creation(self):
        r = HazardResult(
            binary_mask=np.zeros((4, 4), dtype=np.uint8),
            probability_map=np.zeros((4, 4), dtype=np.float32),
            confidence=0.9,
            affected_area_pct=0.0,
        )
        assert r.confidence == 0.9
        assert r.affected_area_pct == 0.0
        assert isinstance(r.metadata, dict)


# ██ Wildfire Module ██

class TestWildfireModule:

    def test_analyze_produces_result(self, img_9band):
        mod = WildfireModule()   # no model → index fallback
        res = mod.analyze(*img_9band)
        assert isinstance(res, HazardResult)
        assert res.binary_mask.shape == (64, 64)
        assert res.probability_map.shape == (64, 64)
        assert 0 <= res.confidence <= 1

    def test_3band_fallback(self, img_3band):
        mod = WildfireModule()
        res = mod.analyze(*img_3band)
        assert res.binary_mask.shape == (64, 64)

    def test_no_change_scenario(self, img_identical):
        mod = WildfireModule()
        res = mod.analyze(*img_identical)
        # identical images → little to no detected change
        assert res.affected_area_pct < 10.0

    def test_all_zero(self, img_zeros):
        mod = WildfireModule()
        res = mod.analyze(*img_zeros)
        assert res.binary_mask.shape == (64, 64)
        # all-zero → no valid change
        assert res.affected_area_pct == 0.0

    def test_invalid_dims(self):
        mod = WildfireModule()
        with pytest.raises(ValueError):
            mod.analyze(np.zeros((64, 64)), np.zeros((64, 64)))

    def test_shape_mismatch(self):
        mod = WildfireModule()
        with pytest.raises(ValueError):
            mod.analyze(np.zeros((9, 64, 64)), np.zeros((9, 32, 32)))

    def test_metadata_has_method(self, img_9band):
        mod = WildfireModule()
        res = mod.analyze(*img_9band)
        assert "method" in res.metadata
        assert res.metadata["method"] == "index_dnbr"

    def test_severity_in_metadata(self, img_9band):
        mod = WildfireModule()
        res = mod.analyze(*img_9band)
        assert "burn_severity" in res.metadata
        assert res.metadata["burn_severity"].shape == (64, 64)


# ██ Drought Module ██

class TestDroughtModule:

    def test_analyze_basic(self, img_9band):
        mod = DroughtModule()
        res = mod.analyze(*img_9band)
        assert isinstance(res, HazardResult)
        assert res.probability_map.min() >= 0
        assert res.probability_map.max() <= 1

    def test_3band_fallback(self, img_3band):
        mod = DroughtModule()
        res = mod.analyze(*img_3band)
        assert res.binary_mask.shape == (64, 64)

    def test_no_change(self, img_identical):
        mod = DroughtModule()
        res = mod.analyze(*img_identical)
        assert res.affected_area_pct < 5.0

    def test_all_zero(self, img_zeros):
        mod = DroughtModule()
        res = mod.analyze(*img_zeros)
        assert res.affected_area_pct == 0.0

    def test_severity_classification(self, img_9band):
        mod = DroughtModule()
        res = mod.analyze(*img_9band)
        assert "stress_severity" in res.metadata


# ██ Rainfall Module ██

class TestRainfallModule:

    def test_analyze_basic(self, img_9band):
        mod = RainfallModule()
        res = mod.analyze(*img_9band)
        assert isinstance(res, HazardResult)
        assert 0 <= res.confidence <= 1

    def test_with_rainfall_grid(self, img_9band):
        before, after = img_9band
        grid = np.full((64, 64), 200.0)  # heavy rain
        mod = RainfallModule()
        res = mod.analyze(before, after, rainfall_grid=grid,
                          historical_avg_rainfall=100.0)
        assert "ndwi_change" in res.metadata

    def test_without_aux_data(self, img_9band):
        mod = RainfallModule()
        res = mod.analyze(*img_9band)
        # should still work, just with lower confidence
        assert 0 <= res.confidence <= 1

    def test_3band(self, img_3band):
        mod = RainfallModule()
        res = mod.analyze(*img_3band)
        assert res.binary_mask.shape == (64, 64)

    def test_no_change(self, img_identical):
        mod = RainfallModule()
        res = mod.analyze(*img_identical)
        assert res.affected_area_pct < 5.0

    def test_water_change_mask(self, img_9band):
        mod = RainfallModule()
        res = mod.analyze(*img_9band)
        wcm = res.metadata.get("water_change_mask")
        assert wcm is not None
        assert set(np.unique(wcm)).issubset({-1, 0, 1})


# ██ Vegetation Module ██

class TestVegetationModule:

    def test_analyze_basic(self, img_9band):
        mod = VegetationModule()
        res = mod.analyze(*img_9band)
        assert isinstance(res, HazardResult)
        assert "delta_ndvi" in res.metadata

    def test_health_categories(self, img_9band):
        mod = VegetationModule()
        res = mod.analyze(*img_9band)
        hc = res.metadata["health_category"]
        assert hc.shape == (64, 64)
        assert set(np.unique(hc)).issubset({0, 1, 2, 3, 4})

    def test_evi_savi_computed(self, img_9band):
        mod = VegetationModule()
        res = mod.analyze(*img_9band)
        assert "evi_change" in res.metadata
        assert "savi_change" in res.metadata

    def test_3band(self, img_3band):
        mod = VegetationModule()
        res = mod.analyze(*img_3band)
        assert res.binary_mask.shape == (64, 64)
        # EVI/SAVI not available with 3 bands
        assert "evi_change" not in res.metadata

    def test_no_change(self, img_identical):
        mod = VegetationModule()
        res = mod.analyze(*img_identical)
        assert res.affected_area_pct < 5.0

    def test_all_zero(self, img_zeros):
        mod = VegetationModule()
        res = mod.analyze(*img_zeros)
        assert res.affected_area_pct == 0.0
