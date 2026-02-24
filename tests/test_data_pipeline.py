"""
MHEIDS Test Suite — Data Pipeline Tests

Tests for image loading, channel adjustment, preprocessing, and NaN handling.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import torch
from pathlib import Path


# ── Channel adjustment (from predict_from_images.py) ─────────────────

class TestChannelAdjustment:

    def test_rgb_to_9band(self):
        from predict_from_images import adjust_channels
        rgb = np.random.rand(3, 64, 64).astype(np.float32)
        result = adjust_channels(rgb, 9)
        assert result.shape == (9, 64, 64)
        assert np.all(np.isfinite(result))

    def test_identity(self):
        from predict_from_images import adjust_channels
        img = np.random.rand(9, 64, 64).astype(np.float32)
        result = adjust_channels(img, 9)
        np.testing.assert_array_equal(result, img)

    def test_truncate(self):
        from predict_from_images import adjust_channels
        img = np.random.rand(12, 64, 64).astype(np.float32)
        result = adjust_channels(img, 9)
        assert result.shape[0] == 9

    def test_replicate_channels(self):
        from predict_from_images import adjust_channels
        img = np.random.rand(4, 64, 64).astype(np.float32)
        result = adjust_channels(img, 9)
        assert result.shape[0] == 9


# ── Preprocessing ────────────────────────────────────────────────────

class TestPreprocessing:

    def test_clamp_scale(self):
        from predict_from_images import preprocess_image
        img = np.array([[[15000, 500], [5000, 0]]], dtype=np.float32)
        t = preprocess_image(img, 'clamp_scale_10000')
        assert t.max() <= 1.0
        assert t.ndim == 4          # batch dim added

    def test_minmax(self):
        from predict_from_images import preprocess_image
        img = np.random.rand(9, 16, 16).astype(np.float32) * 5000
        t = preprocess_image(img, 'min-max')
        assert t.min() >= -0.01
        assert t.max() <= 1.01

    def test_batch_dim(self):
        from predict_from_images import preprocess_image
        img = np.random.rand(9, 16, 16).astype(np.float32)
        t = preprocess_image(img, 'clamp_scale_10000')
        assert t.shape[0] == 1


# ── NaN handling ─────────────────────────────────────────────────────

class TestNaNHandling:

    def test_nan_replaced(self):
        """NaN pixels should be replaced with 0 in hazard modules."""
        from hazard_wildfire import WildfireModule
        before = np.random.rand(9, 32, 32).astype(np.float32)
        after  = before.copy()
        after[0, 10:15, 10:15] = np.nan
        # Module should not crash on NaN input
        mod = WildfireModule()
        res = mod.analyze(before, after)
        assert np.all(np.isfinite(res.probability_map))

    def test_all_nan_image(self):
        """Entirely NaN image should be handled gracefully."""
        from hazard_vegetation import VegetationModule
        nan_img = np.full((9, 32, 32), np.nan, dtype=np.float32)
        mod = VegetationModule()
        res = mod.analyze(nan_img, nan_img)
        # Should not crash; may return zeros
        assert res.binary_mask.shape == (32, 32)


# ── Synthetic dataset ────────────────────────────────────────────────

class TestSyntheticDataset:

    def test_synthetic_files_exist(self):
        """Check that synthetic data was previously created."""
        base = Path(__file__).resolve().parent.parent / 'data' / 'processed' / '2024'
        # skip if no synthetic data present
        if not base.exists():
            pytest.skip("Synthetic dataset not generated yet.")
        npy_files = list(base.glob('*.npy'))
        assert len(npy_files) > 0

    def test_synthetic_patch_shapes(self):
        base = Path(__file__).resolve().parent.parent / 'data' / 'processed' / '2024'
        if not base.exists():
            pytest.skip("Synthetic dataset not generated yet.")
        for p in list(base.glob('*.S2_before.npy'))[:2]:
            arr = np.load(p)
            assert arr.ndim == 3
            assert arr.shape[0] == 9
            assert arr.shape[1] == 256
