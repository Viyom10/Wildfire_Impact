"""
MHEIDS Test Suite — Integration Tests

End-to-end tests that verify multi-module pipelines work together.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pyjson5
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────

def _synth_pair(C=9, H=64, W=64):
    rng = np.random.RandomState(42)
    before = rng.randint(100, 10000, (C, H, W)).astype(np.float32)
    after  = before.copy()
    after[:, 20:50, 20:50] *= 0.4   # simulate change
    return before, after


# ██ Full multi-hazard pipeline ██

class TestMultiHazardPipeline:

    def test_all_modules_to_composite(self):
        """Run all four hazard modules → composite engine → CompositeResult."""
        from hazard_wildfire import WildfireModule
        from hazard_drought import DroughtModule
        from hazard_rainfall import RainfallModule
        from hazard_vegetation import VegetationModule
        from composite_engine import CompositeEngine, CompositeResult

        before, after = _synth_pair()

        results = {
            "wildfire":   WildfireModule().analyze(before, after),
            "drought":    DroughtModule().analyze(before, after),
            "rainfall":   RainfallModule().analyze(before, after),
            "vegetation": VegetationModule().analyze(before, after),
        }

        engine = CompositeEngine()
        comp = engine.evaluate(results)

        assert isinstance(comp, CompositeResult)
        assert comp.pixel_cis.shape == (64, 64)
        assert 0 <= comp.region_cis <= 1
        assert comp.region_risk in ("Low", "Medium", "High", "Critical")
        assert len(comp.hazard_breakdown) == 4

    def test_partial_modules(self):
        """Run only 2 modules → composite should still work."""
        from hazard_wildfire import WildfireModule
        from hazard_vegetation import VegetationModule
        from composite_engine import CompositeEngine

        before, after = _synth_pair()
        results = {
            "wildfire":   WildfireModule().analyze(before, after),
            "vegetation": VegetationModule().analyze(before, after),
        }
        comp = CompositeEngine().evaluate(results)
        assert comp.pixel_cis.shape == (64, 64)
        # Weights should be re-normalised to 2 active hazards
        w = comp.metadata["weights"]
        assert abs(sum(w.values()) - 1.0) < 0.01


# ██ Config loading chain ██

class TestConfigChain:

    def test_main_config_loads(self):
        cfg_path = Path(__file__).resolve().parent.parent / 'configs' / 'config.json'
        if not cfg_path.exists():
            pytest.skip("config.json not found")
        cfg = pyjson5.load(open(cfg_path))
        assert "method" in cfg
        assert "datasets" in cfg

    def test_method_configs_exist(self):
        method_dir = Path(__file__).resolve().parent.parent / 'configs' / 'method'
        if not method_dir.exists():
            pytest.skip("method configs dir not found")
        jsons = list(method_dir.glob('*.json'))
        assert len(jsons) >= 1

    def test_all_method_configs_parse(self):
        method_dir = Path(__file__).resolve().parent.parent / 'configs' / 'method'
        if not method_dir.exists():
            pytest.skip("method configs dir not found")
        for f in method_dir.glob('*.json'):
            cfg = pyjson5.load(open(f))
            assert isinstance(cfg, dict), f"Failed to parse {f.name}"


# ██ Module result consistency ██

class TestResultConsistency:

    def test_all_results_same_shape(self):
        """All hazard modules should produce results with matching spatial dims."""
        from hazard_wildfire import WildfireModule
        from hazard_drought import DroughtModule
        from hazard_rainfall import RainfallModule
        from hazard_vegetation import VegetationModule

        before, after = _synth_pair(9, 48, 48)

        modules = [WildfireModule(), DroughtModule(),
                   RainfallModule(), VegetationModule()]

        shapes = set()
        for mod in modules:
            res = mod.analyze(before, after)
            shapes.add(res.binary_mask.shape)
            shapes.add(res.probability_map.shape)

        assert len(shapes) == 1, f"Shape mismatch across modules: {shapes}"

    def test_probability_range(self):
        """All probability maps should be in [0, 1]."""
        from hazard_wildfire import WildfireModule
        from hazard_drought import DroughtModule
        from hazard_rainfall import RainfallModule
        from hazard_vegetation import VegetationModule

        before, after = _synth_pair()

        for Mod in [WildfireModule, DroughtModule, RainfallModule, VegetationModule]:
            res = Mod().analyze(before, after)
            assert res.probability_map.min() >= 0.0, f"{Mod.name} prob < 0"
            assert res.probability_map.max() <= 1.0, f"{Mod.name} prob > 1"

    def test_confidence_range(self):
        """All confidence scores should be in [0, 1]."""
        from hazard_wildfire import WildfireModule
        from hazard_drought import DroughtModule
        from hazard_rainfall import RainfallModule
        from hazard_vegetation import VegetationModule

        before, after = _synth_pair()

        for Mod in [WildfireModule, DroughtModule, RainfallModule, VegetationModule]:
            res = Mod().analyze(before, after)
            assert 0 <= res.confidence <= 1, f"{Mod.name} confidence out of range"
