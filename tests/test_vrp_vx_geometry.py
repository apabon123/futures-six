"""
Tests for VRP VX leg geometry: Convergence = VX2−VX1 spread, Core/Alt = VX1 only.

Validates:
- Convergence emits exactly VX1 and VX2 with opposite signs (spread).
- Core and Alt emit VX1 only.
- VX2/VX3 survive combine and reindex when emitted.

No DuckDB required; uses injected signal caches where needed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.strat_combined import CombinedStrategy, VRP_SLEEVE_NAMES
from src.agents.strat_vrp_convergence import VRPConvergenceMeta
from src.agents.strat_vrp_core import VRPCoreMeta
from src.agents.strat_vrp_alt import VRPAltMeta


def _mock_market(universe=None):
    if universe is None:
        universe = ("ES_FRONT_CALENDAR_2D", "VX1", "VX2", "VX3")
    m = MagicMock()
    m.universe = universe
    return m


def _dummy_sleeve(signal_dict, name="dummy"):
    """Sleeve that returns fixed signals (no features)."""
    def signals(market, date):
        return pd.Series(signal_dict)
    sleeve = MagicMock()
    sleeve.signals = signals
    return sleeve


# ---------------------------------------------------------------------------
# Convergence: VX1 and VX2 with opposite signs
# ---------------------------------------------------------------------------

class TestConvergenceEmitsVX1AndVX2:
    """VRP Convergence must emit VX1 and VX2 only, with opposite signs."""

    def test_convergence_meta_emits_vx1_and_vx2_with_opposite_signs(self):
        """Convergence sleeve returns exactly two legs: VX1 and VX2, opposite sign."""
        meta = VRPConvergenceMeta()
        meta._signals_cache = pd.Series(
            [-0.3],
            index=[pd.Timestamp("2024-01-15")],
            name="signal",
        )
        market = _mock_market()
        out = meta.signals(market, "2024-01-15")
        assert isinstance(out, pd.Series)
        assert set(out.index) == {"VX1", "VX2"}, (
            f"Convergence must emit exactly VX1 and VX2, got {list(out.index)}"
        )
        assert out["VX1"] == pytest.approx(-0.3)
        assert out["VX2"] == pytest.approx(0.3)
        assert (out["VX1"] * out["VX2"]) < 0, "VX1 and VX2 must be opposite-signed"

    def test_convergence_spread_geometry_explicit(self):
        """Explicit spread: signal = -0.5 → VX1 = -0.5, VX2 = +0.5."""
        meta = VRPConvergenceMeta()
        meta._signals_cache = pd.Series(
            [-0.5],
            index=[pd.Timestamp("2024-06-01")],
            name="signal",
        )
        market = _mock_market()
        out = meta.signals(market, "2024-06-01")
        assert out["VX1"] == pytest.approx(-0.5)
        assert out["VX2"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Core and Alt: VX1 only
# ---------------------------------------------------------------------------

class TestCoreAndAltVX1Only:
    """VRP Core and VRP Alt must emit VX1 only."""

    def test_core_meta_emits_vx1_only(self):
        """VRP Core returns a single entry: VX1."""
        meta = VRPCoreMeta()
        meta._signals_cache = pd.Series(
            [0.4],
            index=[pd.Timestamp("2024-01-15")],
        )
        market = _mock_market()
        out = meta.signals(market, "2024-01-15")
        assert list(out.index) == ["VX1"], f"Core must emit only VX1, got {list(out.index)}"
        assert out["VX1"] == pytest.approx(0.4)

    def test_alt_meta_emits_vx1_only(self):
        """VRP Alt returns a single entry: VX1."""
        meta = VRPAltMeta()
        meta._signals_cache = pd.Series(
            [-0.2],
            index=[pd.Timestamp("2024-01-15")],
        )
        market = _mock_market()
        out = meta.signals(market, "2024-01-15")
        assert list(out.index) == ["VX1"], f"Alt must emit only VX1, got {list(out.index)}"
        assert out["VX1"] == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# Combine / reindex: VX2 and VX3 survive
# ---------------------------------------------------------------------------

class TestVX2VX3SurviveCombine:
    """VX2 and VX3 emitted by a VRP sleeve must survive combine and reindex."""

    def test_convergence_like_sleeve_vx2_survives_combine(self):
        """A Convergence-like sleeve (VX1 + VX2) survives CombinedStrategy combine and reindex."""
        market = _mock_market(universe=("ES_FRONT_CALENDAR_2D", "VX1", "VX2"))
        sleeve = _dummy_sleeve({"VX1": -0.2, "VX2": 0.2})
        strategy = CombinedStrategy(
            strategies={"vrp_convergence_meta": sleeve},
            weights={"vrp_convergence_meta": 1.0},
        )
        signals = strategy.signals(market, "2024-01-05")
        assert abs(signals["VX1"]) > 1e-9, "VX1 should survive combine"
        assert abs(signals["VX2"]) > 1e-9, "VX2 should survive combine"
        assert signals["VX1"] == pytest.approx(-0.2)
        assert signals["VX2"] == pytest.approx(0.2)

    def test_vx2_vx3_emitted_not_dropped_by_reindex(self):
        """When a VRP sleeve emits VX1, VX2, VX3, reindex to universe must not zero them out."""
        market = _mock_market(universe=("ES_FRONT_CALENDAR_2D", "VX1", "VX2", "VX3"))
        sleeve = _dummy_sleeve({"VX1": 0.1, "VX2": 0.2, "VX3": 0.15})
        strategy = CombinedStrategy(
            strategies={"vrp_core_meta": sleeve},
            weights={"vrp_core_meta": 1.0},
        )
        signals = strategy.signals(market, "2024-01-05")
        assert abs(signals["VX1"]) > 1e-9, "VX1 should not be dropped by reindex"
        assert abs(signals["VX2"]) > 1e-9, "VX2 should not be dropped by reindex"
        assert abs(signals["VX3"]) > 1e-9, "VX3 should not be dropped by reindex"
        assert signals["VX1"] == pytest.approx(0.1)
        assert signals["VX2"] == pytest.approx(0.2)
        assert signals["VX3"] == pytest.approx(0.15)
