"""
Tests for VRP tradability: master universe includes VX, sleeve scope, and guardrails.

These tests ensure VRP is a real tradable sleeve by verifying:
1. VX1/VX2/VX3 are in the master universe (data.yaml config)
2. VRP sleeve signals stay within scope (VX1/VX2/VX3 only)
3. Universe mismatch guardrails work (strict mode raises, non-strict warns)
"""

import pytest
import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.strat_combined import (
    CombinedStrategy,
    VRP_ALLOWED_SYMBOLS,
    VRP_SLEEVE_NAMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data_config() -> dict:
    """Load configs/data.yaml and return parsed dict."""
    config_path = Path(__file__).parent.parent / "configs" / "data.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_master_universe(config: dict) -> set:
    """Build the full master universe set from data.yaml (standard + VX)."""
    universe_cfg = config.get("universe", {})
    vx_cfg = config.get("vx_universe", {})

    # Standard futures: map short names to DB symbols
    fx_symbols = {"6E", "6B", "6J"}
    db_symbols = set()
    for short_name, roll_cfg in universe_cfg.items():
        if isinstance(roll_cfg, dict):
            roll_type = roll_cfg.get("roll", "calendar")
            if roll_type == "calendar":
                if short_name in fx_symbols or short_name == "SR3":
                    db_symbols.add(f"{short_name}_FRONT_CALENDAR")
                else:
                    db_symbols.add(f"{short_name}_FRONT_CALENDAR_2D")
            elif roll_type == "volume":
                db_symbols.add(f"{short_name}_FRONT_VOLUME")

    # VX symbols use portfolio-level short names (VX1, VX2, VX3)
    vx_symbols = set(vx_cfg.keys()) if vx_cfg else set()
    return db_symbols | vx_symbols


def _make_mock_market(universe=None):
    """Create a minimal mock MarketData for CombinedStrategy tests."""
    market = MagicMock()
    if universe is None:
        # Default: 2 standard + VX1
        universe = (
            "ES_FRONT_CALENDAR_2D",
            "NQ_FRONT_CALENDAR_2D",
            "VX1",
        )
    market.universe = universe
    return market


def _make_dummy_sleeve(signal_dict: dict, name="dummy"):
    """Create a dummy strategy sleeve that returns fixed signals."""
    sleeve = MagicMock()
    sleeve.signals = MagicMock(return_value=pd.Series(signal_dict))

    # Make inspect.signature work (no 'features' parameter)
    import types
    def signals(market, date):
        return pd.Series(signal_dict)
    sleeve.signals = signals
    return sleeve


# ===========================================================================
# 1) test_master_universe_includes_vx
# ===========================================================================

class TestMasterUniverseIncludesVX:
    """Assert VX1/VX2/VX3 are in the master universe from configs/data.yaml."""

    def test_vx_universe_section_exists(self):
        """data.yaml must have a vx_universe section."""
        config = _load_data_config()
        assert "vx_universe" in config, (
            "configs/data.yaml is missing 'vx_universe' section. "
            "VX1/VX2/VX3 must be defined for VRP tradability."
        )

    def test_vx1_vx2_vx3_present(self):
        """VX1, VX2, VX3 must each be present in vx_universe."""
        config = _load_data_config()
        vx_cfg = config.get("vx_universe", {})
        for sym in ("VX1", "VX2", "VX3"):
            assert sym in vx_cfg, f"{sym} missing from vx_universe in data.yaml"

    def test_vx_db_symbols_non_empty(self):
        """Each VX entry must map to a non-empty DB symbol."""
        config = _load_data_config()
        vx_cfg = config.get("vx_universe", {})
        for sym, db_sym in vx_cfg.items():
            assert db_sym and isinstance(db_sym, str) and len(db_sym) > 0, (
                f"VX entry '{sym}' has invalid db_symbol: {db_sym}"
            )

    def test_full_master_universe_contains_vx(self):
        """The union universe (standard + VX) must include VX1, VX2, VX3."""
        config = _load_data_config()
        master = _build_master_universe(config)
        for sym in ("VX1", "VX2", "VX3"):
            assert sym in master, (
                f"{sym} not in master universe: {sorted(master)}"
            )

    def test_standard_universe_unchanged(self):
        """Standard 13 futures must still be present (no regression)."""
        config = _load_data_config()
        master = _build_master_universe(config)
        # Spot-check a few critical standard symbols
        assert "ES_FRONT_CALENDAR_2D" in master
        assert "NQ_FRONT_CALENDAR_2D" in master
        assert "ZN_FRONT_VOLUME" in master
        assert "SR3_FRONT_CALENDAR" in master
        assert "CL_FRONT_VOLUME" in master


# ===========================================================================
# 2) test_vrp_emits_in_universe
# ===========================================================================

class TestVRPEmitsInUniverse:
    """VRP sleeve signals must survive the combine step when VX is in universe."""

    def test_vx1_signal_retained(self):
        """A VRP sleeve emitting VX1 should survive CombinedStrategy.signals()."""
        market = _make_mock_market(universe=(
            "ES_FRONT_CALENDAR_2D", "VX1",
        ))

        vrp_sleeve = _make_dummy_sleeve({"VX1": 0.75})

        strategy = CombinedStrategy(
            strategies={"vrp_core_meta": vrp_sleeve},
            weights={"vrp_core_meta": 1.0},
        )

        signals = strategy.signals(market, "2024-01-05")

        assert isinstance(signals, pd.Series)
        assert "VX1" in signals.index, "VX1 was dropped from combined signals"
        assert signals["VX1"] == pytest.approx(0.75), (
            f"VX1 signal value wrong: expected 0.75, got {signals['VX1']}"
        )

    def test_vx1_not_dropped_by_reindex(self):
        """
        If VX1 is in market.universe, the reindex step must NOT zero it out.
        """
        market = _make_mock_market(universe=(
            "ES_FRONT_CALENDAR_2D", "NQ_FRONT_CALENDAR_2D", "VX1",
        ))

        vrp_sleeve = _make_dummy_sleeve({"VX1": -0.5})
        tsmom_sleeve = _make_dummy_sleeve({
            "ES_FRONT_CALENDAR_2D": 0.3,
            "NQ_FRONT_CALENDAR_2D": -0.2,
        })

        strategy = CombinedStrategy(
            strategies={
                "tsmom": tsmom_sleeve,
                "vrp_core_meta": vrp_sleeve,
            },
            weights={"tsmom": 0.5, "vrp_core_meta": 0.5},
        )

        signals = strategy.signals(market, "2024-01-05")

        # VX1 should have non-zero signal
        assert abs(signals["VX1"]) > 1e-6, (
            f"VX1 signal is zero after combine: {signals.to_dict()}"
        )
        # Standard symbols should also work
        assert "ES_FRONT_CALENDAR_2D" in signals.index
        assert "NQ_FRONT_CALENDAR_2D" in signals.index

    def test_vx2_vx3_survive_combine_when_emitted(self):
        """
        VRP sleeves can emit VX2/VX3; they must survive combine/reindex.
        Infrastructure supports all three; current sleeves only emit VX1.
        """
        market = _make_mock_market(universe=(
            "ES_FRONT_CALENDAR_2D", "VX1", "VX2", "VX3",
        ))

        # Simulate a future VRP sleeve that emits VX1/VX2/VX3 (e.g. term-structure)
        vrp_multi = _make_dummy_sleeve({
            "VX1": -0.3,
            "VX2": 0.2,
            "VX3": 0.1,
        })

        strategy = CombinedStrategy(
            strategies={"vrp_core_meta": vrp_multi},
            weights={"vrp_core_meta": 1.0},
        )

        signals = strategy.signals(market, "2024-01-05")

        assert abs(signals["VX1"]) > 1e-6, "VX1 should survive"
        assert abs(signals["VX2"]) > 1e-6, "VX2 should survive when emitted"
        assert abs(signals["VX3"]) > 1e-6, "VX3 should survive when emitted"


# ===========================================================================
# 3) test_universe_mismatch_raises_in_strict_mode
# ===========================================================================

class TestUniverseMismatchGuardrails:
    """Universe mismatch detection (strict mode = error, non-strict = warning)."""

    def test_strict_mode_raises_on_unknown_symbol(self):
        """
        If a sleeve emits a symbol NOT in universe and strict_universe=True,
        CombinedStrategy.signals() must raise RuntimeError.
        """
        market = _make_mock_market(universe=(
            "ES_FRONT_CALENDAR_2D",
        ))

        rogue_sleeve = _make_dummy_sleeve({"NOT_IN_UNIVERSE": 1.0})

        strategy = CombinedStrategy(
            strategies={"rogue": rogue_sleeve},
            weights={"rogue": 1.0},
            strict_universe=True,
        )

        with pytest.raises(RuntimeError, match="UNIVERSE MISMATCH"):
            strategy.signals(market, "2024-01-05")

    def test_non_strict_mode_warns(self, caplog):
        """
        In non-strict mode, a mismatch should log a WARNING (not raise).
        """
        market = _make_mock_market(universe=(
            "ES_FRONT_CALENDAR_2D",
        ))

        rogue_sleeve = _make_dummy_sleeve({"NOT_IN_UNIVERSE": 1.0})

        strategy = CombinedStrategy(
            strategies={"rogue": rogue_sleeve},
            weights={"rogue": 1.0},
            strict_universe=False,
        )

        with caplog.at_level(logging.WARNING):
            signals = strategy.signals(market, "2024-01-05")

        # Should have logged a warning about universe mismatch
        assert any("UNIVERSE MISMATCH" in rec.message for rec in caplog.records), (
            "Expected a UNIVERSE MISMATCH warning in logs"
        )
        # Signal should still return (but the rogue symbol is dropped)
        assert isinstance(signals, pd.Series)

    def test_vrp_scope_enforcement(self):
        """VRP sleeves emitting symbols outside VRP_ALLOWED_SYMBOLS get filtered."""
        market = _make_mock_market(universe=(
            "ES_FRONT_CALENDAR_2D", "VX1",
        ))

        # Create a VRP sleeve that (incorrectly) emits both VX1 and ES
        bad_vrp = _make_dummy_sleeve({"VX1": 0.5, "ES_FRONT_CALENDAR_2D": 0.3})

        strategy = CombinedStrategy(
            strategies={"vrp_core_meta": bad_vrp},
            weights={"vrp_core_meta": 1.0},
        )

        signals = strategy.signals(market, "2024-01-05")

        # ES signal from VRP should have been filtered (VRP scope = VX only)
        # The ES signal should be 0.0 (not contributed by VRP)
        assert signals["ES_FRONT_CALENDAR_2D"] == pytest.approx(0.0), (
            f"VRP sleeve should not contribute to ES: {signals.to_dict()}"
        )
        # VX1 should still have the signal
        assert signals["VX1"] == pytest.approx(0.5)

    def test_non_vrp_sleeves_cannot_trade_vx(self):
        """
        Non-VRP sleeves (e.g., TSMOM) must NOT contribute VX signals,
        even if market.universe includes VX symbols.
        """
        market = _make_mock_market(universe=(
            "ES_FRONT_CALENDAR_2D", "VX1",
        ))

        # TSMOM generates signals for all universe symbols (including VX1)
        tsmom_sleeve = _make_dummy_sleeve({
            "ES_FRONT_CALENDAR_2D": 0.5,
            "VX1": 0.3,  # TSMOM should NOT be able to trade VX
        })

        strategy = CombinedStrategy(
            strategies={"tsmom": tsmom_sleeve},
            weights={"tsmom": 1.0},
        )

        signals = strategy.signals(market, "2024-01-05")

        # TSMOM should contribute ES but NOT VX1
        assert signals["ES_FRONT_CALENDAR_2D"] == pytest.approx(0.5)
        assert signals["VX1"] == pytest.approx(0.0), (
            f"TSMOM should not trade VX: VX1={signals['VX1']}"
        )


# ===========================================================================
# 4) VRP_ALLOWED_SYMBOLS constant
# ===========================================================================

class TestVRPConstants:
    """Verify VRP constants are correctly defined."""

    def test_vrp_allowed_symbols_contains_vx(self):
        assert "VX1" in VRP_ALLOWED_SYMBOLS
        assert "VX2" in VRP_ALLOWED_SYMBOLS
        assert "VX3" in VRP_ALLOWED_SYMBOLS

    def test_vrp_sleeve_names(self):
        assert "vrp_core_meta" in VRP_SLEEVE_NAMES
        assert "vrp_convergence_meta" in VRP_SLEEVE_NAMES
        assert "vrp_alt_meta" in VRP_SLEEVE_NAMES
        # TSMOM should NOT be in VRP sleeve names
        assert "tsmom" not in VRP_SLEEVE_NAMES
