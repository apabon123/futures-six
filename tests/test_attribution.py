"""
Tests for portfolio-consistent sleeve-level attribution.

Uses synthetic portfolios with analytically verifiable contributions.
Does NOT require DuckDB or any external data.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.attribution.core import (
    CONSISTENCY_TOL,
    compute_attribution,
    decompose_weights_by_sleeve,
)
from src.attribution.artifacts import generate_attribution_artifacts


# ---------------------------------------------------------------------------
# Helpers: build a small synthetic 2-sleeve, 2-instrument portfolio
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n_days: int = 100,
    n_rebalances: int = 5,
    seed: int = 42,
):
    """
    Build a synthetic portfolio with 2 sleeves and 2 instruments where
    contributions can be computed analytically.

    Sleeve A: trades only ASSET_X with signal +1.0, weight 0.6
    Sleeve B: trades only ASSET_Y with signal -0.5, weight 0.4

    Combined signal:
        ASSET_X = 0.6 * 1.0 = 0.6
        ASSET_Y = 0.4 * (-0.5) = -0.2

    Returns:
        weights_panel, asset_returns_simple, portfolio_returns_simple,
        sleeve_signals_history, universe
    """
    rng = np.random.RandomState(seed)
    universe = ["ASSET_X", "ASSET_Y"]

    # Generate trading dates (business days)
    dates = pd.bdate_range("2023-01-02", periods=n_days, freq="B")

    # Rebalance dates: evenly spaced
    rebal_indices = np.linspace(0, n_days - 1, n_rebalances + 1, dtype=int)[:-1]
    rebalance_dates = dates[rebal_indices]

    # Fixed signals per sleeve (constant throughout backtest)
    sleeve_a_signal = {"ASSET_X": 1.0, "ASSET_Y": 0.0}  # Only trades X
    sleeve_b_signal = {"ASSET_X": 0.0, "ASSET_Y": -0.5}  # Only trades Y

    sleeve_a_weight = 0.6
    sleeve_b_weight = 0.4

    # Combined signal
    combined_x = sleeve_a_weight * sleeve_a_signal["ASSET_X"] + sleeve_b_weight * sleeve_b_signal["ASSET_X"]
    combined_y = sleeve_a_weight * sleeve_a_signal["ASSET_Y"] + sleeve_b_weight * sleeve_b_signal["ASSET_Y"]

    # Final weights = combined signals (no overlay/allocator transforms for simplicity)
    weights_data = {
        "ASSET_X": [combined_x] * len(rebalance_dates),
        "ASSET_Y": [combined_y] * len(rebalance_dates),
    }
    weights_panel = pd.DataFrame(weights_data, index=rebalance_dates)

    # Asset returns: small random log returns
    log_returns = rng.randn(n_days, 2) * 0.01  # ~1% daily vol
    asset_returns_log = pd.DataFrame(log_returns, index=dates, columns=universe)
    asset_returns_simple = np.exp(asset_returns_log) - 1.0

    # Portfolio returns: computed same way as ExecSim
    weights_daily = weights_panel.reindex(dates).ffill().fillna(0.0)
    portfolio_log = (weights_daily * asset_returns_log).sum(axis=1)
    portfolio_simple = np.exp(portfolio_log) - 1.0

    # Sleeve signals history
    sleeve_signals_history = {
        "sleeve_a": [
            (d, pd.Series(sleeve_a_signal) * sleeve_a_weight)
            for d in rebalance_dates
        ],
        "sleeve_b": [
            (d, pd.Series(sleeve_b_signal) * sleeve_b_weight)
            for d in rebalance_dates
        ],
    }

    return (
        weights_panel,
        asset_returns_simple,
        portfolio_simple,
        sleeve_signals_history,
        universe,
        weights_daily,
        asset_returns_log,
    )


def _make_no_signal_data(n_days: int = 60, seed: int = 99):
    """
    Build a synthetic portfolio where Sleeve B has no signal for the
    second half of the backtest.
    """
    rng = np.random.RandomState(seed)
    universe = ["ASSET_X", "ASSET_Y"]
    dates = pd.bdate_range("2023-06-01", periods=n_days, freq="B")

    # Two rebalance dates: one in first half, one in second half
    mid = n_days // 2
    rebalance_dates = pd.DatetimeIndex([dates[0], dates[mid]])

    # Sleeve A: always active
    sleeve_a_weight = 0.7
    sleeve_a_signal = {"ASSET_X": 1.0, "ASSET_Y": 0.0}

    # Sleeve B: active first half, zero in second half
    sleeve_b_weight = 0.3
    sleeve_b_signal_1 = {"ASSET_X": 0.0, "ASSET_Y": 0.8}
    sleeve_b_signal_2 = {"ASSET_X": 0.0, "ASSET_Y": 0.0}  # No signal

    # Combined signals
    combined_1_x = sleeve_a_weight * 1.0
    combined_1_y = sleeve_b_weight * 0.8
    combined_2_x = sleeve_a_weight * 1.0
    combined_2_y = 0.0  # Sleeve B has no signal

    weights_panel = pd.DataFrame(
        {
            "ASSET_X": [combined_1_x, combined_2_x],
            "ASSET_Y": [combined_1_y, combined_2_y],
        },
        index=rebalance_dates,
    )

    log_returns = rng.randn(n_days, 2) * 0.01
    asset_returns_log = pd.DataFrame(log_returns, index=dates, columns=universe)
    asset_returns_simple = np.exp(asset_returns_log) - 1.0

    weights_daily = weights_panel.reindex(dates).ffill().fillna(0.0)
    portfolio_log = (weights_daily * asset_returns_log).sum(axis=1)
    portfolio_simple = np.exp(portfolio_log) - 1.0

    sleeve_signals_history = {
        "sleeve_a": [
            (rebalance_dates[0], pd.Series(sleeve_a_signal) * sleeve_a_weight),
            (rebalance_dates[1], pd.Series(sleeve_a_signal) * sleeve_a_weight),
        ],
        "sleeve_b": [
            (rebalance_dates[0], pd.Series(sleeve_b_signal_1) * sleeve_b_weight),
            (rebalance_dates[1], pd.Series(sleeve_b_signal_2) * sleeve_b_weight),
        ],
    }

    return (
        weights_panel,
        asset_returns_simple,
        portfolio_simple,
        sleeve_signals_history,
        universe,
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestAttributionSumsToPortfolio:
    """Attribution sleeve contributions must sum to portfolio return."""

    def test_attribution_sums_to_portfolio(self):
        """
        Core consistency: sum of sleeve daily contributions must equal
        portfolio daily simple return within tolerance.
        """
        (
            weights_panel,
            asset_returns_simple,
            portfolio_simple,
            sleeve_signals_history,
            universe,
            _,
            _,
        ) = _make_synthetic_data()

        result = compute_attribution(
            weights_panel=weights_panel,
            asset_returns_simple=asset_returns_simple,
            portfolio_returns_simple=portfolio_simple,
            sleeve_signals_history=sleeve_signals_history,
            universe=universe,
        )

        assert result["diagnostics"]["consistency_pass"], (
            f"Consistency check failed: "
            f"max residual = {result['diagnostics']['max_abs_daily_residual_active']:.2e}"
        )

        # Verify numerically
        contribs = result["atomic_contributions"]
        total_contrib = contribs.sum(axis=1)

        # Align dates
        common_dates = total_contrib.index.intersection(portfolio_simple.index)
        residual = total_contrib.loc[common_dates] - portfolio_simple.loc[common_dates]

        # Active dates only (after first rebalance)
        first_rebal = weights_panel.index[0]
        active = common_dates >= first_rebal
        active_residual = residual[active]

        assert active_residual.abs().max() < CONSISTENCY_TOL, (
            f"Max active residual = {active_residual.abs().max():.2e} "
            f"exceeds tolerance {CONSISTENCY_TOL:.0e}"
        )

    def test_cumulative_consistency(self):
        """Cumulative sum of contributions matches cumulative portfolio return."""
        (
            weights_panel,
            asset_returns_simple,
            portfolio_simple,
            sleeve_signals_history,
            universe,
            _,
            _,
        ) = _make_synthetic_data()

        result = compute_attribution(
            weights_panel=weights_panel,
            asset_returns_simple=asset_returns_simple,
            portfolio_returns_simple=portfolio_simple,
            sleeve_signals_history=sleeve_signals_history,
            universe=universe,
        )

        contribs = result["atomic_contributions"]
        first_rebal = weights_panel.index[0]
        active = contribs.index >= first_rebal

        cum_contrib = contribs[active].sum(axis=1).cumsum()
        cum_portfolio = portfolio_simple.reindex(contribs.index).fillna(0.0)[active].cumsum()

        final_diff = abs(cum_contrib.iloc[-1] - cum_portfolio.iloc[-1])
        assert final_diff < 1e-10, (
            f"Cumulative mismatch: contrib={cum_contrib.iloc[-1]:.10f}, "
            f"portfolio={cum_portfolio.iloc[-1]:.10f}, diff={final_diff:.2e}"
        )

    def test_analytical_sleeve_contribution(self):
        """
        With the synthetic data, Sleeve A trades only ASSET_X and Sleeve B
        trades only ASSET_Y, so contributions should match exactly.
        """
        (
            weights_panel,
            asset_returns_simple,
            portfolio_simple,
            sleeve_signals_history,
            universe,
            weights_daily,
            asset_returns_log,
        ) = _make_synthetic_data()

        result = compute_attribution(
            weights_panel=weights_panel,
            asset_returns_simple=asset_returns_simple,
            portfolio_returns_simple=portfolio_simple,
            sleeve_signals_history=sleeve_signals_history,
            universe=universe,
        )

        contribs = result["atomic_contributions"]
        first_rebal = weights_panel.index[0]
        active = contribs.index >= first_rebal

        # Sleeve A should have contribution from ASSET_X only
        # Sleeve B should have contribution from ASSET_Y only
        # Because signals are orthogonal: sleeve_a only has ASSET_X, sleeve_b only ASSET_Y

        # The actual weight for sleeve_a on ASSET_X = 0.6 (all of combined X signal)
        # The actual weight for sleeve_b on ASSET_Y = -0.2 (all of combined Y signal)

        # Verify that sleeve_a contribution ≈ 0.6 * r_X (in log-scaled terms)
        # and sleeve_b contribution ≈ -0.2 * r_Y
        # The approximation is very close since daily returns are small

        sleeve_a_expected_log = weights_daily["ASSET_X"] * asset_returns_log["ASSET_X"]
        sleeve_b_expected_log = weights_daily["ASSET_Y"] * asset_returns_log["ASSET_Y"]

        # Total expected log return
        total_log = sleeve_a_expected_log + sleeve_b_expected_log
        total_simple = np.exp(total_log) - 1.0

        # Normalized contributions
        # sleeve_a_contrib = sleeve_a_expected_log / total_log * total_simple
        # This should closely match contribs["sleeve_a"]
        # Due to exact proportional scaling, they should match within 1e-12
        nonzero = total_log.abs() > 1e-18
        expected_a = pd.Series(0.0, index=contribs.index)
        expected_a[nonzero] = sleeve_a_expected_log[nonzero] / total_log[nonzero] * total_simple[nonzero]

        diff = (contribs["sleeve_a"][active] - expected_a[active]).abs().max()
        assert diff < 1e-12, f"Sleeve A contribution mismatch: max diff = {diff:.2e}"


class TestNoSignalDaysHandled:
    """No-signal days should be handled gracefully."""

    def test_no_signal_days_handled(self):
        """Sleeve with zero signal should have zero contribution on those days."""
        (
            weights_panel,
            asset_returns_simple,
            portfolio_simple,
            sleeve_signals_history,
            universe,
        ) = _make_no_signal_data()

        result = compute_attribution(
            weights_panel=weights_panel,
            asset_returns_simple=asset_returns_simple,
            portfolio_returns_simple=portfolio_simple,
            sleeve_signals_history=sleeve_signals_history,
            universe=universe,
        )

        contribs = result["atomic_contributions"]
        diag = result["diagnostics"]

        # Consistency should still pass
        assert diag["consistency_pass"], (
            f"Consistency failed with no-signal sleeve: "
            f"max residual = {diag['max_abs_daily_residual_active']:.2e}"
        )

        # Sleeve B should have zero contribution on ASSET_Y in the second half
        mid_date = weights_panel.index[1]
        second_half = contribs.index >= mid_date
        sleeve_b_second_half = contribs.loc[second_half, "sleeve_b"]

        # Since sleeve B has no signal in second half, and ASSET_Y has zero
        # weight, sleeve B's contribution should be zero
        assert sleeve_b_second_half.abs().max() < 1e-15, (
            f"Sleeve B should have zero contribution in second half, "
            f"but max = {sleeve_b_second_half.abs().max():.2e}"
        )

        # No-signal info should be populated
        ns_info = diag["no_signal_info"]
        assert "sleeve_b" in ns_info
        assert ns_info["sleeve_b"]["no_signal_days"] > 0

    def test_no_signal_does_not_break_consistency(self):
        """Even with no-signal periods, sum of contributions equals portfolio."""
        (
            weights_panel,
            asset_returns_simple,
            portfolio_simple,
            sleeve_signals_history,
            universe,
        ) = _make_no_signal_data()

        result = compute_attribution(
            weights_panel=weights_panel,
            asset_returns_simple=asset_returns_simple,
            portfolio_returns_simple=portfolio_simple,
            sleeve_signals_history=sleeve_signals_history,
            universe=universe,
        )

        contribs = result["atomic_contributions"]
        first_rebal = weights_panel.index[0]
        active = contribs.index >= first_rebal

        total_contrib = contribs[active].sum(axis=1)
        port_aligned = portfolio_simple.reindex(contribs.index).fillna(0.0)[active]

        residual = (total_contrib - port_aligned).abs().max()
        assert residual < CONSISTENCY_TOL, (
            f"Residual with no-signal periods: {residual:.2e}"
        )


class TestUnitsAndHeadersPresent:
    """Output artifacts must have correct column names and units."""

    def test_units_and_headers_present(self):
        """Verify correct column names in attribution CSVs."""
        (
            weights_panel,
            asset_returns_simple,
            portfolio_simple,
            sleeve_signals_history,
            universe,
            _,
            _,
        ) = _make_synthetic_data()

        result = compute_attribution(
            weights_panel=weights_panel,
            asset_returns_simple=asset_returns_simple,
            portfolio_returns_simple=portfolio_simple,
            sleeve_signals_history=sleeve_signals_history,
            universe=universe,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            out_dir = generate_attribution_artifacts(
                attribution_result=result,
                portfolio_returns_simple=portfolio_simple,
                run_dir=run_dir,
                run_id="test_run_001",
            )

            # Check metasleeve CSV
            meta_csv = out_dir / "attribution_by_metasleeve.csv"
            assert meta_csv.exists(), "attribution_by_metasleeve.csv not created"
            meta_df = pd.read_csv(meta_csv)
            required_meta_cols = {"date", "metasleeve", "daily_contribution_return", "cumulative_contribution_return"}
            assert required_meta_cols.issubset(set(meta_df.columns)), (
                f"Missing columns in metasleeve CSV: {required_meta_cols - set(meta_df.columns)}"
            )

            # Check atomic sleeve CSV
            atomic_csv = out_dir / "attribution_by_atomic_sleeve.csv"
            assert atomic_csv.exists(), "attribution_by_atomic_sleeve.csv not created"
            atomic_df = pd.read_csv(atomic_csv)
            required_atomic_cols = {"date", "atomic_sleeve", "daily_contribution_return", "cumulative_contribution_return"}
            assert required_atomic_cols.issubset(set(atomic_df.columns)), (
                f"Missing columns in atomic sleeve CSV: {required_atomic_cols - set(atomic_df.columns)}"
            )

            # Verify no column named "PnL" (requirement: no ambiguity)
            all_cols = set(meta_df.columns) | set(atomic_df.columns)
            pnl_cols = [c for c in all_cols if "pnl" in c.lower()]
            assert len(pnl_cols) == 0, (
                f"Found PnL-named columns (ambiguous units): {pnl_cols}"
            )

            # Check summary JSON
            summary_json = out_dir / "attribution_summary.json"
            assert summary_json.exists(), "attribution_summary.json not created"
            with open(summary_json) as f:
                summary = json.load(f)
            required_keys = {
                "total_portfolio_cum_return",
                "sum_sleeve_cum_return",
                "residual_cum_return",
                "residual_pct_of_portfolio",
                "consistency_check",
                "per_sleeve",
            }
            assert required_keys.issubset(set(summary.keys())), (
                f"Missing keys in summary JSON: {required_keys - set(summary.keys())}"
            )

            # Check markdown
            md_path = out_dir / "ATTRIBUTION_SUMMARY.md"
            assert md_path.exists(), "ATTRIBUTION_SUMMARY.md not created"
            md_text = md_path.read_text()
            assert "What This Is Measuring" in md_text
            assert "Consistency Checks" in md_text
            assert "If Something Is Failing" in md_text

    def test_correlation_csv_generated(self):
        """Correlation matrix CSV should be generated when enough data."""
        (
            weights_panel,
            asset_returns_simple,
            portfolio_simple,
            sleeve_signals_history,
            universe,
            _,
            _,
        ) = _make_synthetic_data(n_days=200)

        result = compute_attribution(
            weights_panel=weights_panel,
            asset_returns_simple=asset_returns_simple,
            portfolio_returns_simple=portfolio_simple,
            sleeve_signals_history=sleeve_signals_history,
            universe=universe,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            out_dir = generate_attribution_artifacts(
                attribution_result=result,
                portfolio_returns_simple=portfolio_simple,
                run_dir=run_dir,
                run_id="test_run_corr",
            )
            corr_csv = out_dir / "sleeve_contribution_correlation.csv"
            assert corr_csv.exists(), "sleeve_contribution_correlation.csv not created"

    def test_summary_json_per_sleeve_metrics(self):
        """Each sleeve should have complete metrics in the summary."""
        (
            weights_panel,
            asset_returns_simple,
            portfolio_simple,
            sleeve_signals_history,
            universe,
            _,
            _,
        ) = _make_synthetic_data()

        result = compute_attribution(
            weights_panel=weights_panel,
            asset_returns_simple=asset_returns_simple,
            portfolio_returns_simple=portfolio_simple,
            sleeve_signals_history=sleeve_signals_history,
            universe=universe,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            out_dir = generate_attribution_artifacts(
                attribution_result=result,
                portfolio_returns_simple=portfolio_simple,
                run_dir=run_dir,
                run_id="test_metrics",
            )

            with open(out_dir / "attribution_summary.json") as f:
                summary = json.load(f)

            per_sleeve = summary["per_sleeve"]
            required_metrics = {
                "cum_return", "mean_daily_contrib", "mean_abs_daily_contrib",
                "vol_of_contrib", "max_dd_of_contrib", "hit_rate",
                "active_days", "no_signal_days",
            }
            for sleeve_name, metrics in per_sleeve.items():
                missing = required_metrics - set(metrics.keys())
                assert not missing, (
                    f"Sleeve {sleeve_name} missing metrics: {missing}"
                )


class TestWeightDecomposition:
    """Weight decomposition must preserve total weights."""

    def test_decomposition_sums_to_final_weights(self):
        """Sum of sleeve weight allocations must equal final weights."""
        (
            weights_panel,
            _,
            _,
            sleeve_signals_history,
            universe,
            _,
            _,
        ) = _make_synthetic_data()

        decomp = decompose_weights_by_sleeve(
            weights_panel, sleeve_signals_history, universe
        )

        for date in weights_panel.index:
            sleeve_sum = decomp.loc[date].sum(axis=0)
            final_w = weights_panel.loc[date].reindex(universe, fill_value=0.0)
            residual = (sleeve_sum - final_w).abs().max()
            assert residual < 1e-12, (
                f"Decomposition residual at {date}: {residual:.2e}"
            )

    def test_orthogonal_sleeves(self):
        """Sleeves with non-overlapping instruments get full weight allocation."""
        (
            weights_panel,
            _,
            _,
            sleeve_signals_history,
            universe,
            _,
            _,
        ) = _make_synthetic_data()

        decomp = decompose_weights_by_sleeve(
            weights_panel, sleeve_signals_history, universe
        )

        # Sleeve A only trades ASSET_X, so all of ASSET_X weight should go to sleeve_a
        for date in weights_panel.index:
            sleeve_a_x = decomp.loc[(date, "sleeve_a"), "ASSET_X"]
            final_x = weights_panel.loc[date, "ASSET_X"]
            assert abs(sleeve_a_x - final_x) < 1e-12, (
                f"Sleeve A should get all of ASSET_X weight: "
                f"got {sleeve_a_x}, expected {final_x}"
            )

            sleeve_b_y = decomp.loc[(date, "sleeve_b"), "ASSET_Y"]
            final_y = weights_panel.loc[date, "ASSET_Y"]
            assert abs(sleeve_b_y - final_y) < 1e-12, (
                f"Sleeve B should get all of ASSET_Y weight: "
                f"got {sleeve_b_y}, expected {final_y}"
            )


class TestEdgeCases:
    """Edge cases: empty data, single sleeve, zero returns."""

    def test_empty_inputs_handled(self):
        """Empty inputs should return empty results without error."""
        result = compute_attribution(
            weights_panel=pd.DataFrame(),
            asset_returns_simple=pd.DataFrame(),
            portfolio_returns_simple=pd.Series(dtype=float),
            sleeve_signals_history={},
            universe=[],
        )
        assert result["atomic_contributions"].empty
        assert "error" in result["diagnostics"]

    def test_single_sleeve(self):
        """Single-sleeve portfolio: contribution equals portfolio return."""
        rng = np.random.RandomState(123)
        universe = ["A"]
        dates = pd.bdate_range("2023-01-02", periods=50, freq="B")
        rebal = pd.DatetimeIndex([dates[0], dates[25]])

        weights_panel = pd.DataFrame({"A": [1.0, 1.0]}, index=rebal)

        log_r = pd.DataFrame(rng.randn(50, 1) * 0.005, index=dates, columns=universe)
        simple_r = np.exp(log_r) - 1.0

        w_daily = weights_panel.reindex(dates).ffill().fillna(0.0)
        port_log = (w_daily * log_r).sum(axis=1)
        port_simple = np.exp(port_log) - 1.0

        signals = {
            "only_sleeve": [
                (rebal[0], pd.Series({"A": 1.0})),
                (rebal[1], pd.Series({"A": 1.0})),
            ]
        }

        result = compute_attribution(
            weights_panel=weights_panel,
            asset_returns_simple=simple_r,
            portfolio_returns_simple=port_simple,
            sleeve_signals_history=signals,
            universe=universe,
        )

        assert result["diagnostics"]["consistency_pass"]

        contribs = result["atomic_contributions"]["only_sleeve"]
        active = contribs.index >= rebal[0]
        diff = (contribs[active] - port_simple.reindex(contribs.index).fillna(0.0)[active]).abs().max()
        assert diff < 1e-14, f"Single sleeve should equal portfolio: diff={diff:.2e}"
