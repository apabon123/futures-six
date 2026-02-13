"""
Core attribution logic for Futures-Six.

Computes portfolio-consistent sleeve-level return contributions.

DEFINITIONS
-----------
- **daily_contribution_return**: The portion of the portfolio's daily simple
  return attributable to a given sleeve. Formally:

      contrib_s(t) = [sleeve_log_contrib_s(t) / port_log_return(t)]
                     * port_simple_return(t)

  where sleeve_log_contrib_s(t) = sum_i( alpha_{s,i}(t) * w_i(t) * log_r_i(t) )
  and alpha_{s,i} is the fraction of final weight w_i attributable to sleeve s.

  These contributions are EXACTLY additive:
      sum_s( contrib_s(t) ) == port_simple_return(t)

- **cumulative_contribution_return**: Cumulative sum of daily_contribution_return
  over time. NOT the same as compound return of the sleeve, but represents
  the sleeve's total additive contribution to portfolio cumulative performance.

CONSISTENCY GUARANTEE
---------------------
The sum of all sleeve daily contributions equals the portfolio daily simple
return within floating-point tolerance (~1e-14). This is enforced by
construction: we decompose portfolio log returns proportionally by sleeve,
then scale to simple returns.

UNITS
-----
All return quantities are in DECIMAL form (e.g., 0.01 = 1%).
No currency PnL is computed (we do not have notional information).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tolerance for portfolio consistency checks
CONSISTENCY_TOL = 1e-8


def decompose_weights_by_sleeve(
    weights_panel: pd.DataFrame,
    sleeve_signals_history: Dict[str, List[Tuple]],
    universe: List[str],
) -> pd.DataFrame:
    """
    Decompose final portfolio weights into per-sleeve weight allocations.

    For each rebalance date and asset, computes the fraction of the final
    weight attributable to each sleeve based on signal contribution ratios.

    Args:
        weights_panel: [rebalance_date x symbol] final portfolio weights.
        sleeve_signals_history: {sleeve_name: [(date, weighted_signals_series), ...]}.
            Each entry is the sleeve's weighted signal (sleeve_weight * raw_signal).
        universe: List of all symbols in the universe.

    Returns:
        DataFrame with MultiIndex (date, sleeve) and columns = symbols.
        Values are the portion of the final weight attributable to each sleeve.
        Guaranteed: sum over sleeves for each (date, symbol) == final_weight.
    """
    if weights_panel.empty or not sleeve_signals_history:
        logger.warning("[Attribution] Empty weights or sleeve signals; cannot decompose.")
        return pd.DataFrame()

    sleeve_names = sorted(sleeve_signals_history.keys())
    rebalance_dates = weights_panel.index

    # Build per-sleeve signal DataFrames at rebalance frequency
    sleeve_signal_dfs = {}
    for sleeve_name in sleeve_names:
        entries = sleeve_signals_history[sleeve_name]
        if not entries:
            continue
        dates = [d for d, _ in entries]
        signals = [s for _, s in entries]
        df = pd.DataFrame(signals, index=dates)
        # Reindex to universe, fill missing with 0
        df = df.reindex(columns=universe, fill_value=0.0).fillna(0.0)
        # Align to rebalance dates (use nearest earlier date)
        df = df.reindex(rebalance_dates, method="ffill").fillna(0.0)
        sleeve_signal_dfs[sleeve_name] = df

    if not sleeve_signal_dfs:
        logger.warning("[Attribution] No sleeve signal data available.")
        return pd.DataFrame()

    # Compute combined signal: sum of all sleeve weighted signals
    combined_signal = pd.DataFrame(0.0, index=rebalance_dates, columns=universe)
    for sleeve_name, df in sleeve_signal_dfs.items():
        combined_signal = combined_signal.add(df, fill_value=0.0)

    # Compute fractional allocation per sleeve
    # Where combined_signal == 0, distribute equally among sleeves that have
    # non-zero signals, or if all zero, attribute to no sleeve.
    rows = []
    for date in rebalance_dates:
        final_w = weights_panel.loc[date].reindex(universe, fill_value=0.0)
        combined = combined_signal.loc[date]

        for sleeve_name in sleeve_names:
            sleeve_sig = sleeve_signal_dfs[sleeve_name].loc[date]
            # Compute fraction: sleeve_signal / combined_signal
            fraction = pd.Series(0.0, index=universe)
            nonzero_mask = combined.abs() > 1e-15
            fraction[nonzero_mask] = sleeve_sig[nonzero_mask] / combined[nonzero_mask]

            # For assets where combined signal is zero but weight is nonzero,
            # attribute weight equally across all sleeves
            zero_combined_nonzero_weight = (~nonzero_mask) & (final_w.abs() > 1e-15)
            if zero_combined_nonzero_weight.any():
                n_sleeves = len(sleeve_names)
                fraction[zero_combined_nonzero_weight] = 1.0 / n_sleeves

            sleeve_weight_alloc = fraction * final_w
            rows.append((date, sleeve_name, sleeve_weight_alloc))

    # Build MultiIndex DataFrame
    index_tuples = [(d, s) for d, s, _ in rows]
    data = [w.values for _, _, w in rows]
    mi = pd.MultiIndex.from_tuples(index_tuples, names=["date", "sleeve"])
    result = pd.DataFrame(data, index=mi, columns=universe)

    # Validation: sum of sleeve weights should equal final weights
    for date in rebalance_dates:
        sleeve_sum = result.loc[date].sum(axis=0)
        final_w = weights_panel.loc[date].reindex(universe, fill_value=0.0)
        residual = (sleeve_sum - final_w).abs().max()
        if residual > 1e-10:
            logger.warning(
                f"[Attribution] Weight decomposition residual at {date}: {residual:.2e}"
            )

    return result


def compute_attribution(
    weights_panel: pd.DataFrame,
    asset_returns_simple: pd.DataFrame,
    portfolio_returns_simple: pd.Series,
    sleeve_signals_history: Dict[str, List[Tuple]],
    universe: List[str],
    metasleeve_mapping: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Compute portfolio-consistent sleeve-level return attribution.

    Args:
        weights_panel: [rebalance_date x symbol] final portfolio weights.
        asset_returns_simple: [date x symbol] daily simple returns.
        portfolio_returns_simple: [date] daily portfolio simple returns.
        sleeve_signals_history: {sleeve_name: [(date, weighted_signals_series), ...]}.
        universe: List of all symbols.
        metasleeve_mapping: Optional {sleeve_name: metasleeve_name} for grouping.
            If None, each sleeve is its own metasleeve.

    Returns:
        Dict with keys:
            "atomic_contributions": DataFrame [date x sleeve] daily contributions.
            "metasleeve_contributions": DataFrame [date x metasleeve] daily contributions.
            "sleeve_weight_decomposition": MultiIndex DataFrame of weight allocations.
            "diagnostics": Dict with consistency check results.
    """
    if weights_panel.empty or asset_returns_simple.empty:
        logger.error("[Attribution] Empty inputs; cannot compute attribution.")
        return {
            "atomic_contributions": pd.DataFrame(),
            "metasleeve_contributions": pd.DataFrame(),
            "sleeve_weight_decomposition": pd.DataFrame(),
            "diagnostics": {"error": "Empty inputs"},
        }

    # Step 1: Decompose weights by sleeve
    decomp = decompose_weights_by_sleeve(
        weights_panel, sleeve_signals_history, universe
    )
    if decomp.empty:
        logger.error("[Attribution] Weight decomposition failed.")
        return {
            "atomic_contributions": pd.DataFrame(),
            "metasleeve_contributions": pd.DataFrame(),
            "sleeve_weight_decomposition": decomp,
            "diagnostics": {"error": "Weight decomposition failed"},
        }

    sleeve_names = sorted(sleeve_signals_history.keys())
    rebalance_dates = weights_panel.index
    all_dates = asset_returns_simple.index

    # Step 2: Convert simple returns to log returns for consistent computation
    # Clip to avoid log(0): if simple_return <= -1, clip to a very small value
    asset_returns_clipped = asset_returns_simple.clip(lower=-0.9999)
    asset_log_returns = np.log1p(asset_returns_clipped)
    asset_log_returns = asset_log_returns.reindex(columns=universe, fill_value=0.0).fillna(0.0)

    # Step 3: Forward-fill sleeve weight decomposition to daily frequency
    # For each sleeve, build a daily weight DataFrame
    sleeve_daily_weights = {}
    for sleeve_name in sleeve_names:
        # Extract this sleeve's weight allocation at rebalance dates
        try:
            sleeve_rebal = decomp.loc[(slice(None), sleeve_name), :]
            sleeve_rebal = sleeve_rebal.droplevel("sleeve")
        except KeyError:
            sleeve_rebal = pd.DataFrame(0.0, index=rebalance_dates, columns=universe)

        # Forward-fill to daily
        daily = sleeve_rebal.reindex(all_dates).ffill().fillna(0.0)
        sleeve_daily_weights[sleeve_name] = daily

    # Step 4: Compute daily sleeve contributions in log-return space
    sleeve_log_contribs = {}
    for sleeve_name in sleeve_names:
        w = sleeve_daily_weights[sleeve_name]
        common_cols = w.columns.intersection(asset_log_returns.columns)
        if len(common_cols) == 0:
            sleeve_log_contribs[sleeve_name] = pd.Series(0.0, index=all_dates)
            continue
        contrib = (w[common_cols] * asset_log_returns[common_cols]).sum(axis=1)
        sleeve_log_contribs[sleeve_name] = contrib

    sleeve_log_df = pd.DataFrame(sleeve_log_contribs)

    # Step 5: Compute total portfolio log return from decomposition (for verification)
    portfolio_log_from_decomp = sleeve_log_df.sum(axis=1)

    # Step 6: Compute portfolio log return from original data (for consistency check)
    weights_daily = weights_panel.reindex(all_dates).ffill().fillna(0.0)
    common_syms = weights_daily.columns.intersection(asset_log_returns.columns)
    portfolio_log_direct = (weights_daily[common_syms] * asset_log_returns[common_syms]).sum(axis=1)

    # Step 7: Normalize sleeve contributions to simple returns
    # contrib_simple_s = (sleeve_log_s / port_log) * port_simple
    # This ensures exact additivity in simple return space
    portfolio_simple = portfolio_returns_simple.reindex(all_dates).fillna(0.0)

    # Use portfolio log return from direct computation for the normalization
    # to ensure exact match with published portfolio returns
    sleeve_simple_contribs = {}
    for sleeve_name in sleeve_names:
        log_c = sleeve_log_contribs[sleeve_name]
        simple_c = pd.Series(0.0, index=all_dates)

        # Where portfolio log return is nonzero, scale proportionally
        nonzero = portfolio_log_direct.abs() > 1e-18
        simple_c[nonzero] = (
            log_c[nonzero] / portfolio_log_direct[nonzero] * portfolio_simple[nonzero]
        )

        # Where portfolio log return ≈ 0, sleeve contributions should also be ≈ 0
        # (they're proportional fractions of a near-zero quantity)
        sleeve_simple_contribs[sleeve_name] = simple_c

    atomic_contributions = pd.DataFrame(sleeve_simple_contribs)

    # Step 8: Build metasleeve contributions
    if metasleeve_mapping is None:
        metasleeve_mapping = {s: s for s in sleeve_names}

    metasleeve_names = sorted(set(metasleeve_mapping.values()))
    meta_contribs = {}
    for meta_name in metasleeve_names:
        member_sleeves = [s for s, m in metasleeve_mapping.items() if m == meta_name]
        meta_contribs[meta_name] = atomic_contributions[
            [s for s in member_sleeves if s in atomic_contributions.columns]
        ].sum(axis=1)
    metasleeve_contributions = pd.DataFrame(meta_contribs)

    # Step 9: Diagnostics
    total_contrib = atomic_contributions.sum(axis=1)
    daily_residual = total_contrib - portfolio_simple
    max_abs_daily_residual = daily_residual.abs().max()
    cum_residual = daily_residual.sum()

    # Also check log-space consistency
    log_residual = portfolio_log_from_decomp - portfolio_log_direct
    max_log_residual = log_residual.abs().max()

    # Filter to active dates (after first rebalance)
    first_rebal = rebalance_dates[0] if len(rebalance_dates) > 0 else all_dates[0]
    active_mask = all_dates >= first_rebal
    n_active_days = active_mask.sum()
    max_abs_active_residual = daily_residual[active_mask].abs().max()

    consistency_pass = max_abs_active_residual < CONSISTENCY_TOL

    # No-signal detection per sleeve
    no_signal_info = {}
    for sleeve_name in sleeve_names:
        sleeve_w = sleeve_daily_weights[sleeve_name]
        active_w = sleeve_w[active_mask]
        total_active_days = len(active_w)
        zero_days = (active_w.abs().sum(axis=1) < 1e-15).sum()
        no_signal_info[sleeve_name] = {
            "active_days": int(total_active_days - zero_days),
            "no_signal_days": int(zero_days),
            "total_days": int(total_active_days),
        }

    # Partial attribution warning
    captured_sleeves = set(sleeve_names)
    partial = False
    partial_note = ""
    # Check if sum of sleeve signals at rebalance covers the combined weight
    # by comparing sum-of-sleeve-weights to final weights
    for date in rebalance_dates[:3]:  # Spot-check first few dates
        if date in decomp.index.get_level_values("date"):
            sleeve_sum = decomp.loc[date].sum(axis=0)
            final_w = weights_panel.loc[date].reindex(universe, fill_value=0.0)
            residual = (sleeve_sum - final_w).abs().max()
            if residual > 1e-6:
                partial = True
                partial_note = (
                    f"Weight decomposition residual {residual:.2e} > 1e-6 "
                    f"at {date}. Attribution may be partial."
                )
                break

    diagnostics = {
        "consistency_pass": bool(consistency_pass),
        "tolerance": CONSISTENCY_TOL,
        "max_abs_daily_residual": float(max_abs_daily_residual),
        "max_abs_daily_residual_active": float(max_abs_active_residual),
        "cum_residual": float(cum_residual),
        "max_log_space_residual": float(max_log_residual),
        "n_active_days": int(n_active_days),
        "n_sleeves_captured": len(captured_sleeves),
        "sleeves_captured": sorted(captured_sleeves),
        "no_signal_info": no_signal_info,
        "partial_attribution": partial,
        "partial_note": partial_note,
        "first_rebalance_date": str(first_rebal),
    }

    if consistency_pass:
        logger.info(
            f"[Attribution] Consistency check PASSED: "
            f"max residual = {max_abs_active_residual:.2e} "
            f"(tolerance = {CONSISTENCY_TOL:.0e})"
        )
    else:
        logger.warning(
            f"[Attribution] Consistency check FAILED: "
            f"max residual = {max_abs_active_residual:.2e} "
            f"(tolerance = {CONSISTENCY_TOL:.0e})"
        )

    return {
        "atomic_contributions": atomic_contributions,
        "metasleeve_contributions": metasleeve_contributions,
        "sleeve_weight_decomposition": decomp,
        "diagnostics": diagnostics,
    }
