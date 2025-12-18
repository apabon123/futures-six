# Documentation Index

All project documentation lives in this `docs/` directory so it stays easy to discover and maintain.

## Essential Reading

**Start here:**
- **`SOTs/STRATEGY.md`** ⭐ – **Complete step-by-step strategy execution flow and Core v1 baseline configuration**. This is the single source of truth for understanding exactly what happens at each stage of the backtest. Read this first to understand the full pipeline.
- **`SOTs/DIAGNOSTICS.md`** ⭐ – **Performance diagnostics framework**. How to analyze backtest runs, compute metrics, and perform ablation testing. Essential for evaluating strategy components.
- **`SOTs/PROCEDURES.md`** ⭐ – **Step-by-step procedures** for adding/changing sleeves, assets, and parameters (when and how to run Phase-0 → Phase-3).
- **`SOTs/ROADMAP.md`** ⭐ – **Strategic development roadmap** (2026–2028). Long-term sequencing, meta-sleeve expansion plans, production deployment planning, and Sharpe targets.
- **`DUAL_PRICE_ARCHITECTURE.md`** ⭐ – **Dual-price architecture**: Raw vs continuous prices, back-adjustment logic, and which modules use which price source. Essential for understanding how roll jumps are handled.

## Component Documentation

- **`META_SLEEVES/TREND_IMPLEMENTATION.md`** – Trend Meta-Sleeve implementation reference (current production)
- **`META_SLEEVES/TREND_RESEARCH.md`** – Trend Meta-Sleeve research notebook (structured research document with Phase-0/1/2 results for all tested sleeves, including Breakout 50-100d)
- **`REPORTS_STRUCTURE.md`** – Reports directory structure and phase indexing system (canonical Phase-0/1/2 results)
- **`legacy/TSMOM_IMPLEMENTATION.md`** – Legacy TSMOM class reference (not used in production)
- **`SR3_CARRY_CURVE.md`** – SR3 carry and curve features (Phase 2): Feature definitions, implementation, configuration, and usage guide
- **Rates Curve** (Phase 2): See `src/agents/feature_rates_curve.py` and `src/agents/strat_rates_curve.py` – Treasury futures curve trading using FRED-anchored yields (2s10s, 5s30s)
- **`MACRO_REGIME_FILTER.md`** – MacroRegimeFilter design, configuration, and FRED indicators integration
- **`CROSS_SECTIONAL_MOMENTUM.md`** – Cross-Sectional Momentum strategy (alternative to TSMOM)
- **`PARAM_SWEEP.md`** – Parameter sweep runner for grid search and configuration optimization
- **`SLEEVE_ALLOCATOR.md`** – Multi-sleeve portfolio allocation (for combining multiple strategies)

## Additional Resources

- **`../README.md`** – High-level overview, quick start, and component inventory
- **`../examples/`** – Runnable demos (`demo_macro_regime.py`)
- **`REPORTS_STRUCTURE.md`** – Reports directory structure and phase indexing system (canonical Phase-0/1/2 results)
- **`../reports/`** – Generated backtest outputs when diagnostics are run
  - **`phase_index/`** – Canonical references to current Phase-0/1/2 runs
  - **`sanity_checks/`** – Phase-0 sanity check results (archive/ and latest/)
  - **`runs/`** – Phase-1/2 backtest run artifacts

## Quick Reference

**Current Strategy Flow:**
1. CombinedStrategy generates signals (TSMOM / TSMOMMultiHorizon v2 + SR3 carry/curve + Rates curve + FX/Commodity carry) → 2. MacroRegimeFilter scales by regime → 3. VolManagedOverlay targets volatility → 4. Allocator optimizes weights → 5. ExecSim calculates returns

**Strategy Profiles:**
- **`core_v1_no_macro`**: Baseline (Long-term TSMOM + FX/Commodity Carry)
- **`core_v2_no_macro`**: Multi-Horizon TSMOM (v2) with EWMA normalization

See `SOTs/STRATEGY.md` for complete details.
