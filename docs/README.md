# Documentation Index

All project documentation lives in this `docs/` directory so it stays easy to discover and maintain.

## Essential Reading

**Start here:**
- **`SOTs/STRATEGY.md`** ⭐ – **Complete step-by-step strategy execution flow and Core v1 baseline configuration**. This is the single source of truth for understanding exactly what happens at each stage of the backtest. Read this first to understand the full pipeline.
- **`SOTs/DIAGNOSTICS.md`** ⭐ – **Performance diagnostics framework**. How to analyze backtest runs, compute metrics, and perform ablation testing. Essential for evaluating strategy components.
- **`SOTs/PROCEDURES.md`** ⭐ – **Step-by-step procedures** for adding/changing sleeves, assets, and parameters (when and how to run Phase-0 → Phase-3).
- **`SOTs/ROADMAP.md`** ⭐ – **Strategic development roadmap** (2026–2028). Long-term sequencing, meta-sleeve expansion plans, production deployment planning, and Sharpe targets.
- **`SOTs/SYSTEM_CONSTRUCTION.md`** ⭐ – **System architecture and layer definitions**. Canonical execution stack, engine vs allocator separation, and production-ready components.
- **`architecture/DUAL_PRICE_ARCHITECTURE.md`** ⭐ – **Dual-price architecture**: Raw vs continuous prices, back-adjustment logic, and which modules use which price source. Essential for understanding how roll jumps are handled.

## Documentation Structure

### 📚 SOTs/ (Source of Truth)
**Authoritative documentation** - These are the single source of truth documents:
- `SYSTEM_CONSTRUCTION.md` - System architecture and layer definitions
- `DIAGNOSTICS.md` - Performance diagnostics framework
- `PROCEDURES.md` - Step-by-step procedures
- `ROADMAP.md` - Strategic development roadmap
- `STRATEGY.md` - Strategy execution flow

### 🎯 Milestones/
**Major project milestones and completion summaries:**
- `CANONICAL_FROZEN_STACK_BASELINE.md` - Canonical frozen stack baseline documentation
- `CANONICAL_FROZEN_STACK_EXECUTION_SUMMARY.md` - Execution summary for canonical baseline
- `STEPS_1_3_COMPLETE_SUMMARY.md` - Summary of Steps 1-3 completion
- `STEP_3_CLEANUP_COMPLETE.md` - Step 3 cleanup completion documentation

### 🧩 Components/
**Component-specific documentation:**

#### `components/allocator/`
Allocator v1 implementation and production documentation:
- `ALLOCATOR_V1_PRODUCTION_MODE.md` - Production mode specification
- `ALLOCATOR_V1_QUICK_START.md` - Quick start guide
- `ALLOCATOR_V1_STAGE_*.md` - Stage completion documentation
- `ALLOCATOR_STATE_V1_*.md` - State implementation documentation
- `STAGE_4_IMPLEMENTATION.md` - Stage 4 implementation details

#### `components/strategies/`
Strategy implementation documentation:
- `CROSS_SECTIONAL_MOMENTUM.md` - Cross-Sectional Momentum strategy
- `SR3_CARRY_CURVE.md` - SR3 carry and curve features
- `MACRO_REGIME_FILTER.md` - MacroRegimeFilter design and configuration
- `PARAM_SWEEP.md` - Parameter sweep runner
- `SLEEVE_ALLOCATOR.md` - Multi-sleeve portfolio allocation

### 📁 META_SLEEVES/
**Meta-sleeve specific documentation:**
- `TREND_IMPLEMENTATION.md` - Trend Meta-Sleeve implementation reference (current production)
- `TREND_RESEARCH.md` - Trend Meta-Sleeve research notebook (structured research document with Phase-0/1/2 results)
- `DEPLOYMENT_SUMMARY_20251119.md` - Deployment summary

### 📦 Archive/
**Historical and completed phase documentation:**
- `archive/phases/` - Completed phase documentation (Phase 1C, etc.)
- `archive/RT_BUG_FOUND.md` - Historical bug documentation

### 🗄️ Legacy/
**Legacy documentation:**
- `legacy/TSMOM_IMPLEMENTATION.md` - Legacy TSMOM class reference (not used in production)

## Additional Resources

- **`REPORTS_STRUCTURE.md`** – Reports directory structure and phase indexing system (canonical Phase-0/1/2 results)
- **`../README.md`** – High-level overview, quick start, and component inventory
- **`../examples/`** – Runnable demos (`demo_macro_regime.py`)
- **`../reports/`** – Generated backtest outputs when diagnostics are run
  - **`phase_index/`** – Canonical references to current Phase-0/1/2 runs
  - **`sanity_checks/`** – Phase-0 sanity check results (archive/ and latest/)
  - **`runs/`** – Phase-1/2 backtest run artifacts

## Quick Reference

**Current Strategy Flow:**
1. CombinedStrategy generates signals (TSMOM / TSMOMMultiHorizon v2 + SR3 carry/curve + Rates curve + FX/Commodity carry) → 2. Engine Policy v1 applies validity gates → 3. VolManagedOverlay targets volatility → 4. Risk Targeting Layer defines portfolio size → 5. Allocator optimizes weights → 6. ExecSim calculates returns

**Strategy Profiles:**
- **`core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro`**: Current production baseline (Core v9)

See `SOTs/STRATEGY.md` and `SOTs/SYSTEM_CONSTRUCTION.md` for complete details.
