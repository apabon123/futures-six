# Futures-Six: Institutional Systematic Macro Platform

**A production-ready, layered systematic trading framework for futures markets with comprehensive risk control.**

Futures-Six is an institutional-grade systematic macro platform implementing a **canonical 7-layer execution stack** with strict separation of concerns. The system separates economic return generation (Engines) from conditional risk control (Allocator), ensuring auditability, portability, and production safety.

---

## 🎯 What This Project Does

Futures-Six is a complete systematic trading framework that:

- 📊 **Reads market data** from DuckDB with strict read-only access
- 🎯 **Generates alpha signals** from multiple economic engines (Trend, CSMOM, VRP, Carry, Curve RV)
- ⚖️ **Constructs portfolios** with static, conviction-based weights
- 🎛️ **Targets volatility** to achieve consistent risk exposure (20% annual vol)
- 🛡️ **Controls risk** with Allocator v1 (state → regime → risk scalars)
- 💰 **Simulates execution** with realistic slippage and transaction costs
- 📉 **Produces comprehensive metrics**: CAGR, Sharpe, MaxDD, regime analysis, allocator diagnostics

---

## 🏗️ System Architecture

Futures-Six implements a **canonical 7-layer execution stack** (authoritative order):

1. **Engine Signals** (alpha generation)
2. **Engine Policy** (validity & selectivity) - *Phase 2 (Next)*
3. **Portfolio Construction** (static weights)
4. **Discretionary Overlay** (bounded tilts) - *Optional*
5. **Risk Targeting** (vol → leverage) - ✅ **Phase 1C Complete (Production-Ready)**
6. **Allocator** (risk brake) - ✅ **Phase 1C Complete (H/M/L Profiles Production-Ready)**
7. **Margin & Execution Constraints**

**Key Principle:** Each layer answers one question. No layer subsumes another's responsibilities.

**See:** `docs/SOTs/SYSTEM_CONSTRUCTION.md` for complete architectural specification.

---

## 📈 Current Production Baseline (Core v9)

**Strategy Profile:** `core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro`

**Performance (Canonical Window: 2020-01-06 to 2025-10-31):**
- **CAGR:** 7.20% (baseline), 4.96% (with Allocator v1)
- **Sharpe:** 0.65 (baseline), 0.55 (with Allocator v1)
- **MaxDD:** -15.32% (baseline), -14.33% (with Allocator v1) ✅
- **Volatility:** 11.75% (baseline), 9.64% (with Allocator v1) ✅

**Meta-Sleeve Composition:**
- **Trend:** 52.4% (5 atomic sleeves: long/medium/short momentum, residual trend, breakout)
- **CSMOM:** 21.85% (cross-sectional momentum)
- **VRP-Core:** 6.555% (volatility risk premium)
- **VRP-Convergence:** 2.185% (VRP convergence)
- **VRP-Alt:** 13.11% (alternative VRP expression)
- **VX Carry:** 4.6% (volatility carry)
- **Curve RV:** 8% (curve relative value: rank fly + pack slope)

**Risk Targeting Status:** ✅ **Phase 1C Complete (Production-Ready, January 2026)**
- Target volatility: 20% (configurable)
- Leverage cap: 7.0×, Leverage floor: 1.0×
- Vol estimation: Rolling 63-day covariance
- Update frequency: Weekly (on rebalances)
- Artifacts: Complete audit trail (leverage series, realized vol, weights pre/post)

**Allocator v1 Status:** ✅ **Phase 1C Complete (Production-Ready, January 2026)**
- State layer: 10 features (volatility, drawdown, correlation, engine health)
- Regime classification: 4 regimes (NORMAL, ELEVATED, STRESS, CRISIS)
- Risk scalars: Portfolio-level exposure scaling (0.25-1.0)
- Profiles: H (high risk tolerance), M (medium), L (low/institutional)
- Mode: `precomputed` (production-safe, deterministic, auditable)
- Golden proof: `rt_alloc_h_apply_precomputed_2024` (validated end-to-end)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Database Path

Edit `configs/data.yaml` to point to your database:

```yaml
db:
  path: "path/to/your/database"
  engine: "auto"  # auto-detect: duckdb or sqlite
```

### 3. Run the Strategy

```bash
# Run Core v9 baseline (allocator off)
python run_strategy.py \
  --strategy_profile core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro \
  --start 2020-01-06 \
  --end 2025-10-31

# Run with Allocator v1 (two-pass audit)
python scripts/diagnostics/run_allocator_two_pass.py \
  --strategy_profile core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro \
  --start 2020-01-06 \
  --end 2025-10-31
```

### 4. View Performance Diagnostics

**Command-line diagnostics:**
```bash
# Single run diagnostics
python scripts/run_perf_diagnostics.py --run_id <run_id>

# Compare baseline vs allocator
python scripts/run_perf_diagnostics.py \
  --run_id <allocator_run_id> \
  --baseline_id <baseline_run_id>
```

**Interactive dashboard:**
```bash
# Launch the Canonical Dashboard (interactive Streamlit app)
streamlit run src/dashboards/canonical_dashboard.py
```

The dashboard provides interactive visualization of backtest results with 9 views:
- Run Overview (artifact completeness, warnings, run notes)
- Equity + Drawdown
- Exposure Over Time (pre/post-allocator with policy markers)
- Position-Level View (holdings snapshot, PnL contribution, turnover)
- Allocator State Timeline (regime, scalars, drawdown overlay)
- Drag Waterfall (return decomposition)
- Correlation & Diversification Health
- Sleeve Concentration Timeline
- Baseline Comparison (if baseline selected)
- Diagnostics Summary

**See:** `docs/SOTs/DIAGNOSTICS.md` § "Canonical Dashboard" for complete documentation.

---

## 📚 Documentation

**Essential Reading (SOTs - Single Source of Truth):**

- **`docs/SOTs/SYSTEM_CONSTRUCTION.md`** ⭐ – Canonical 7-layer architecture, Engine/Allocator separation, Allocator v1 implementation
- **`docs/SOTs/STRATEGY.md`** ⭐ – Complete strategy execution flow, Meta-Sleeve architecture, Core v9 baseline
- **`docs/SOTs/DIAGNOSTICS.md`** ⭐ – Performance diagnostics framework, Canonical Dashboard (interactive tool), Allocator v1 diagnostics, Phase-0/1/2 validation
- **`docs/SOTs/PROCEDURES.md`** ⭐ – Step-by-step procedures for adding/changing sleeves, Allocator v1 production procedures
- **`docs/SOTs/ROADMAP.md`** ⭐ – Strategic development roadmap (2026-2028), sleeve status, production deployment planning

**Phase 1C Documentation (Risk Targeting + Allocator Integration):**

- **`docs/PHASE_1C_COMPLETE.md`** ⭐ – Comprehensive Phase 1C completion summary
- **`docs/PHASE_1C_SOT_REVIEW.md`** – SOT alignment verification
- **`docs/PHASE_1C_FINAL_ANALYSIS.md`** – Detailed A/B backtest analysis
- **`docs/PHASE_1C_BUG_FIXES_COMPLETE.md`** – Critical bug fixes summary
- **`docs/PHASE_1C_PROOF_RUN.md`** – Golden proof run documentation

**Allocator v1 Documentation:**

- **`docs/ALLOCATOR_V1_FREEZE.md`** – Production freeze specification (v1.0 locked)
- **`docs/ALLOCATOR_V1_PRODUCTION_MODE.md`** – Production mode specification (`precomputed` default)
- **`docs/ALLOCATOR_V1_STAGE_6_5_VALIDATION.md`** – Stability validation checklist
- **`docs/ALLOCATOR_V1_STAGE_5_COMPLETE.md`** – Stage 5 & 5.5 implementation (two-pass audit)
- **`docs/ALLOCATOR_V1_QUICK_START.md`** – Quick start guide

**Component Documentation:**

- `docs/META_SLEEVES/` – Meta-sleeve implementation details
- `docs/DUAL_PRICE_ARCHITECTURE.md` – Raw vs continuous prices
- `docs/PARAM_SWEEP.md` – Parameter optimization guide

---

## 🎛️ Strategy Components

### Engines (Meta-Sleeves)

**1. Trend Meta-Sleeve** ✅ **Production**
- 5 atomic sleeves: long/medium/short momentum, residual trend, breakout
- Multi-horizon feature combination
- Risk-normalized signals

**2. CSMOM Meta-Sleeve** ✅ **Production**
- Cross-sectional momentum (market-neutral)
- Ranks assets by returns, long top 33%, short bottom 33%

**3. VRP Meta-Sleeve** ✅ **Production**
- VRP-Core: VIX - RV21 spread
- VRP-Convergence: VIX - VX1 convergence
- VRP-Alt: VIX - RV5 spread (alternative expression)

**4. Carry Meta-Sleeve** ✅ **Production**
- VX Calendar Carry: VX2-VX1 spread
- SR3 Calendar Carry: SOFR calendar spreads

**5. Curve RV Meta-Sleeve** ✅ **Production**
- SR3 Rank Fly: 2-6-10 rank fly momentum
- SR3 Pack Slope: Front vs back pack momentum

### Risk Targeting (Layer 5)

**Status:** ✅ **Phase 1C Complete (Production-Ready, January 2026)**

**Purpose:** Define portfolio size by converting target volatility to leverage.

**Implementation:**
- Target volatility: 20% (configurable)
- Leverage cap: 7.0×, Leverage floor: 1.0×
- Vol estimation: Rolling 63-day covariance matrix
- Update frequency: Weekly (on rebalance dates)
- Weight scaling: Normalizes to unit gross, applies leverage, renormalizes if needed

**Artifacts:**
- `risk_targeting/leverage_series.csv` (time series)
- `risk_targeting/realized_vol.csv` (time series)
- `risk_targeting/weights_pre_risk_targeting.csv` (panel: date × instrument)
- `risk_targeting/weights_post_risk_targeting.csv` (panel: date × instrument)
- `risk_targeting/params.json` (config snapshot)

**See:** `docs/SOTs/SYSTEM_CONSTRUCTION.md` § "Phase 1C" for complete details.

### Allocator v1 (Risk Control, Layer 6)

**Status:** ✅ **Phase 1C Complete (Production-Ready, January 2026)**

**Architecture:**
- **State Layer:** 10 features (volatility, drawdown, correlation, engine health)
- **Regime Layer:** 4 regimes (NORMAL, ELEVATED, STRESS, CRISIS)
- **Risk Layer:** Risk scalars (0.25-1.0, EWMA smoothed)
- **Exposure Layer:** Portfolio-level weight scaling (1-rebalance lag)

**Profiles:**
- **Profile-H:** High risk tolerance (rare intervention, tail-only)
- **Profile-M:** Medium risk tolerance (balanced)
- **Profile-L:** Low risk tolerance (conservative, institutional-style)

**Production Mode:** `precomputed` (loads scalars from validated baseline run)

**Key Properties:**
- Deterministic (rule-based, no ML)
- Auditable (complete artifact trail)
- Sticky (hysteresis, anti-thrash)
- Portfolio-level only (no sleeve-specific scaling)

**Golden Proof:** `rt_alloc_h_apply_precomputed_2024` (validated end-to-end)

**See:** `docs/SOTs/SYSTEM_CONSTRUCTION.md` § "Phase 1C" and `docs/ALLOCATOR_V1_FREEZE.md` for complete specification.

---

## 📊 Universe

The strategy trades **13 continuous futures contracts** across equities, rates, commodities, and FX:

**Equities:** ES, NQ, RTY  
**Rates:** ZN, ZF, ZT, UB, SR3  
**Commodities:** CL, GC  
**FX:** 6E, 6B, 6J

All contracts use appropriate roll conventions (calendar or volume-weighted).

---

## 🧪 Testing & Validation

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific component tests
pytest tests/test_strat_momentum.py -v
pytest tests/test_marketdata.py -v
```

### Phase-0 Sanity Checks

Before any sleeve enters production, it must pass Phase-0 (sign-only, no overlays):

```bash
# Trend Meta-Sleeve
python scripts/run_tsmom_sanity.py --start 2020-01-06 --end 2025-10-31

# Cross-Sectional Momentum
python scripts/run_csmom_sanity.py --start 2020-01-06 --end 2025-10-31
```

**Phase-0 Pass Criteria:** Sharpe ≥ 0.10 over canonical window.

### Phase 1C Validation (Risk Targeting + Allocator)

```bash
# Validate Phase 1C completion (golden proof run)
python scripts/diagnostics/validate_phase1c_completion.py rt_alloc_h_apply_precomputed_2024

# Run Phase 1C A/B backtests (Baseline, RT only, RT + Alloc-H)
python scripts/diagnostics/run_phase1c_ab_backtests.py \
  --strategy_profile core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --allocator_mode precomputed \
  --precomputed_run_id <compute_run_id>

# Validate RT artifacts
python scripts/diagnostics/test_rt_artifact_fix.py <run_id>
```

### Allocator v1 Validation

```bash
# Two-pass audit (baseline vs allocator)
python scripts/diagnostics/run_allocator_two_pass.py \
  --strategy_profile core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro \
  --start 2020-01-06 \
  --end 2025-10-31

# Review comparison report
# reports/runs/<scaled_run_id>/two_pass_comparison.md
```

---

## 🏛️ Project Structure

```
futures-six/
├── configs/
│   ├── data.yaml              # Database connection and universe
│   ├── strategies.yaml        # Strategy parameters (engines, allocator, overlays)
│   └── fred_series.yaml       # FRED economic indicators
├── src/
│   ├── agents/
│   │   ├── data_broker.py     # MarketData: Read-only OHLCV access
│   │   ├── exec_sim.py        # ExecSim: Backtest execution engine
│   │   └── ...                 # Strategy agents (Trend, CSMOM, VRP, etc.)
│   ├── layers/
│   │   ├── risk_targeting.py  # RiskTargetingLayer: vol → leverage (Layer 5)
│   │   └── artifact_writer.py # ArtifactWriter: CSV/JSON artifact management
│   ├── allocator/
│   │   ├── state_v1.py        # AllocatorStateV1: 10 state features
│   │   ├── regime_v1.py       # RegimeClassifierV1: 4 regimes
│   │   ├── risk_v1.py         # RiskTransformerV1: Risk scalars
│   │   ├── profiles.py        # AllocatorProfile: H/M/L profiles
│   │   └── scalar_loader.py   # Precomputed scalar loader
│   └── diagnostics/
│       └── ...                 # Diagnostic utilities
├── scripts/
│   ├── diagnostics/
│   │   ├── run_allocator_two_pass.py         # Two-pass allocator audit
│   │   ├── run_phase1c_ab_backtests.py      # Phase 1C A/B backtests
│   │   ├── validate_phase1c_completion.py   # Phase 1C validation
│   │   ├── test_rt_artifact_fix.py           # RT artifact validation
│   │   ├── compare_two_runs.py              # Comparison report generator
│   │   └── run_perf_diagnostics.py          # Performance diagnostics
│   └── run_*.py               # Phase-0 sanity checks
├── docs/
│   ├── SOTs/                  # Single Source of Truth documents
│   ├── PHASE_1C_*.md          # Phase 1C completion documentation
│   ├── ALLOCATOR_V1_*.md      # Allocator v1 documentation
│   └── META_SLEEVES/          # Meta-sleeve implementation docs
├── tests/
│   ├── test_risk_targeting_contracts.py      # RT semantic correctness
│   ├── test_allocator_profile_activation.py  # Allocator profile tests
│   └── ...                     # Comprehensive test suite
├── run_strategy.py            # Main entry point
└── README.md                  # This file
```

---

## 🔒 Safety Guarantees

1. ✅ **Read-Only Database Access:** All connections use `read_only=True`
2. ✅ **No Mutations:** No CREATE, INSERT, UPDATE, DELETE, DROP, ALTER operations
3. ✅ **Dual-Price Architecture:** Raw prices unchanged; continuous prices built in-memory
4. ✅ **Deterministic:** All components are fully reproducible
5. ✅ **Auditable:** Complete artifact trail for all runs
6. ✅ **Frozen Production Logic:** Allocator v1 locked at v1.0 (changes require v2)

---

## 🎯 Key Features

- ✅ **Layered Architecture:** Canonical 7-layer execution stack with strict boundaries
- ✅ **Multiple Engines:** Trend, CSMOM, VRP, Carry, Curve RV meta-sleeves
- ✅ **Risk Targeting:** Production-ready volatility-to-leverage conversion (Layer 5)
- ✅ **Allocator v1:** Production-ready risk control with H/M/L profiles (Layer 6)
- ✅ **Two-Pass Audit:** Baseline vs allocator comparison framework
- ✅ **Comprehensive Diagnostics:** Performance metrics, regime analysis, allocator usage
- ✅ **Interactive Dashboard:** Streamlit-based visualization tool for run analysis (9 views, baseline comparison, warnings)
- ✅ **Phase-0 Validation:** Sign-only sanity checks before production
- ✅ **Contract Tests:** Prevent regressions in RT and Allocator semantics
- ✅ **Deterministic:** Fully reproducible backtests
- ✅ **Auditable:** Complete artifact trail (CSV, JSON metadata)

---

## 📝 License

[Specify license]

---

## 📧 Contact

[Specify contact information]

---

## 🙏 Acknowledgments

Futures-Six follows institutional best practices for systematic trading:
- Separation of engines and allocator
- Deterministic, auditable design
- Production-grade risk control
- Comprehensive validation framework

**See:** `docs/SOTs/SYSTEM_CONSTRUCTION.md` for complete architectural philosophy.

---

**Last Updated:** January 2026  
**Version:** Core v9 + Risk Targeting (Layer 5) + Allocator v1 H/M/L Profiles (Layer 6)  
**Status:** Phase 1C Complete ✅ | Next: Phase 2 (Engine Policy)
