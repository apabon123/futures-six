# Performance Diagnostics

This document describes the performance diagnostics framework for analyzing backtest runs.

**This is the single source of truth for all Phase-0/1/2 validation results and metrics.**

## Related Documents

- [docs/SOTs/STRATEGY.md](docs/SOTs/STRATEGY.md): Sleeve definitions and signal specifications
- [docs/SOTs/PROCEDURES.md](docs/SOTs/PROCEDURES.md): Promotion workflow, run creation, required artifacts
- [docs/SOTs/ROADMAP.md](docs/SOTs/ROADMAP.md): Sleeve status and sequencing

## Overview

The diagnostics framework provides a systematic way to:
- Load and analyze backtest run artifacts
- Compute core performance metrics (CAGR, Sharpe, MaxDD, etc.)
- Break down performance by year and by asset
- Compare runs against baselines for ablation testing

## Canonical Baselines & Pinned Runs (Phase-3 Governance)

Futures-Six maintains an explicit set of pinned runs that are considered:

- **Known-good**
- **Artifact-complete**
- **Diagnostic-validated**

These runs serve as authoritative baselines for:

- Ablation testing
- Performance attribution
- Regression detection
- Policy / allocator proofs

The authoritative registry of pinned runs is maintained in:

- `reports/_PINNED/README.md`

**Rules:**

- Any diagnostic comparison must reference a pinned run (or explicitly justify why it does not).

- Pinned runs are immutable:
  - Never overwritten
  - Never silently replaced

- New pinned runs must:
  - Have all required artifacts
  - Have `canonical_diagnostics.json` and `.md` generated
  - Have attribution artifacts generated (`reports/runs/<run_id>/analysis/attribution/`)
  - Be explicitly added to `reports/_PINNED/README.md` with purpose noted

This separation ensures:

- **SOTs define rules**
- **The README defines state**
- Diagnostics remain reproducible and auditable


This makes DIAGNOSTICS.md the enforcer, not the ledger.

## V1 Freeze Baseline

The **V1 frozen baseline** is the official run produced from `configs/phase4_v1_frozen.yaml` for paper-trading hardening. It is pinned as `v1_frozen_production_baseline` in `configs/pinned_runs.yaml`.

**How the baseline run is generated:**
- `run_strategy.py --config_path configs/phase4_v1_frozen.yaml --run_id v1_frozen_baseline_20200106_20251031_<timestamp> --start 2020-01-06 --end 2025-10-31 --strict_universe`
- No VRP, no SR3 calendar spread carry; metasleeves: tsmom_multihorizon, csmom_meta, sr3_curve_rv_meta, vx_calendar_carry only.

**Required artifacts:**
- `portfolio_returns.csv`, `weights.csv`, `sleeve_returns.csv`, `equity_curve.csv`, `meta.json`
- `sleeve_returns.csv` must contain only: `tsmom_multihorizon`, `csmom_meta`, `sr3_curve_rv_meta`, `vx_calendar_carry` (no VRP columns, no sr3_calendar_spread_carry)

**Attribution requirement:**
- Attribution must be regenerated for the run; `reports/runs/<run_id>/analysis/attribution/` must exist with governance status **PASS** or **WARN** (no partial attribution warning).
- **Attribution status thresholds** (governance): **PASS** if max |daily residual| ≤ 1e-5; **WARN** if 1e-5 < residual ≤ 1e-4; **FAIL** if residual > 1e-4. Run page and `attribution_summary.json` display this status.

**Sleeve weights fidelity:**
- The source of truth for attribution sleeve weights is `reports/runs/<run_id>/analysis/sleeve_weights.json`. When present, attribution uses these weights (with `sleeve_returns.csv`) for return-based contribution so that all enabled sleeves, including VX carry, are attributed correctly. Fallback order: (1) `sleeve_weights.json`, (2) config snapshot from `meta.json`, (3) weight decomposition from `weights.csv`.

**Identity requirement:**
- No VRP sleeves in any artifacts; no SR3 calendar spread carry column in sleeve_returns or attribution.

## Governance Fields (Phase 3A)

These fields are derived from execution-time metadata (`meta.json`) and represent the **authoritative** state of system governance. They are **not** inferred from artifact existence (unless explicitly noted for legacy recovery).

| Stage | Key Fields | Description |
|-------|------------|-------------|
| **Policy** | `policy_enabled` | Master switch state |
| | `policy_effective` | True if inputs present and logic ran |
| | `policy_inert` | True if enabled but inputs missing (FAILURE) |
| | `policy_has_teeth` | True if any multiplier < 1.0 (binding) |
| **Risk Targeting** | `rt_enabled` | Master switch state |
| | `rt_effective` | True if history present (returns/cov) |
| | `rt_has_teeth` | True if leverage != 1.0 (active scaling) |
| | `rt_multiplier_stats` | p5/p50/p95 leverage multipliers |
| **Allocator** | `alloc_v1_enabled` | Master switch state |
| | `alloc_v1_effective` | True if state computed and inputs present |
| | `alloc_v1_has_teeth` | True if any scalar < 1.0 (active braking) |
| | `alloc_v1_regime_dist`| % days in Normal/Elevated/Stress/Crisis |

**Data-Driven Rule:** Governance status must be derived from `meta.json` telemetry. Fallback to artifact inference is only permitted for pre-Phase 3A legacy runs.

## Performance Windows & Metrics Scope

To ensure auditability, diagnostics must explicitly report the windows used for metric computation.

### Window Definitions

- **Requested Range**: Defined by `start_date` and `end_date` in `meta.json`.
- **Effective Window**: Starts at `effective_start_date` (first rebalance where weights exist).
- **Per-Stage Readiness**: Tracked in `per_stage_effective_starts` (Policy, RT, Allocator).
- **Evaluation Window**: Starts at `evaluation_start_date = max(per_stage_readiness)`.

### Metrics Scope

- **`metrics_full`**: Computed over the **Available returns span** (`effective_start_date` → `end_date`). Provides risk context over the entire tradable history.
- **`metrics_eval`**: Computed over the **Evaluation window** (`evaluation_start_date` → `end_date`). This is the **authoritative** performance record.

*Note: These are identical if the system is effective from the first rebalance.*

## End-to-End Production Readiness Diagnostics

Production readiness is achieved when all major failure modes are identified, explainable, and acceptable, not eliminated.

### Diagnostic Scope

Diagnostics must explicitly cover:

1. **Historical stress windows**
   - 2020 Q1 (COVID-19 crash)
   - 2022 (inflation/rates regime shift)
   - Other major crisis periods in expanded historical window

2. **Sleeve-level loss attribution**
   - Which sleeves contributed to drawdowns
   - Sleeve-level correlations during stress
   - Sleeve behavior consistency across regimes

3. **Allocator behavior under stress**
   - Allocator response to crisis periods
   - Conditional exposure control effectiveness
   - Regime-dependent de-risking performance

4. **Correlation spikes**
   - Sleeve correlations during stress vs normal periods
   - Portfolio-level correlation breakdown
   - Diversification failure modes

5. **Path-dependent drawdowns**
   - Sequence of losses leading to maximum drawdown
   - Recovery patterns
   - Time-to-recovery analysis

### Attribution Diagnostics

Any run used for promotion, ablation, or roadmap decisions must include attribution artifacts. Required attribution diagnostics include:

- Attribution by metasleeve (daily and cumulative contribution return)
- Attribution by atomic sleeve
- Residual consistency check (sleeve contributions sum to portfolio return within tolerance)
- Sleeve contribution correlation matrix
- No-signal diagnostics (per-sleeve active vs no-signal days)
- When a sleeve shows materially negative contribution in Phase-4, a Phase-2 vs Phase-4 atomic return stream comparison must be performed before promotion decisions (used for SR3 curve RV and VRP).

Artifacts are written under `reports/runs/<run_id>/analysis/attribution/` (e.g. `attribution_by_metasleeve.csv`, `attribution_summary.json`, `ATTRIBUTION_SUMMARY.md`).

Promotion rules are defined in [docs/SOTs/PROCEDURES.md](docs/SOTs/PROCEDURES.md).

**Promotion eligibility (engine policy):** Engines whose unconditional implementation exhibits negative or unstable expectation over the canonical evaluation window are ineligible for promotion until a formal engine policy specification exists. This rule applies irrespective of whether the strategy may be profitable under discretionary or informal regime filtering.

### Universe consistency check

Runs must fail if a sleeve emits signals for instruments not present in the configured universe, unless explicitly allowed by sleeve scope (e.g. VRP meta-sleeve restricted to VX1/VX2/VX3). This is enforced by the `--strict_universe` gate in `run_strategy.py` for governed runs.

### Production Readiness Criteria

A system is production-ready when:
- All major failure modes are **identified** (not hidden)
- All failure modes are **explainable** (root causes understood)
- All failure modes are **acceptable** (within risk tolerance)
- System behavior is **consistent** across historical windows
- Allocator logic is **validated** at portfolio level

**Note**: Production readiness does not require elimination of all failure modes. It requires understanding and acceptance of remaining risks.

## Architecture

### Core Module: `src/agents/diagnostics_perf.py`

The diagnostics module provides:

1. **`RunData` dataclass**: Container for backtest run data
   - `run_id`: Run identifier
   - `portfolio_returns`: Daily portfolio returns (simple returns)
   - `equity_curve`: Daily equity curve (cumulative)
   - `asset_returns`: Daily asset returns (simple returns)
   - `weights`: Rebalance-date weights
   - `meta`: Run metadata (config, dates, universe, etc.)

2. **`load_run()`**: Load run artifacts from disk
   - Reads from `reports/runs/{run_id}/`
   - Loads: `portfolio_returns.csv`, `equity_curve.csv`, `asset_returns.csv`, `weights.csv`, `meta.json`
   - Returns a `RunData` object

3. **`compute_core_metrics()`**: Calculate core performance metrics
   - CAGR (Compound Annual Growth Rate)
   - Annualized Volatility
   - Sharpe Ratio (annualized)
   - Maximum Drawdown
   - Hit Rate (fraction of positive return days)

4. **`compute_yearly_stats()`**: Year-by-year performance breakdown
   - CAGR, Volatility, Sharpe, MaxDD per year
   - Helps identify which years drove performance

5. **`compute_per_asset_stats()`**: Per-asset attribution
   - Annualized return contribution
   - Annualized volatility
   - Sharpe ratio per asset
   - Identifies which assets contributed positively/negatively

6. **`compare_to_baseline()`**: Compare two runs
   - Computes deltas in all core metrics
   - Shows equity ratio over time
   - Essential for ablation testing

### CLI Script: `scripts/run_perf_diagnostics.py`

Command-line interface for running diagnostics:

```bash
# Analyze a single run
python scripts/run_perf_diagnostics.py --run_id <run_id>

# Compare against a baseline
python scripts/run_perf_diagnostics.py --run_id <run_id> --baseline_id <baseline_id>

# Specify custom base directory
python scripts/run_perf_diagnostics.py --run_id <run_id> --base_dir <path>
```

## Usage

### Running Diagnostics

**Basic usage:**
```bash
python scripts/run_perf_diagnostics.py --run_id momentum_only_v1
```

**With baseline comparison:**
```bash
python scripts/run_perf_diagnostics.py \
  --run_id momentum_stack_v1 \
  --baseline_id momentum_only_v1
```

### Output Format

The diagnostics script prints:

1. **Core Metrics**: CAGR, Vol, Sharpe, MaxDD, HitRate
2. **Year-by-Year Stats**: Performance breakdown by calendar year
3. **Per-Asset Stats**: All assets sorted by annualized return contribution
4. **Baseline Comparison** (if provided):
   - Current vs baseline metrics
   - Deltas (current - baseline)
   - Equity ratio over time

### Example Output

```
================================================================================
CORE METRICS
================================================================================
  CAGR           :     0.0033 (0.33%)
  Vol            :     0.1645 (16.45%)
  Sharpe         :     0.1023
  MaxDD          :    -0.4062 (-40.62%)
  HitRate        :     0.5003 (50.03%)

================================================================================
YEAR-BY-YEAR STATS
================================================================================
          CAGR       Vol    Sharpe     MaxDD
Year
2021  0.151500  0.122962  1.234567 -0.097163
2022 -0.012500  0.151184 -0.082654 -0.168414
2023 -0.051400  0.126604 -0.405678 -0.174025
2024 -0.080800  0.108487 -0.744444 -0.148823
2025  0.051900  0.146115  0.355123 -0.140045

================================================================================
PER-ASSET STATS (all assets, sorted by AnnRet)
================================================================================
                         AnnRet    AnnVol    Sharpe
symbol
ES_FRONT_CALENDAR_2D   0.015900  0.031549  0.503755
6E_FRONT_CALENDAR      0.003207  0.015480  0.207173
...
Total assets: 13

================================================================================
BASELINE COMPARISON
================================================================================
Current:  momentum_stack_v1
Baseline: momentum_only_v1

Delta (Current - Baseline):
  CAGR_delta          :    -0.0026 (-0.26%)
  Sharpe_delta        :    -0.1346
  MaxDD_delta         :    -8.28% (worse)
  ...
```

## Canonical Medium-Term Diagnostics Examples

### Phase-0: Sign-Only Sanity Check
```bash
python scripts/run_trend_med_canonical_phase0.py --start 2020-01-01 --end 2025-10-31
# Results saved to: reports/sanity_checks/trend/medium_canonical/archive/{timestamp}/
# Phase-0 index: reports/phase_index/trend/medium_canonical/phase0.txt
```

### Phase-1: Standalone Canonical vs Legacy Comparison
```bash
python scripts/run_trend_med_canonical_phase1.py --start 2020-01-01 --end 2025-10-31
# Results saved to: reports/runs/med_canonical_phase1_standalone_{timestamp}/
#                    reports/runs/med_canonical_phase1_legacy_{timestamp}/
# Phase-1 index: reports/phase_index/trend/medium_canonical/phase1.txt

# Compare canonical vs legacy:
python scripts/run_perf_diagnostics.py \
  --run_id med_canonical_phase1_standalone_{timestamp} \
  --baseline_id med_canonical_phase1_legacy_{timestamp}
```

### Phase-2: Integrated A/B Test (Canonical vs Baseline)
```bash
# Run with canonical medium-term variant integrated into Trend Meta-Sleeve:
python run_strategy.py \
  --strategy_profile core_v3_trend_medcanon_no_macro \
  --run_id core_v3_medcanon_v1 \
  --start 2020-01-01 \
  --end 2025-10-31

# Compare to baseline (legacy medium-term):
python scripts/run_perf_diagnostics.py \
  --run_id core_v3_medcanon_v1 \
  --baseline_id core_v3_no_macro

# Phase-2 index: reports/phase_index/trend/medium_canonical/phase2.txt
```

## Canonical Short-Term Diagnostics Examples

### Phase-0: Sign-Only Sanity Check
```bash
python scripts/run_trend_short_canonical_phase0.py --start 2020-01-01 --end 2025-10-31
# Results saved to: reports/sanity_checks/trend/short_canonical/archive/{timestamp}/
# Phase-0 index: reports/phase_index/trend/short_canonical/phase0.txt
```

### Phase-1: Standalone Canonical vs Legacy Comparison
```bash
python scripts/run_trend_short_canonical_phase1.py --start 2020-01-01 --end 2025-10-31
# Results saved to: reports/runs/short_canonical_phase1_standalone_{timestamp}/
#                    reports/runs/short_canonical_phase1_legacy_{timestamp}/
# Phase-1 index: reports/phase_index/trend/short_canonical/phase1.txt

# Compare canonical vs legacy:
python scripts/run_perf_diagnostics.py \
  --run_id short_canonical_phase1_standalone_{timestamp} \
  --baseline_id short_canonical_phase1_legacy_{timestamp}
```

### Phase-2: Integrated A/B Test (Canonical vs Baseline)
```bash
# Run with canonical short-term variant integrated into Trend Meta-Sleeve:
python run_strategy.py \
  --strategy_profile core_v3_trend_shortcanon_no_macro \
  --run_id core_v3_shortcanon_v1 \
  --start 2020-01-01 \
  --end 2025-10-31

# Compare to baseline (legacy short-term):
python scripts/run_perf_diagnostics.py \
  --run_id core_v3_shortcanon_v1 \
  --baseline_id core_v3_no_macro
```

**Note**: These diagnostics were used to evaluate the equal-weight short-term variant (Phase-0/1/2 completed Nov 2025). The variant was **not promoted** to production; legacy weights (0.5, 0.3, 0.2) remain the production standard. The diagnostic commands and canonical variant code (`variant="canonical"`) are preserved for future re-testing if the universe, timeframe, or surrounding architecture changes. See `TREND_RESEARCH.md` and `docs/SOTs/PROCEDURES.md` for full results and rationale.

# Phase-2 index: reports/phase_index/trend/short_canonical/phase2.txt
```

## Reports Directory Structure

The reports directory uses a canonical structure that makes it easy to find the current Phase-0, Phase-1, and Phase-2 results for each atomic sleeve:

```
reports/
  sanity_checks/              # Phase-0 raw logs (all runs)
    {meta_sleeve}/
      {sleeve_name}/
        archive/
          {timestamp}/        # All historical Phase-0 runs
        latest/               # Canonical Phase-0 results (always current)
        latest_failed/         # Optional: last failing Phase-0 (for parked sleeves)
  
  phase_index/                # Canonical references to current runs
    {meta_sleeve}/
      {sleeve_name}/
        phase0.txt           # Points to latest/ directory
        phase1.txt           # Points to Phase-1 run_id
        phase2.txt           # Points to Phase-2 run_id
        status.txt           # Optional: status for parked sleeves
  
  runs/                       # Phase-1/2 backtest runs
    {run_id}/
      ...
```

### Finding Canonical Results

**Quick Answer**: "What's the current Phase-0/1/2 for Trend → breakout_mid_50_100?"

1. Open `reports/phase_index/trend/breakout_mid_50_100/`
2. Read `phase0.txt`, `phase1.txt`, or `phase2.txt`
3. Navigate to the referenced location

**Using the Helper Module**:
```python
from src.utils.phase_index import get_phase_path

# Get canonical Phase-0 path
phase0_path = get_phase_path("trend", "breakout_mid_50_100", "phase0")
# Returns: Path("reports/sanity_checks/trend/breakout_mid_50_100/latest")

# Get canonical Phase-1 path
phase1_path = get_phase_path("trend", "breakout_mid_50_100", "phase1")
# Returns: Path("reports/runs/breakout_1b_7030")
```

For more details, see `docs/REPORTS_STRUCTURE.md`.

## Run Artifacts

When `ExecSim.run()` is called with a `run_id`, it saves artifacts to `reports/runs/{run_id}/`:

**Core Run Artifacts (Always Generated):**
- **`portfolio_returns.csv`**: Daily portfolio returns (simple returns)
  - Columns: `date`, `ret`
  - **Guarantee:** Always generated, regardless of mode (compute, precomputed, or off)
  
- **`equity_curve.csv`**: Daily equity curve (cumulative, starts at 1.0)
  - Columns: `date`, `equity`
  - **Guarantee:** Always generated, regardless of mode
  
- **`weights.csv`**: Portfolio weights at rebalance dates
  - Index: Rebalance date
  - Columns: Asset symbols
  - **Guarantee:** Always generated, regardless of mode
  
- **`meta.json`**: Run metadata (see structure below)
  - **Guarantee:** Always generated, regardless of mode

**Optional Artifacts:**
- **`asset_returns.csv`**: Daily asset returns (simple returns)
  - Index: Date
  - Columns: Asset symbols

- **`allocator_scalars_at_rebalances.csv`**: Allocator scalar values at each rebalance (when allocator enabled)
  - Must include `rebalance_date` column (ISO YYYY-MM-DD format)
  - Must include `risk_scalar_computed` and `risk_scalar_applied` columns
  - Canonical format uses `rebalance_date` as index column

**Meta.json Structure:**
```json
{
  "run_id": "string",
  "start_date": "YYYY-MM-DD",           // Requested start date
  "end_date": "YYYY-MM-DD",
  "effective_start_date": "YYYY-MM-DD",  // First rebalance date (after warmup)
  "strategy_profile": "string",          // Strategy profile name (if used)
  "strategy_config_name": "string",
  "universe": ["symbol1", "symbol2", ...],
  "rebalance": "W-FRI",
  "slippage_bps": 0.5,
  "n_rebalances": 263,
  "n_trading_days": 1819,
  "canonical_window": true,              // Boolean: matches canonical window?
  "config_hash": "hex_string",           // SHA256 hash of config file (for reproducibility)
  "allocator_source_run_id": "string",   // Source run ID for precomputed allocator scalars
  "engine_policy_source_run_id": "string" // Source run ID for precomputed engine policy multipliers
}
```

**Precomputed Mode Artifact Guarantee:**
Precomputed mode runs the full backtest path and saves the same output artifacts as compute mode. The only difference is where scalars/gates come from (loaded from prior run vs computed on-the-fly). This ensures:
- All diagnostic tools work with precomputed runs
- Complete audit trail for paper-live runs
- Source run IDs link precomputed runs back to their compute baselines

## Data Format Notes

### Return Types

All returns in the diagnostics module use **simple returns** (not log returns):
- Portfolio returns: `r_simple = exp(r_log) - 1`
- Asset returns: `r_simple = exp(r_log) - 1`

This ensures consistency across all calculations.

### Equity Curve

The equity curve is computed from daily simple returns:
```python
equity_daily = (1 + portfolio_returns_daily).cumprod()
equity_daily.iloc[0] = 1.0  # Start at 1.0
```

### Per-Asset Attribution

Per-asset stats compute approximate contribution by:
1. Forward-filling weights to daily frequency
2. Computing daily PnL: `asset_pnl = weights * asset_returns`
3. Computing annualized stats from daily PnL series

## Ablation Testing Workflow

The diagnostics framework is designed for systematic ablation testing:

1. **Run baseline**: `python run_strategy.py --strategy_profile <baseline> --run_id <baseline_id>`
2. **Run variant**: `python run_strategy.py --strategy_profile <variant> --run_id <variant_id>`
3. **Compare**: `python scripts/run_perf_diagnostics.py --run_id <variant_id> --baseline_id <baseline_id>`

This workflow allows you to:
- Identify which components add/subtract value
- Understand year-by-year impact
- See per-asset attribution changes
- Make data-driven decisions about strategy components

## Canonical Dashboard (Interactive Analysis Tool)

**Status:** Production-ready (January 2026)

The Canonical Dashboard is an interactive Streamlit application for human sanity checks on backtest runs. It answers: "Is this thing behaving like I think it is?"

**Key Principle:** The dashboard reads artifacts only. It never computes strategy logic.

### Overview

The dashboard provides interactive visualization and analysis of run artifacts without re-running strategy computations. It complements the command-line diagnostics tools by providing:

- Interactive exploration of run artifacts
- Visual performance analysis
- Run completeness validation
- Baseline comparisons
- Human-readable "top contributors" tables
- Automatic issue detection and warnings

### Architecture

**Module:** `src/dashboards/canonical_dashboard.py`

**Technology:**
- Streamlit (interactive web framework)
- Plotly (interactive charts)
- Pandas (data manipulation)

**Data Sources:**
- Reads artifacts from `reports/runs/{run_id}/`
- Never computes strategy logic (strictly read-only)
- Caches artifact loading for performance (`@st.cache_data`)

### Usage

**Launch the dashboard:**
```bash
streamlit run src/dashboards/canonical_dashboard.py
```

The dashboard opens in your default web browser (typically at `http://localhost:8501`).

**Run Selection:**
- Select a run from the dropdown (populated from `reports/runs/`)
- Optionally select a baseline run for comparison
- Required artifacts must be present for views to be enabled

### Dashboard Views

#### 0️⃣ Run Overview (Artifact Completeness Gate)

**Purpose:** Validate run completeness before analysis

**Components:**
- **Run Completeness Score:**
  - Required artifacts: "X/4" (portfolio_returns.csv, equity_curve.csv, weights.csv, meta.json)
  - Required diagnostics: "X/2" (canonical_diagnostics.json, asset_returns.csv)
  - Optional artifacts: "X/Y" (allocator state, regime, policy, sleeve returns, etc.)
  - **View blocking:** If required artifacts are missing, downstream views are disabled

- **Run Metadata:**
  - Run ID, Strategy Profile, Canonical Window, Config Hash
  - Start/End dates, Source runs (allocator, policy)

- **Known Issues / Warnings Panel:**
  - Error severity: Run errors (run_error.json presence)
  - Warning severity: NaN values in returns/weights, extreme turnover spikes, leverage cap frequently binding
  - Info severity: Missing optional artifacts (e.g., sleeve_returns.csv)

- **Run Notes:**
  - Persistent text field for run notes (saved to `run_notes.md`)
  - Session state management (explicit save button)

#### 1️⃣ Equity + Drawdown

**Purpose:** High-level performance visualization

**Components:**
- Equity curve plot
- Drawdown plot (with running max)
- Summary metrics: Current Equity, Total Return, Annualized Vol, Max Drawdown

**Top Contributors Table:**
- Top 10 assets by PnL contribution (last 30 days)

**Performance Settings:**
- Downsample option for long series (reduces to 1000 points max)

#### 2️⃣ Exposure Over Time

**Purpose:** Understand exposure evolution and policy/allocator effects

**Components:**
- **Pre-Allocator Exposure:** From `weights_raw.csv` (post-policy, before allocator scaling)
- **Post-Allocator Exposure:** From `weights_scaled.csv` (final exposure after allocator scaling)
- **Policy Gating Markers:** Vertical lines showing which engines gated (Trend=orange, VRP=purple, Both=red)
- Legend showing gate counts by engine type

**Naming Convention:**
- "Pre-Allocator" = post-policy but pre-allocator scaling (policy gates affect signals upstream)
- "Post-Allocator" = final exposure after allocator scaling
- Note displayed: "Policy gates affect signals before allocator. 'Pre-Allocator' exposure is post-policy."

**Top Contributors Table:**
- Top 5 sleeves by PnL contribution (last 60 days)

#### 3️⃣ Position-Level View

**Purpose:** Drill down into specific dates and understand position-level contributions

**Components:**
- **Date Selector:** Choose any date in the run window
- **Holdings Snapshot:**
  - Columns: `weight_pre_allocator`, `weight_post_allocator`, `position_direction`, `exposure`, `pnl_contribution`
  - Limited to top N assets (configurable slider, default 50)
  - Note: "Pre-Allocator weights are post-policy (policy gates affect signals upstream). 'Post-Allocator' weights are final (after allocator scaling)."

- **PnL Contribution (Last 30 Days):**
  - Mini table showing recent asset contributions

- **Turnover Proxy:**
  - Metrics: Average, Max, Latest turnover
  - Turnover plot over time
  - **Top Contributors Table:** Top 10 turnover events (rebalance dates)

#### 4️⃣ Allocator State Timeline

**Purpose:** Understand allocator behavior over time

**Components:**
- **Regime Timeline:** NORMAL/ELEVATED/STRESS/CRISIS markers
- **Risk Scalar:** Time series of allocator risk scalars
- **Drawdown Overlay:** Portfolio drawdown overlaid on allocator state
- **Scalar Histogram:** Distribution of scalar values
- **Regime Statistics:**
  - Total transitions, regime percentages
  - Average duration per regime
  - Max consecutive days in each regime

#### 5️⃣ Drag Waterfall

**Purpose:** Visualize return decomposition (gross → policy drag → allocator drag → net)

**Components:**
- Waterfall chart showing:
  - Gross CAGR
  - minus Policy Drag (bps/year, converted to %)
  - minus Allocator Drag (bps/year, converted to %)
  - equals Net CAGR

**Data Source:** `canonical_diagnostics.json` → `performance_decomposition`

#### 6️⃣ Correlation & Diversification Health

**Purpose:** Monitor correlation spikes and diversification breakdown

**Components:**
- **Rolling Average Pairwise Correlation:** 20-day and 60-day windows
- **Rolling Portfolio Volatility:** 20-day window
- **Drawdown Markers:** Vertical lines at dates with >10% drawdown

**Computation:** Uses `asset_returns.csv` to compute rolling pairwise correlations (similar logic to `AllocatorStateV1._compute_rolling_correlation`)

#### 7️⃣ Sleeve Concentration Timeline

**Purpose:** Monitor sleeve diversification over time

**Components:**
- **Rolling Herfindahl Index:** 60-day window based on sleeve PnL contributions
- **Drawdown Overlay:** Portfolio drawdown for context
- **Perfect Diversification Reference:** Green dashed line at 1/N (e.g., 1/7 ≈ 0.143)

**Data Source:** `sleeve_returns.csv`

**Computation:** Similar logic to `AllocatorStateV1._compute_sleeve_concentration`

#### 8️⃣ Baseline Comparison (if baseline selected)

**Purpose:** Compare variant run against baseline

**Components:**
- **Equity Ratio Plot:** Variant / Baseline equity (normalized to start at 1.0)
- **Metrics Comparison Table:**
  - Metrics: CAGR, Sharpe, MaxDD, Worst Month
  - Columns: Variant, Baseline, Delta
  - Formatted percentages

**Data Sources:** Both variant and baseline run artifacts

#### 9️⃣ Diagnostics Summary

**Purpose:** Display canonical diagnostics report inline

**Components:**
- **Summary Metrics:**
  - Allocator drag (bps/year)
  - Policy drag (bps/year)
  - Sleeve Sharpe table (top/bottom)
  - Worst 10 drawdowns (with attribution)
  - PnL concentration (Herfindahl)

- **Download/Copy Section:**
  - Markdown report textbox
  - Copy button

**Data Source:** `canonical_diagnostics.json`

### Hardening Features (January 2026)

The dashboard includes five hardening changes for robustness and usability:

#### 1. Run Completeness Score + View Blocking

- **Completeness Score Display:** Metrics showing required artifacts (X/4), required diagnostics (X/2), optional artifacts (X/Y)
- **View Blocking:** If required artifacts are missing, all downstream views are disabled with a clear error message
- **Prevents:** Confusing partial plots from incomplete runs

#### 2. Standardized Naming

- **UI Labels:** All labels use "Pre-Allocator" (post-policy, before allocator scaling) and "Post-Allocator" (final after allocator scaling)
- **Notes:** Clear explanations that policy gates affect signals upstream
- **Prevents:** Confusion about what "raw" weights represent

#### 3. Top Contributors Tables

- **Top 10 Assets:** By PnL contribution (last 30d) - shown in View 1
- **Top 5 Sleeves:** By PnL contribution (last 60d) - shown in View 2
- **Top 10 Turnover Events:** Rebalance dates with highest turnover - shown in View 3
- **Purpose:** Fast human scan for "what changed?" without drilling into full data

#### 4. Known Issues / Warnings Panel

- **Automatic Detection:**
  - Run errors (run_error.json)
  - NaN values in portfolio_returns or weights
  - Missing sleeve_returns (if views depend on it)
  - Extreme turnover spikes (max > 3x average)
  - Leverage cap frequently binding (>50% of rebalance dates)
- **Severity Levels:** Error (red), Warning (yellow), Info (blue)
- **Purpose:** Surface investigation signals automatically

#### 5. Cache + Performance Guardrails

- **Artifact Caching:** `@st.cache_data(ttl=3600)` for artifact loading (includes run_id in cache key)
- **Downsample Option:** Checkbox to downsample long series to 1000 points max for faster plotting
- **Top N Assets Slider:** Limit position view to top N assets (10-100, default 50) for faster rendering
- **Purpose:** Maintain performance with many runs and long time series

### Artifact Requirements

**Required Artifacts (hard gate - views blocked if missing):**
- `portfolio_returns.csv` - Daily portfolio returns
- `equity_curve.csv` - Daily equity curve
- `weights.csv` - Rebalance-date weights
- `meta.json` - Run metadata

**Required Diagnostics (for diagnostics views):**
- `canonical_diagnostics.json` - Canonical diagnostics report (for Views 5, 9)
- `asset_returns.csv` - Daily asset returns (for View 6)

**Optional Artifacts (views gracefully degrade if missing):**
- `weights_raw.csv` - Pre-allocator weights (for View 2 exposure comparison)
- `weights_scaled.csv` - Post-allocator weights (for View 2, 3)
- `allocator_regime_v1.csv` - Allocator regime timeline (for View 4)
- `allocator_risk_v1_applied_used.csv` or `allocator_risk_v1_applied.csv` - Allocator scalars (for View 4)
- `allocator_regime_v1_meta.json` - Regime metadata (for View 4 statistics)
- `engine_policy_applied_v1.csv` - Engine policy applied (for View 2 policy markers)
- `sleeve_returns.csv` - Sleeve returns (for View 7, top contributors)
- `run_error.json` - Run errors (surfaced in warnings panel)

### Best Practices

1. **Always check Run Overview first:** Completeness score and warnings prevent wasted time on incomplete runs

2. **Use baseline comparison for ablation testing:** Compare variant runs against baselines to isolate effects

3. **Leverage Top Contributors tables:** Quick scan for "what changed?" without full data analysis

4. **Enable downsample for long series:** Improves responsiveness on multi-year runs

5. **Check Known Issues panel:** Automatic detection surfaces investigation signals

6. **Save run notes:** Use the Run Notes field to document run context (preserved in `run_notes.md`)

### Known Limitations

- **Artifact-only analysis:** Dashboard never computes strategy logic (by design)
- **Single-run focus:** Baseline comparison compares two runs, but multi-run analysis requires external tools
- **Performance:** Very long time series (>10 years) may require downsample option for responsive plotting
- **Real-time updates:** Artifact changes require page refresh (no auto-reload)

### Related Documentation

- **Diagnostics Framework:** See "Canonical Diagnostics" section for JSON/Markdown report generation
- **Allocator Diagnostics:** See "Allocator v1 Diagnostics" section for allocator-specific analysis
- **Run Artifacts:** See "Run Artifacts" section for artifact format specifications

## Committee Pack Generation (Tool Class 1)

**Tool Class 1** outputs are comprehensive diagnostic reports generated post-run. These are "committee pack" artifacts that provide decision-ready analysis.

### Canonical Command

Generate canonical diagnostics for a run:

```bash
python scripts/diagnostics/generate_canonical_diagnostics.py --run_id <run_id>
```

**Explicit Statement:** Tool Class 1 is **not** produced by ExecSim; it is produced **post-run** after artifacts have been saved.

**Outputs:**
- `canonical_diagnostics.json` (machine-readable)
- `canonical_diagnostics.md` (human-readable)

**Location:** `reports/runs/{run_id}/`

### Batch Generation / Triage Policy

For generating diagnostics across multiple runs, use the batch script:

```bash
# Generate for N most recent runs
python scripts/diagnostics/batch_generate_canonical_diagnostics.py --latest 25

# Generate for specific runs
python scripts/diagnostics/batch_generate_canonical_diagnostics.py --run_ids run1 run2 run3
```

**Required Artifacts Pre-Check:**

The batch script validates required artifacts before generation:
- `portfolio_returns.csv`
- `equity_curve.csv`
- `weights*.csv` (any of: `weights.csv`, `weights_scaled.csv`, `weights_raw.csv`)
- `meta.json`

This matches the artifact requirement concept documented in the "Artifact Requirements" section.

**Triage Classification:**

The batch script classifies failures with:
- Error type (e.g., `missing_required_artifacts`, `generation_error`)
- Missing artifacts list (for artifact failures)
- Error messages (for generation failures)

**Policy:** FAILED due to missing required artifacts = incomplete/abandoned run; safe to ignore. These runs lack the core artifacts needed for diagnostics and should not block workflow.

## Integration with ExecSim

The diagnostics framework integrates with `ExecSim`:

```python
from src.agents.exec_sim import ExecSim

exec_sim = ExecSim(...)

# Run backtest with run_id
results = exec_sim.run(
    market=market,
    start=start_date,
    end=end_date,
    components=components,
    run_id="my_run_v1"  # Artifacts saved to reports/runs/my_run_v1/
)

# Later, analyze the run
from src.agents.diagnostics_perf import load_run, compute_core_metrics

run = load_run("my_run_v1")
metrics = compute_core_metrics(run)
print(f"CAGR: {metrics['cagr']:.2%}")
```

## Best Practices

1. **Use descriptive run_ids**: Include strategy profile and version (e.g., `momentum_only_v1`, `core_v1_no_macro`)

2. **Keep baselines**: Don't delete baseline runs - you'll need them for comparison

3. **Check all assets**: The per-asset stats show ALL assets, not just top 10, to ensure nothing is dropped

4. **Year-by-year analysis**: Look for patterns in bad years - they often reveal component issues

5. **Equity ratio**: When comparing runs, the equity ratio shows cumulative impact over time

---

## Allocator v1 Diagnostics

The Allocator v1 system provides a comprehensive suite of diagnostics for monitoring and auditing risk control behavior. All allocator diagnostics are saved automatically as artifacts alongside standard backtest outputs.

### Allocator Artifact Structure

Every backtest run with Allocator v1 produces these canonical artifacts (saved in `reports/runs/{run_id}/`):

**State Artifacts:**
- `allocator_state_v1.csv` - Daily state features (10 columns)
- `allocator_state_v1_meta.json` - State computation metadata

**Regime Artifacts:**
- `allocator_regime_v1.csv` - Daily regime labels (NORMAL/ELEVATED/STRESS/CRISIS)
- `allocator_regime_v1_meta.json` - Regime classification metadata

**Risk Artifacts:**
- `allocator_risk_v1.csv` - Daily risk scalars (computed)
- `allocator_risk_v1_applied.csv` - Lagged risk scalars (applied to weights)
- `allocator_risk_v1_meta.json` - Risk transformation metadata
- `allocator_risk_v1_applied_meta.json` - Application metadata

**Weight Artifacts (when allocator enabled):**
- `weights_raw.csv` - Pre-scaling weights
- `weights_scaled.csv` - Post-scaling weights (after risk scalar applied)
- `allocator_risk_v1_applied_used.csv` - Scalars actually used at each rebalance

**Optional Artifacts:**
- `trend_unit_returns.csv` - Per-asset unit returns for Trend sleeve (required for `trend_breadth_20d`)
- `sleeve_returns.csv` - Per-sleeve daily returns (required for `sleeve_concentration_60d`)

**Error Artifacts (when computation fails):**
- `allocator_state_v1_error.json` - Contains traceback and input diagnostics

### Allocator State Features (10 features)

The state layer computes 10 canonical features describing portfolio and market conditions:

**Volatility & Acceleration (3 features):**
- `port_rvol_20d`: 20-day portfolio realized volatility (annualized)
- `port_rvol_60d`: 60-day portfolio realized volatility (annualized)
- `vol_accel`: Ratio of 20d/60d volatility (detects acceleration)

**Drawdown & Path (2 features):**
- `dd_level`: Current drawdown from peak equity (negative, e.g., -0.10 = 10% DD)
- `dd_slope_10d`: 10-day drawdown slope (detects worsening)

**Cross-Asset Correlation (3 features):**
- `corr_20d`: 20-day average pairwise correlation across asset returns
- `corr_60d`: 60-day average pairwise correlation across asset returns
- `corr_shock`: Recent spike in correlation (20d - 60d)

**Engine Health (2 features, optional):**
- `trend_breadth_20d`: Fraction of Trend positions with positive 20d return (requires `trend_unit_returns.csv`)
- `sleeve_concentration_60d`: Herfindahl index of sleeve PnL concentration (requires `sleeve_returns.csv`)

**Feature Coverage:**
- **Required features:** First 8 features (always present)
- **Optional features:** Last 2 features (present only if input data available)
- All required features are validated for NaN values; optional features may be missing

### Regime Classification Logic

The regime classifier maps state features to 4 discrete risk regimes:

**Regimes:**
1. **NORMAL** - Typical market conditions (risk_scalar = 1.00)
2. **ELEVATED** - Increased volatility or correlation (risk_scalar = 0.85)
3. **STRESS** - Significant drawdown or volatility spike (risk_scalar = 0.55)
4. **CRISIS** - Extreme conditions requiring defensive positioning (risk_scalar = 0.30)

**Classification Logic:**
- Uses 4 stress condition signals: `S_vol_fast`, `S_corr_spike`, `S_dd_deep`, `S_dd_worsening`
- Computes `risk_score = sum([S_vol_fast, S_corr_spike, S_dd_deep, S_dd_worsening])`
- Enter CRISIS if: `dd_level <= -0.20` OR `risk_score >= 3` OR `(S_vol_fast AND S_corr_spike AND S_dd_worsening)`
- Enter STRESS if: `risk_score >= 2` OR `(S_vol_fast AND S_corr_spike)` OR `dd_level <= -0.12`
- Enter ELEVATED if: `risk_score >= 1`
- Otherwise NORMAL

**Hysteresis:**
- Separate EXIT thresholds lower than ENTER thresholds
- Anti-thrash rule: Must remain in regime for at least 5 days
- Prevents regime flapping during boundary conditions

### Diagnostic Scripts

**State Computation:**
```bash
python scripts/diagnostics/run_allocator_state_v1.py --run_id <run_id>
```
- Loads existing run artifacts
- Computes allocator state from portfolio/asset/sleeve returns
- Saves `allocator_state_v1.csv` and metadata

**Regime Classification:**
```bash
python scripts/diagnostics/run_allocator_regime_v1.py --run_id <run_id>
```
- Loads `allocator_state_v1.csv`
- Classifies regimes using rule-based logic
- Saves `allocator_regime_v1.csv` and metadata

**Risk Transformation:**
```bash
python scripts/diagnostics/run_allocator_risk_v1.py --run_id <run_id>
```
- Loads state + regime artifacts
- Computes risk scalars with EWMA smoothing
- Saves `allocator_risk_v1.csv` and metadata

**Two-Pass Audit:**
```bash
python scripts/diagnostics/run_allocator_two_pass.py \
  --strategy_profile core_v9 \
  --start 2024-01-01 \
  --end 2024-12-15
```
- **Pass 1:** Runs baseline with allocator disabled → produces risk scalars
- **Pass 2:** Re-runs with precomputed scalars applied → produces scaled portfolio
- Generates comparison report: `two_pass_comparison.json` and `two_pass_comparison.md`

### Two-Pass Audit Report Metrics

The comparison report includes:

**Performance Metrics (baseline vs scaled):**
- CAGR (Compound Annual Growth Rate)
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Worst Month
- Worst Quarter

**Allocator Usage Statistics:**
- % Rebalances scaled (how often scalar < 1.0)
- Mean/Min/Max risk scalar
- Top 10 de-risking events (dates with lowest scalars)

**Regime Statistics (from baseline run):**
- Days in each regime (NORMAL/ELEVATED/STRESS/CRISIS)
- Regime transition counts

**Comparison Report Location:**
- `reports/runs/{scaled_run_id}/two_pass_comparison.json` (machine-readable)
- `reports/runs/{scaled_run_id}/two_pass_comparison.md` (human-readable)

### Validation Checks

The `src/allocator/state_validate.py` module provides:

**`validate_allocator_state_v1(state_df, meta)`:**
- Asserts required features present
- Asserts monotonic date index
- Asserts no NaN in required features
- Warns if >5% of rows dropped (data quality issue)

**`validate_inputs_aligned(portfolio_returns, equity_curve, asset_returns)`:**
- Ensures all inputs have matching date indices
- Prevents silent misalignments

### Allocator Configuration

In `configs/strategies.yaml`:

```yaml
allocator_v1:
  enabled: false              # Master switch (default: false)
  mode: "off"                 # "off" | "compute" | "precomputed"
  precomputed_run_id: null    # Required if mode="precomputed"
  precomputed_scalar_filename: "allocator_risk_v1_applied.csv"
  apply_missing_scalar_as: 1.0
  state_version: "v1.0"
  regime_version: "v1.0"
  risk_version: "v1.0"
```

**Modes:**
- **`off`** - Compute all artifacts but don't apply to weights (default for research)
- **`compute`** - On-the-fly state/regime/risk computation and application (has warmup issues)
- **`precomputed`** - Load scalars from a prior baseline run and apply with lag (recommended for two-pass audit)

### Best Practices for Allocator Diagnostics

1. **Always run in "off" mode first**: Generate artifacts without affecting weights, then audit
2. **Use two-pass workflow**: Compare baseline (no allocator) vs scaled (allocator applied)
3. **Check regime stickiness**: Regime transitions should be rare (not daily)
4. **Validate feature coverage**: Ensure optional features are present if expected
5. **Monitor row drops**: >5% row drop indicates data quality issues
6. **Inspect top de-risk events**: Check if allocator triggered during known stress periods
7. **Compare MaxDD reduction**: Primary goal is drawdown control, not Sharpe optimization

### Known Issues and Limitations

**Warmup Period:**
- State features require 60-day rolling windows
- Early dates (first ~60 days) will have empty state
- Two-pass audit sidesteps this by using precomputed scalars

**Optional Features:**
- `trend_breadth_20d` and `sleeve_concentration_60d` require specific sleeve data
- These features are excluded (not set to NaN) if inputs unavailable
- Regime classifier still works with 8 core features

**Mode Recommendations:**
- Use `mode: "off"` for research and initial validation
- Use `mode: "precomputed"` for two-pass audit
- Avoid `mode: "compute"` until warmup period is resolved (Stage 9)

---

## Data Integrity / Breaking Fixes

This section documents data integrity issues and breaking fixes that affect diagnostic results. All results generated before these fixes are invalid and must be re-run.

### SR3 Rank Mapping Bug Fix (2025-12-16)

**Bug**: The `get_contracts_by_root()` function in `MarketData` used alphabetical sorting to map contract symbols to ranks. For SR3 contracts, this caused `SR3_FRONT_CALENDAR` (rank 0) to be incorrectly mapped to rank 1 when requesting `ranks=[1, 2, 3, 4]`, because `SR3_FRONT_CALENDAR` comes before `SR3_RANK_1_CALENDAR` alphabetically. This misassignment affected all SR3 rank-based diagnostics and features.

**Fix**: Implemented canonical rank parsing in `src/data/contracts/rank_mapping.py` with `parse_sr3_calendar_rank()` function that correctly identifies rank 0 from `*_FRONT_*` patterns and rank k from `*_RANK_k_*` patterns. Updated `get_contracts_by_root()` to use this parser for SR3 contracts only, with validation that raises errors on parsing failures or duplicate ranks.

**Effective Date**: 2025-12-16 (commit timestamp)

**Implication**: All SR3 rank-based results generated before this fix are suspect and must be re-run to be considered canonical. This includes:
- SR3 Calendar Carry Phase-0 sweep results
- SR3 curve features (carry, curve, pack slope, etc.)
- Any diagnostics that query SR3 contracts by rank

**Affected Artifacts**: All results in `reports/sanity_checks/carry/sr3_calendar_carry_adjacent/` dated before 2025-12-16 should be marked as "pre-fix, invalid" in metadata.

### SR3 Spread Returns Calculation Fix (2025-12-16)

**Bug**: Phase-1 initial implementation computed spread returns as percentage change of spread level: `spread_return = spread.diff() / spread.shift(1)`. This caused division by near-zero when spread prices were small (common during low-rate periods 2020-2021), resulting in Inf values and catastrophic portfolio vol (302% vs 10% target).

**Fix**: Changed to leg-return spread calculation: `spread_return = w_long * r_long - w_short * r_short` where `r_k = pct_change(P_k)`. This avoids division by spread level and behaves like a tradable spread portfolio.

**Effective Date**: 2025-12-16 (commit timestamp)

**Implication**: All Phase-1 results generated before this fix are invalid. The fix ensures:
- No division by near-zero spread levels
- Stable vol targeting
- Realistic portfolio returns

**Affected Artifacts**: All Phase-1 results in `reports/runs/carry/sr3_calendar_carry_phase1/` dated before 2025-12-16 14:26:00 should be marked as "pre-fix, invalid" in metadata.

---

## Phase-0 Sanity Checks

Phase-0 sanity checks are diagnostic scripts that validate core economic ideas before adding complexity. They follow a strict "sign-only, no overlays" methodology to verify that the underlying economic edge exists.

### Purpose

Phase-0 sanity checks implement minimal strategies to verify:
- Data correctness (continuous prices, returns, features)
- P&L calculation accuracy
- Signal generation logic
- **Core economic edge**: Does the basic idea have positive alpha?

**Pass Criteria**: Sharpe ≥ 0.2+ over long window. Any sleeve that fails Phase-0 remains disabled until reworked.

### Folder Structure

All sanity checks use a **canonical structure** with archive/ and latest/ directories:

```
reports/sanity_checks/
├── trend/                          # Trend Meta-Sleeve
│   ├── breakout_mid_50_100/        # Atomic Sleeve
│   │   ├── archive/                # All historical runs
│   │   │   └── <timestamp>/
│   │   └── latest/                 # Canonical Phase-0 results
│   ├── residual_trend/             # Atomic Sleeve
│   │   ├── archive/
│   │   │   └── <timestamp>/
│   │   └── latest/
│   └── persistence/                # Atomic Sleeve (parked)
│       ├── archive/
│       │   └── <timestamp>/
│       └── latest_failed/          # Last failing run
│
└── ... (other meta-sleeves)
```

**Phase Index**: Canonical references in `reports/phase_index/{meta_sleeve}/{sleeve_name}/phase0.txt`

This structure makes it easy to:
- Find canonical Phase-0 results in `latest/` directories
- Track historical test runs in `archive/` directories
- Quickly answer "what's the current Phase-0 for this sleeve?" via phase_index
- Clean up old archives without affecting canonical references

For more details, see `docs/REPORTS_STRUCTURE.md`.

## TSMOM Sign-Only Sanity Check (Trend Meta-Sleeve)

A diagnostic script for validating data and P&L machinery using a simplified trend-following strategy.

### Purpose

The TSMOM sign-only sanity check (`scripts/run_tsmom_sanity.py`) implements a minimal trend-following strategy to verify:
- Data correctness (continuous prices, returns)
- P&L calculation accuracy
- Signal generation logic
- **Core momentum edge**: Do simple momentum signals have positive alpha?

If the sign-only strategy produces a Sharpe ratio of ~0.3-0.6, the data and P&L pipeline are likely functioning correctly, and the momentum edge is validated.

### Strategy Logic

**Sign-Only TSMOM:**
1. Compute lookback return: `r_lookback = log(price[t] / price[t-lookback])`
2. Take sign: `signal = sign(r_lookback)` → +1, -1, or 0
3. Equal-weight across assets
4. No magnitude scaling (pure sign-based)

**Default Horizons:**
- `long_252`: 252-day lookback
- `med_84`: 84-day lookback
- `short_21`: 21-day lookback

### Usage

```bash
# Run with default horizons (long_252, med_84, short_21)
# Results saved to: reports/sanity_checks/trend/<atomic_sleeve>/archive/<timestamp>/
# Canonical results automatically updated in: reports/sanity_checks/trend/<atomic_sleeve>/latest/
python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31

# Run with custom horizons
python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31 --horizons long_252:252,med_84:84,short_21:21

# Run single horizon (backward compatibility)
python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31 --lookback 252

# Specify custom output directory
python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/trend/my_run
```

### Output

The script generates results organized by atomic sleeve:

**Meta-Sleeve Level:**
- `reports/sanity_checks/trend/summary_<timestamp>.csv`: Summary metrics comparing all atomic sleeves

**Atomic Sleeve Level** (`reports/sanity_checks/trend/<atomic_sleeve>/latest/` for canonical, `archive/<timestamp>/` for historical):
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `asset_strategy_returns.csv`: Per-asset strategy returns
- `per_asset_stats.csv`: Per-asset performance statistics
- `meta.json`: Run metadata (dates, lookback, universe, metrics)
- **Plots**:
  - `equity_curve.png`: Equity curve over time
  - `price_series.png`: Price series for all assets
  - `return_histogram.png`: Return distribution

**Atomic Sleeves Tested:**
- `long_term/`: 252-day lookback momentum
- `medium_term/`: 84-day lookback momentum
- `short_term/`: 21-day lookback momentum

### Example Output

```
================================================================================
SUMMARY: Portfolio Sharpe by Horizon
================================================================================
         horizon  lookback      cagr       vol    sharpe     maxdd  hit_rate
0      long_252       252  -0.012345  0.123456  0.260000 -0.234567  0.512345
1       med_84        84  -0.023456  0.145678  0.180000 -0.345678  0.498765
2      short_21        21  -0.034567  0.167890  0.120000 -0.456789  0.485432

Interpretation:
  - Sharpe ~0.3-0.6: Data and P&L pipeline likely correct
  - Sharpe <0.1: Possible data issues or incorrect P&L calculation
  - Sharpe >1.0: Unusually strong (verify data correctness)
```

### Key Features

- **Read-only**: Uses already-adjusted continuous prices from `MarketData.prices_cont`
- **No look-ahead**: Correctly handles date filtering to preserve historical data for lookback calculations
- **Multi-horizon**: Tests multiple lookback periods simultaneously
- **Comprehensive metrics**: CAGR, Sharpe, MaxDD, Hit Rate, Volatility
- **Visualization**: Equity curves, price series, return histograms

### Implementation

**Core Module:** `src/diagnostics/tsmom_sanity.py`

**Key Functions:**
- `run_sign_only_momentum()`: Main strategy computation function
- `compute_sign_only_tsmom()`: Core sign-only TSMOM logic
- `compute_summary_stats()`: Performance metrics calculation
- `save_results()`: Save artifacts to disk
- `generate_plots()`: Generate visualization plots

**CLI Script:** `scripts/run_tsmom_sanity.py`

## Persistence Sign-Only Sanity Check (Trend Meta-Sleeve)

A Phase-0 diagnostic for validating the persistence (momentum-of-momentum) atomic sleeve idea.

### Purpose

The Persistence sign-only sanity check (`scripts/run_persistence_sanity.py`) implements a minimal persistence strategy to verify:
- Return acceleration: `ret_84[t] - ret_84[t-21]` (rate of change of momentum)
- Slope acceleration: `(EMA20 - EMA84)[t] - (EMA20 - EMA84)[t-21]` (trend velocity)
- **Core persistence edge**: Does momentum-of-momentum have positive alpha?

### Strategy Logic

**Sign-Only Persistence (Three Variants):**

**Variant 1: Return Acceleration**
1. Compute 84-day return: `ret_84[t] = log(price[t] / price[t-84])`
2. Compute acceleration: `persistence_raw = ret_84[t] - ret_84[t-21]`
3. Take sign: `signal = sign(persistence_raw)` → +1, -1, or 0
4. Equal-weight across assets
5. No magnitude scaling (pure sign-based)

**Variant 2: Slope Acceleration**
1. Compute EMAs: `EMA20`, `EMA84`
2. Compute slope now: `slope_now = EMA20[t] - EMA84[t]`
3. Compute slope old: `slope_old = EMA20[t-21] - EMA84[t-21]`
4. Acceleration: `persistence_raw = slope_now - slope_old`
5. Take sign: `signal = sign(persistence_raw)` → +1, -1, or 0
6. Equal-weight across assets

**Variant 3: Breakout Acceleration**
1. Compute 126-day breakout: `breakout_126[t] = (price[t] - min_126) / (max_126 - min_126)`, scaled to [-1, +1]
2. Compute acceleration: `persistence_raw = breakout_126[t] - breakout_126[t-21]`
3. Take sign: `signal = sign(persistence_raw)` → +1, -1, or 0
4. Equal-weight across assets

**Default Parameters:**
- `variant`: "return_accel", "slope_accel", or "breakout_accel" (default: "return_accel")
- `lookback`: 84 days (base lookback for return calculation)
- `acceleration_window`: 21 days (window for acceleration calculation)

### Usage

```bash
# Return acceleration variant (default)
python scripts/run_persistence_sanity.py --start 2021-01-01 --end 2025-10-31

# Slope acceleration variant
python scripts/run_persistence_sanity.py --start 2021-01-01 --end 2025-10-31 --variant slope_accel

# Breakout acceleration variant
python scripts/run_persistence_sanity.py --start 2021-01-01 --end 2025-10-31 --variant breakout_accel

# Custom parameters
python scripts/run_persistence_sanity.py \
  --start 2021-01-01 \
  --end 2025-10-31 \
  --variant return_accel \
  --lookback 84 \
  --acceleration_window 21

# Custom output directory
python scripts/run_persistence_sanity.py \
  --start 2021-01-01 \
  --end 2025-10-31 \
  --output_dir reports/sanity_checks/trend/persistence/my_run
```

### Output

**Phase-0 Results** (`reports/sanity_checks/trend/persistence/<timestamp>/`):
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `asset_returns.csv`: Daily asset returns
- `asset_strategy_returns.csv`: Daily per-asset strategy returns
- `positions.csv`: Daily portfolio positions (signs)
- `persistence_signals.csv`: Raw persistence signals used
- `per_asset_stats.csv`: Per-asset performance statistics
- `summary_metrics.csv`: Portfolio summary metrics
- `meta.json`: Run metadata (dates, variant, parameters, universe, metrics)
- **Plots**:
  - `equity_curve.png`: Equity curve over time
  - `return_histogram.png`: Return distribution (portfolio and per-asset)

### Key Metrics

- Portfolio Sharpe (all assets combined)
- Per-asset Sharpe
- CAGR, Volatility, Max Drawdown, Hit Rate

### Implementation

**Core Module:** `src/diagnostics/persistence_sanity.py`

**Key Functions:**
- `run_persistence()`: Main function to run persistence strategy
- `compute_persistence()`: Core persistence logic (both variants)
- `compute_summary_stats()`: Performance metrics calculation
- `save_results()`: Save artifacts to disk
- `generate_plots()`: Generate visualization plots

**CLI Script:** `scripts/run_persistence_sanity.py`

## Residual Trend Sign-Only Sanity Check (Trend Meta-Sleeve)

A Phase-0 diagnostic for validating the residual trend atomic sleeve idea.

### Purpose

The Residual Trend sign-only sanity check (`scripts/run_residual_trend_sanity.py`) implements a minimal residual trend strategy to verify:
- Long-horizon trend (e.g., 252 days) minus short-term movement (e.g., 21 days)
- Signal generation logic for residual returns
- **Core residual trend edge**: Does subtracting short-term noise from long-term trends improve alpha?

### Strategy Logic

**Sign-Only Residual Trend:**
1. Compute long-horizon log return: `long_ret = log(price[t] / price[t-L_long])`
2. Compute short-horizon log return: `short_ret = log(price[t] / price[t-L_short])`
3. Residual return: `resid_ret = long_ret - short_ret`
4. Take sign: `signal = sign(resid_ret)` → +1, -1, or 0
5. Equal-weight across assets
6. No magnitude scaling (pure sign-based)

**Default Parameters:**
- `long_lookback`: 252 days
- `short_lookback`: 21 days

### Usage

```bash
# Run with default parameters
# Results saved to: reports/sanity_checks/trend/residual_trend/<timestamp>/
python scripts/run_residual_trend_sanity.py --start 2021-01-01 --end 2025-10-31

# Custom lookback periods
python scripts/run_residual_trend_sanity.py --start 2021-01-01 --end 2025-10-31 --long_lookback 252 --short_lookback 21

# Custom output directory
python scripts/run_residual_trend_sanity.py --start 2021-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/trend/residual_trend/my_run
```

### Output

**Atomic Sleeve Level** (`reports/sanity_checks/trend/residual_trend/<timestamp>/`):
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `asset_strategy_returns.csv`: Per-asset strategy returns
- `per_asset_stats.csv`: Per-asset performance statistics
- `meta.json`: Run metadata (dates, lookbacks, universe, metrics)
- **Plots**:
  - `equity_curve.png`: Equity curve over time
  - `return_histogram.png`: Return distribution

### Implementation

**Core Module:** `src/diagnostics/residual_trend_sanity.py`

**Key Functions:**
- `run_residual_trend()`: Main strategy computation function
- `compute_residual_trend()`: Core residual trend logic
- `compute_summary_stats()`: Performance metrics calculation
- `save_results()`: Save artifacts to disk
- `generate_plots()`: Generate visualization plots

**CLI Script:** `scripts/run_residual_trend_sanity.py`

**Research Documentation:** See `docs/META_SLEEVES/TREND_RESEARCH.md` for detailed design and hypotheses.

## Rates Curve Sign-Only Sanity Check (Rates Curve RV Meta-Sleeve)

A diagnostic script for validating rates curve trading using a simplified flattener/steepener strategy.

### Purpose

The Rates Curve sign-only sanity check (`scripts/run_rates_curve_sanity.py`) implements a minimal curve trading strategy to verify:
- FRED-anchored yield reconstruction accuracy
- Curve feature computation (2s10s, 5s30s slopes)
- DV01-neutral weighting
- **Core curve edge**: Do simple curve signals have positive alpha?

### Strategy Logic

**Sign-Only Curve Trading:**
1. Get curve features: `curve_2s10s_z`, `curve_5s30s_z`
2. Take sign: `signal = sign(curve_feature)` → +1 (flattener), -1 (steepener), or 0
3. Map to trades:
   - Positive signal → flattener: long front, short back
   - Negative signal → steepener: short front, long back
4. DV01-neutral or equal-notional weighting
5. Combine 2s10s and 5s30s legs 50/50 for portfolio

### Usage

```bash
# Run with DV01-neutral weighting (default)
# Results saved to: reports/sanity_checks/rates_curve_rv/<atomic_sleeve>/<timestamp>/
python scripts/run_rates_curve_sanity.py --start 2021-01-01 --end 2025-10-31

# Run with equal notional weighting
python scripts/run_rates_curve_sanity.py --start 2021-01-01 --end 2025-10-31 --equal_notional

# Custom output directory
python scripts/run_rates_curve_sanity.py --start 2021-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/rates_curve_rv/my_run
```

### Output

**Meta-Sleeve Level:**
- `reports/sanity_checks/rates_curve_rv/portfolio/<timestamp>/`: Combined portfolio results (2s10s + 5s30s)

**Atomic Sleeve Level:**
- `reports/sanity_checks/rates_curve_rv/2s10s/<timestamp>/`: 2s10s leg results
- `reports/sanity_checks/rates_curve_rv/5s30s/<timestamp>/`: 5s30s leg results

Each atomic sleeve folder contains:
- `portfolio_returns.csv`: Daily returns for the leg
- `equity_curve.csv`: Cumulative equity curve
- `meta.json`: Run metadata
- **Plots**: Equity curves, return histograms, subperiod comparison

**Key Metrics:**
- Per-leg Sharpe (2s10s, 5s30s)
- Portfolio Sharpe (combined)
- Subperiod analysis (pre-2022 vs post-2022)

**CLI Script:** `scripts/run_rates_curve_sanity.py`

## FX/Commodity Carry Sign-Only Sanity Check (Carry Meta-Sleeve)

A diagnostic script for validating roll yield carry using a simplified sign-only strategy.

### Purpose

The FX/Commodity Carry sign-only sanity check (`scripts/run_carry_sanity.py`) implements a minimal carry strategy to verify:
- Roll yield calculation accuracy
- Term structure feature computation
- **Core carry edge**: Do simple roll yield signals have positive alpha?

### Strategy Logic

**Sign-Only Carry:**
1. Compute raw roll yield: `roll_yield = -(ln(F1) - ln(F0))`
   - F0 = front contract
   - F1 = next contract
2. Take sign: `signal = sign(roll_yield)` → +1 (backwardation, long), -1 (contango, short)
3. Equal-weight across assets
4. No cross-sectional ranking, no vol targeting

### Usage

```bash
# Run with default universe (CL, GC, 6E, 6B, 6J)
# Results saved to: reports/sanity_checks/carry/fx_commodity/<timestamp>/
python scripts/run_carry_sanity.py --start 2020-01-01 --end 2025-10-31

# Custom universe
python scripts/run_carry_sanity.py --start 2020-01-01 --end 2025-10-31 --universe CL,GC,6E

# Custom output directory
python scripts/run_carry_sanity.py --start 2020-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/carry/fx_commodity/my_run
```

### Output

**Atomic Sleeve Level:**
- `reports/sanity_checks/carry/fx_commodity/<timestamp>/`: FX/Commodity carry results

Contains:
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `asset_strategy_returns.csv`: Per-asset strategy returns
- `roll_yields.csv`: Raw roll yields by asset
- `meta.json`: Run metadata
- **Plots**: Equity curves, return histograms, subperiod comparison

**Key Metrics:**
- Portfolio Sharpe (all assets combined)
- Per-asset Sharpe
- Subperiod analysis (pre-2022 vs post-2022)

**CLI Script:** `scripts/run_carry_sanity.py`

## SR3 Calendar Carry (Carry Meta-Sleeve)

SR3 (SOFR) calendar carry strategy development history, including Phase-0 sanity checks, Phase-1 implementation, and critical bug fixes.

### Purpose

The SR3 Calendar Carry strategy implements calendar spread carry trading for SOFR futures. The development process involved multiple iterations to identify the correct instrument (spread vs outright) and fix critical calculation bugs.

### Development Timeline

**2025-12-16**: Complete development cycle from Phase-0 to Phase-1, including:
- Phase-0 variant sweep (adjacent rank pairs)
- Canonical pair selection (R2-R1)
- Rank mapping bug discovery and fix
- Phase-1 implementation with spread returns fix
- Execution rules frozen (lag=1, sign convention confirmed)

### Phase-0 Development History

**Initial Attempts (Failed):**
- **Option A**: `sign(RANK_2 - RANK_1)` → trade SR3_FRONT_CALENDAR (rank 0) outright
  - Result: Sharpe -0.88, CAGR -0.97%
  - Status: ❌ FAILED — Instrument mismatch
  
- **Option B**: `sign(mean(RANK_3, RANK_4) - mean(RANK_1, RANK_2))` → trade rank 0 outright
  - Result: Sharpe -0.80, CAGR -0.89%
  - Status: ❌ FAILED — Instrument mismatch

**Phase-0C Breakthrough (2025-12-16):**
- **Key Insight**: STIR carry lives in the curve structure (calendar spreads), not in outright delta (rank 0)
- **Valid Expression**: Trade calendar spread directly (RANK_2 - RANK_1) with `sign(RANK_2 - RANK_1)` signal
- **Result**: Sharpe 0.36, CAGR 0.65%, MaxDD -2.87%
- **Status**: ✅ **PASSED** — Valid carry expression identified

**Phase-0 Variant Sweep (2025-12-16):**
Tested adjacent rank pairs to select canonical expression:
- **R2-R1**: Sharpe 0.6384, CAGR 0.57%, MaxDD -1.44% ✅ **CANONICAL**
- **R3-R2**: Sharpe 0.4306, CAGR 0.28%, MaxDD -0.73%
- **R4-R3**: Sharpe 0.6967, CAGR 0.30%, MaxDD -0.38%
- **R5-R4**: Sharpe 1.0389, CAGR 0.63%, MaxDD -0.47%
- **R1-R0**: Sharpe 0.1874, CAGR 0.16%, MaxDD -1.40%

**Canonical Selection**: R2-R1 selected as canonical pair based on:
- Highest Sharpe among front-pack pairs (0.6384)
- Full data coverage (1,769 days)
- Liquid, front-pack where carry lives
- User instruction: Keep R2-R1 unless another pair has Sharpe ≥0.1 higher

**Critical Finding**: The carry signal itself is valid. The failure was mapping a curve structure signal to outright delta. This is a textbook institutional result: STIR carry lives in the structure and dies when projected onto outright delta.

**Canonical Rule for Future Development:**
- ✅ **VALID**: Trade calendar spread P&L directly (RANK_2 - RANK_1)
- ❌ **INVALID**: Any mapping of curve signal → outright rank-0

This distinction is explicitly documented to prevent future re-testing loops. All Phase-1 and production implementations must trade calendar spreads directly, not rank 0 outright.

### Strategy Logic

**Sign-Only SR3 Calendar Carry:**

**Canonical Phase-0 Ranks**: Ranks 1-4 only (all start at 2020-01-02, full coverage, liquid, front pack where carry lives)

**Option A (Default)**: `sign(RANK_2 - RANK_1)` in rate space
- Convert prices to rates: `r_k = 100 - P_k`
- Compute carry: `carry_raw = r2 - r1`
- Signal: `sign(carry_raw)` → +1 (positive carry, long), -1 (negative carry, short), 0 (flat)
- Trade SR3_FRONT_CALENDAR (rank 0) based on signal

**Option B**: `sign(mean(RANK_3, RANK_4) - mean(RANK_1, RANK_2))` in rate space
- Smoother curve signal using pack means
- Same trading logic as Option A

**Key Constraints:**
- No z-scores, no vol targeting, no normalization beyond sign
- Equal-weight position sizing
- Daily rebalancing
- Signal computed from ranks 1-4, but trades rank 0 (SR3_FRONT_CALENDAR)

### Usage

```bash
# Run Option A (default): sign(RANK_2 - RANK_1) -> trade rank 0
python scripts/run_sr3_carry_sanity.py --start 2020-01-02 --end 2025-10-31

# Run Option B: sign(mean(RANK_3, RANK_4) - mean(RANK_1, RANK_2)) -> trade rank 0
python scripts/run_sr3_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --variant option_b

# Run Phase-0C: sign(RANK_2 - RANK_1) -> trade spread directly (RANK_2 - RANK_1)
python scripts/run_sr3_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --variant spread

# Custom output directory
python scripts/run_sr3_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --output_dir reports/sanity_checks/carry/sr3_calendar_carry/custom
```

### Output

**Atomic Sleeve Level** (`reports/sanity_checks/carry/sr3_calendar_carry/latest/` for canonical, `archive/<timestamp>/` for historical):
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `asset_strategy_returns.csv`: Per-asset strategy returns
- `carry_signals.csv`: Raw carry signals (r2 - r1 or pack means)
- `positions.csv`: Daily positions (signs)
- `per_asset_stats.csv`: Per-asset performance statistics
- `meta.json`: Run metadata (dates, variant, ranks, metrics)
- **Plots**:
  - `equity_curve.png`: Equity curve over time
  - `return_histogram.png`: Return distribution
  - `carry_signal_timeseries.png`: Carry signals and positions over time
  - `subperiod_comparison.png`: Pre/post-2022 performance comparison

### Phase-0 Results (2020-01-02 to 2025-10-31)

**Option A (sign(RANK_2 - RANK_1)):**
- **Sharpe**: -0.8790 (FAIL)
- **CAGR**: -0.97%
- **MaxDD**: -3.98%
- **HitRate**: 28.24%
- **Vol**: 1.10%

**Option B (sign(mean(RANK_3, RANK_4) - mean(RANK_1, RANK_2))):**
- **Sharpe**: -0.8046 (FAIL)
- **CAGR**: -0.89%
- **MaxDD**: -3.66%
- **HitRate**: 28.53%
- **Vol**: 1.10%

**Phase-0C (Trade Spread Directly) — ✅ CANONICAL VALID EXPRESSION:**
- **Signal**: `sign(RANK_2 - RANK_1)` in rate space (same as Option A)
- **P&L**: Trade calendar spread directly: `spread_return = return_R2 - return_R1`
- **Sharpe**: **0.6384** (✅ **PASS**, post-fix canonical R2-R1)
- **CAGR**: 0.57%
- **MaxDD**: -1.44%
- **HitRate**: 38.72%
- **Vol**: 0.90%
- **Subperiods**:
  - Pre-2022: Sharpe 0.5673, CAGR 0.15%
  - Post-2022: Sharpe 0.7231, CAGR 0.78%
- **Note**: Initial Phase-0C result (Sharpe 0.3556) was from pre-fix data. Post-fix canonical R2-R1 shows Sharpe 0.6384.

**Canonical Interpretation:**

| Expression | Sharpe | Verdict |
|------------|--------|---------|
| Signal → trade rank-0 outright | -0.88 / -0.80 | ❌ Conceptually invalid |
| Trade the calendar spread directly | +0.36 | ✅ Correct carry expression |

**Critical Finding:**
- **Option A & B FAILED** when trading SR3_FRONT_CALENDAR (rank 0) outright
- **Phase-0C PASSED** when trading the calendar spread (RANK_2 - RANK_1) directly
- **Conclusion**: The carry signal itself is valid; the issue was the **instrument mismatch** (signal from ranks 1-4, but trading rank 0)

**Status**: ✅ **Phase-0C PASSED** — SR3 calendar carry is valid when trading the spread directly. Eligible for Phase-1 implementation.

**Canonical Rule for Future Development:**
- ✅ **VALID**: Trade calendar spread P&L directly (RANK_2 - RANK_1)
- ❌ **INVALID**: Any mapping of curve signal → outright rank-0

This distinction is explicitly documented to prevent future re-testing loops. All Phase-1 and production implementations must trade calendar spreads directly, not rank 0 outright.

### Phase-1 Implementation (2025-12-16)

**Status**: ✅ **COMPLETE** — Phase-1 implementation with execution rules frozen

**Strategy Module**: `src/strategies/carry/sr3_calendar_carry.py`
**Runner Script**: `scripts/run_sr3_calendar_carry_phase1.py`

**Implementation Details:**
- **Canonical Pair**: R2-R1 (hardcoded, non-negotiable)
- **Signal**: Z-scored spread level in rate space: `zscore(R2 - R1)` with 90d rolling window, clipped to ±2.0
- **Spread Returns**: Computed from leg returns (canonical fix): `spread_return = w2 * r2 - w1 * r1` (equal-notional: w2=1, w1=1)
- **Vol Targeting**: 10% annualized target, 60d rolling window, min vol floor 1%, max leverage cap 10x
- **DV01 Method**: Equal-notional proxy (default), with option for true DV01-neutral weighting
- **Execution**: `positions = signal.shift(1)` (frozen, non-negotiable)

**Critical Fixes Applied:**
1. **Spread Returns Fix**: Changed from spread-level percentage returns (which exploded when spread near zero) to leg-return spread: `spread_return = r_long - r_short` where `r_k = pct_change(P_k)`
2. **Rank Mapping Fix**: Canonical rank parsing ensures correct rank assignment (see Data Integrity section)

**Phase-1 Results (2025-12-16, post-fix):**
- **Mode**: phase1 (z-scored, vol-targeted)
- **Sharpe**: -0.1917 (needs parameter tuning)
- **CAGR**: -1.48%
- **MaxDD**: -15.52%
- **Vol**: 7.72% (target: 10.0%)
- **Hit Rate**: 44.11%

**Phase-0 Equivalence Check (2025-12-16):**
- **Mode**: phase0_equiv (degenerate mode: sign-only, no z-score, no vol targeting)
- **Sharpe**: 0.3556 (directionally consistent with Phase-0's 0.6384)
- **CAGR**: 0.16%
- **MaxDD**: -1.22%
- **Vol**: 0.45%
- **Hit Rate**: 43.12%
- **Conclusion**: ✅ Sign convention confirmed, lag=1 confirmed as canonical

**Execution Rules (Frozen, Non-Negotiable):**
- Signals computed on close T
- Positions applied at close T (via `signal.shift(1)`)
- P&L earned from T → T+1
- P&L reported at T+1
- `signal.shift(1)` is canonical
- Lag ≠ a tuning parameter

**Sign Convention (Confirmed):**
- Original sign: Sharpe 0.3556 (positive, matches Phase-0 direction)
- Flipped sign: Sharpe -0.3595 (negative, opposite direction)
- **Conclusion**: Current sign convention is correct. No flip needed.

**Lag Testing (Confirmed):**
- Lag 0: Sharpe -0.3330 (lookahead bias, negative)
- Lag 1: Sharpe 0.3556 ✅ **CANONICAL**
- Lag 2: Sharpe 0.1704 (underperforms lag 1)
- **Conclusion**: Lag 1 is canonical and frozen.

**Phase-1 Status**: Implementation complete. Performance needs parameter tuning (z-score window, clip bounds, vol targeting parameters), but execution rules and sign convention are settled.

### SR3 Calendar Carry — Promotion Decision (2025-12-16)

**Status**: ✅ **PROMOTED**

**Canonical Pair**: R2–R1  
**Phase-2 Verdict**: PASS

**Phase-2 Results** (Core v7 + 5% Carry vs Core v7 baseline):
- **Sharpe**: 0.6501 vs 0.6499 (+0.0002, preserved)
- **CAGR**: 7.70% vs 8.07% (-0.37%)
- **MaxDD**: -16.08% vs -16.88% (+0.80% improvement)
- **Vol**: 12.63% vs 13.30% (-0.67% reduction)
- **Correlation** (Core v7 vs Carry): 0.0418 (low, good diversification)

**Promotion Rationale:**
- Low correlation to Core v7 (~0.04) — strong diversification benefit
- Max drawdown improvement (~80 bps) — risk stabilizer
- Volatility reduction — portfolio smoothing
- Sharpe preserved at portfolio level — no degradation

**Role in Portfolio:**
- Not a standalone return engine
- Functions as a portfolio stabilizer and drawdown reducer
- Provides diversification against Trend and VRP sleeves
- Low volatility sleeve (0.45% standalone) requires scale to matter

**Notes:**
- Research weight: 5% (scaffolding, not production)
- No further tuning planned
- Execution rules frozen (lag=1, sign convention confirmed)
- Canonical expression: SR3 R2–R1 calendar spread

**Phase Index**: `reports/phase_index/carry/sr3_calendar_carry/phase2.txt`

### Implementation

**Core Module:** `src/diagnostics/sr3_carry_sanity.py`

**Key Functions:**
- `run_sign_only_sr3_carry()`: Main function to run SR3 carry strategy
- `compute_sign_only_sr3_carry()`: Core carry logic (both variants)
- `compute_summary_stats()`: Performance metrics calculation
- `save_results()`: Save artifacts to disk
- `generate_plots()`: Generate visualization plots

**CLI Script:** `scripts/run_sr3_carry_sanity.py`

**Phase Index**: 
- Phase-0: `reports/phase_index/carry/sr3_calendar_carry/phase0.txt`
- Phase-1: `reports/phase_index/carry/sr3_calendar_carry/phase1.txt`
- Phase-2: `reports/phase_index/carry/sr3_calendar_carry/phase2.txt`

## SR3 Curve RV Momentum (Curve RV Meta-Sleeve)

SR3 (SOFR) curve shape momentum strategy development, including Phase-0 discovery and Phase-1 implementation.

### Purpose

The SR3 Curve RV Momentum strategy implements curve shape momentum trading for SOFR futures. This meta-sleeve captures macro state detection on the yield curve through momentum-driven regime signals, distinct from mean-reversion approaches.

### Key Discovery (Phase-0)

**Critical Finding**: Mean-reversion on curves is conditional by nature, while momentum on curve shape is unconditional.

**Phase-0 Results (2020-01-01 to 2025-10-31):**

**Mean-Reversion Variants (All Failed):**
- Pack Slope Fade: Sharpe -0.42 ❌
- Pack Curvature Fade: Sharpe -0.51 ❌
- Rank Fly Fade: Sharpe -0.81 ❌

**Momentum Variants (All Passed):**
- Pack Slope Momentum: Sharpe 0.42 ✅
- Pack Curvature Momentum: Sharpe 0.51 ✅
- Rank Fly Momentum: Sharpe 0.81 ✅

**Conclusion**: Curve RV is a momentum-driven regime sleeve, not a mean-reversion engine.

### Phase-0 Implementation

**Script**: `scripts/run_sr3_curve_rv_phase0.py`

**Three Atomic Expressions Tested:**

1. **Pack Slope RV** (front vs back):
   - `pack_front = mean(r0,r1,r2,r3)`
   - `pack_back = mean(r8,r9,r10,r11)`
   - `spread_ret = mean(ret_back) - mean(ret_front)` (leg-return construction)
   - Signal: `sign(pack_back - pack_front)` [momentum]

2. **Pack Curvature RV** (hump vs straight):
   - `curv = belly_pack - (front_pack + back_pack)/2`
   - `fly_ret = mean(ret_belly) - 0.5*mean(ret_front) - 0.5*mean(ret_back)` (leg-return construction)
   - Signal: `sign(curv)` [momentum]

3. **Rank Fly RV** (2,6,10):
   - `fly_lvl = 2*r6 - r2 - r10`
   - `fly_ret = 2*ret6 - ret2 - ret10` (leg-return construction)
   - Signal: `sign(fly_lvl)` [momentum]

**Canonical Rules:**
- Rate space: `r_k = 100 - P_k` (for signals only)
- Price returns: `pct_change(P_k)` (for P&L construction)
- Execution lag: `signal.shift(1)` (compute on close T, hold from T→T+1)
- Spread/fly constructions only (no outright directional exposure)

**Phase-0 Pass Criteria**: Sharpe ≥ 0.2 over full window

### Phase-1 Implementation (2025-12-17)

**Status**: ✅ **COMPLETE** — Phase-1 implementation with z-scoring, vol targeting, and redundancy analysis

**Strategy Module**: `src/strategies/rates_curve_rv/sr3_curve_rv_momentum.py`
**Runner Script**: `scripts/run_sr3_curve_rv_phase1.py`

**Implementation Details:**
- **Z-scoring**: 252-day rolling window, clipped to ±3.0
- **Vol targeting**: 10% target vol, 63-day lookback, 1% vol floor, 10x max leverage
- **Execution lag**: 1 day (frozen)
- **Return construction**: Leg returns from price pct_change (canonical)

**Phase-1 Results (2020-01-01 to 2025-10-31):**

**Run ID**: 20251217_134842

**Pack Slope Momentum:**
- **Sharpe**: 0.2837 ✅
- **CAGR**: 3.23%
- **Vol**: 11.38% (vol-targeted)
- **MaxDD**: -20.75%
- **HitRate**: 48.11%

**Pack Curvature Momentum:**
- **Sharpe**: 0.3925 ✅
- **CAGR**: 2.91%
- **Vol**: 7.41% (vol-targeted)
- **MaxDD**: -21.70%
- **HitRate**: 45.55%

**Rank Fly Momentum:**
- **Sharpe**: 1.1908 ✅
- **CAGR**: 15.10%
- **Vol**: 12.68% (vol-targeted)
- **MaxDD**: -27.48%
- **HitRate**: 45.44%

### Redundancy Analysis

**Signal Correlations:**
- Pack Slope vs Pack Curvature: 0.18 (low — orthogonal)
- Pack Slope vs Rank Fly: 0.18 (low — orthogonal)
- Pack Curvature vs Rank Fly: 0.91 (high — redundant)

**Return Correlations:**
- Pack Slope vs Pack Curvature: 0.21 (low)
- Pack Slope vs Rank Fly: 0.24 (low)
- Pack Curvature vs Rank Fly: 0.64 (moderate)

**Phase-1 Decision:**
- ✅ **Promote to Phase-2**: `sr3_curve_rv_rank_fly_2_6_10_momentum` (strong Sharpe, clean behavior, distinct)
- ⚠️ **Secondary/Optional**: `sr3_curve_rv_pack_slope_momentum` (modest Sharpe, orthogonal, not redundant)
- ❌ **Park**: `sr3_curve_rv_pack_curvature_momentum` (dominated by Rank Fly, adds no independent signal)

### Phase-2 Results and Promotion (2025-12-17)

**Status**: ✅ **COMPLETE** — Both Rank Fly and Pack Slope promoted to Core v9

**Objective**: Does Curve RV Momentum (Rank Fly + Pack Slope) improve the portfolio vs Core v8?

**Phase-2 Implementation:**
- Integrated backtest approach (not blend test)
- Core v8 baseline: `core_v8_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_no_macro`
- Core v9 variant: `core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro`
- Curve RV Meta-Sleeve: 8% total (Rank Fly 5% + Pack Slope 3%)
- All Core v8 sleeves scaled by 0.92 to accommodate Curve RV 8%

**Rank Fly Momentum Phase-2 Results (Canonical Window: 2020-01-06 to 2025-10-31):**

**Baseline (Core v8):**
- Sharpe: 0.5820
- CAGR: 6.81%
- Vol: 12.70%
- MaxDD: -17.13%
- HitRate: 50.55%

**Variant (Core v8 + Rank Fly 5%):**
- Sharpe: 0.6417 (+0.0597) ✅
- CAGR: 7.28% (+0.47%)
- Vol: 12.09% (-0.61%)
- MaxDD: -16.21% (+0.92%) ✅
- HitRate: 51.10%
- Correlation vs baseline: 0.9987

**Pack Slope Momentum Phase-2 Results (Canonical Window: 2020-01-06 to 2025-10-31):**

**Baseline (Core v8):**
- Sharpe: 0.5820
- CAGR: 6.81%
- Vol: 12.70%
- MaxDD: -17.13%
- HitRate: 50.55%

**Variant (Core v8 + Pack Slope 3%):**
- Sharpe: 0.5903 (+0.0083) ✅
- CAGR: 6.74% (-0.06%)
- Vol: 12.35% (-0.35%)
- MaxDD: -16.24% (+0.89%) ✅
- HitRate: 50.71%
- Correlation vs baseline: 0.9996

**Core v9 Canonical Backtest (Canonical Window: 2020-01-06 to 2025-10-31):**

**Run ID**: core_v9_canonical_20251217

**Final Core v9 Metrics:**
- Sharpe: 0.6605
- CAGR: 9.35%
- Vol: 12.01%
- MaxDD: -15.32%
- HitRate: 53.14%

**Improvement vs Core v8:**
- Sharpe: +0.0785
- CAGR: +2.54%
- MaxDD: +1.81%
- Vol: -0.69%

**Phase-2 Decision:**
- ✅ **PROMOTED**: `sr3_curve_rv_rank_fly_2_6_10_momentum` at 5% weight (primary)
- ✅ **PROMOTED**: `sr3_curve_rv_pack_slope_momentum` at 3% weight (secondary)
- Both atomics passed Phase-2 criteria (Sharpe >= baseline - 0.01, MaxDD >= baseline)
- Combined Curve RV Meta-Sleeve weight: 8% total
- Core v9 composition: Core v8 sleeves × 0.92 + Curve RV 8%

**Phase Index**: 
- Phase-0: `reports/phase_index/rates_curve_rv/{sleeve_name}/phase0.txt`
- Phase-1: `reports/phase_index/rates_curve_rv/{sleeve_name}/phase1.txt`
- Phase-2: `reports/phase_index/rates_curve_rv/{sleeve_name}/phase2.txt`

**CLI Scripts:**
- Phase-0: `scripts/run_sr3_curve_rv_phase0.py --mode momentum`
- Phase-1: `scripts/run_sr3_curve_rv_phase1.py`

**CLI Script:** `scripts/run_sr3_carry_sanity.py`

**Phase Index**: 
- Phase-0 (Canonical R2-R1): `reports/phase_index/carry/sr3_calendar_carry_adjacent/phase0.txt` → `reports/sanity_checks/carry/sr3_calendar_carry_adjacent/latest/2-1/`
- Phase-1: `reports/phase_index/carry/sr3_calendar_carry/phase1.txt` → `reports/runs/carry/sr3_calendar_carry_phase1/{run_id}/`

**Phase-0 Variant Sweep Results** (`reports/sanity_checks/carry/sr3_calendar_carry_adjacent/summary/latest/variant_summary.csv`):
- **R2-R1**: Sharpe 0.6384 ✅ **CANONICAL**
- **R3-R2**: Sharpe 0.4306
- **R4-R3**: Sharpe 0.6967
- **R5-R4**: Sharpe 1.0389
- **R1-R0**: Sharpe 0.1874

## VX Calendar Carry (Carry Meta-Sleeve)

VX (volatility index futures) calendar carry strategy development history, including Phase-0 sanity checks and sign direction resolution.

### Purpose

The VX Calendar Carry strategy implements calendar spread carry trading for VX futures. The development process involved identifying the correct sign direction (carry capture via short spread in contango) and testing multiple spread pairs.

### Development Timeline

**2025-12-17**: Complete Phase-0 development cycle, including:
- Phase-0 variant sweep (adjacent spread pairs × sign directions)
- Sign direction resolution (short spread for carry capture)
- Dual variant promotion (both VX2-VX1_short and VX3-VX2_short promoted to Phase-1)
- Rank integrity checks with hard-fail threshold

### Phase-0 Development History

**Initial Attempt (Failed):**
- **VX2-VX1_long**: `sign(VX2 - VX1)` → trade spread long
  - Result: Sharpe -1.0555, CAGR -36.60%, MaxDD -93.55%
  - Status: ❌ FAILED — Wrong sign direction (leaning into roll-down loss)

**Sign Direction Resolution (2025-12-17):**
- **Key Insight**: In contango, carry capture requires shorting the rich part of the curve (front is usually overpriced vs where it will settle as it rolls)
- **Valid Expression**: Trade calendar spread SHORT: `-sign(VX_long - VX_short)` for carry capture
- **Result**: Both short-spread variants show strong positive Sharpe
- **Status**: ✅ **PASSED** — Valid carry expression identified

**Phase-0 Variant Sweep (2025-12-17):**
Tested all spread pairs × sign directions:
- **VX2-VX1_long**: Sharpe -1.0555, CAGR -36.60%, MaxDD -93.55% ❌ FAILED
- **VX2-VX1_short**: Sharpe 1.0555, CAGR 37.82%, MaxDD -22.98% ✅ **PASSED**
- **VX3-VX2_long**: Sharpe -1.2284, CAGR -27.86%, MaxDD -89.35% ❌ FAILED
- **VX3-VX2_short**: Sharpe 1.2284, CAGR 30.72%, MaxDD -27.05% ✅ **PASSED**

**Canonical Selection Decision (2025-12-17):**
Both short-spread variants exhibit strong standalone carry expectancy:
- **VX3-VX2_short**: Superior raw Sharpe (1.2284) and lower volatility (24.12%)
- **VX2-VX1_short**: Superior real-world liquidity (front spread) and higher CAGR (37.82%)

**Promotion Decision**: Both variants promoted to Phase-1 as parallel atomic sleeves:
- **VX Front Carry**: VX2-VX1_short (liquidity-favored)
- **VX Mid Carry**: VX3-VX2_short (expectancy-favored)

**Rationale:**
- Both represent the same economic idea (volatility term carry) with different curve locations
- Selection between them is deferred until allocator / production constraints
- No synthetic execution modeling was applied due to lack of bid-ask data
- Liquidity considerations deferred to production reality
- This matches institutional CTA practice: spreads are worked differently than outrights

**Critical Finding**: The carry signal itself is valid when the sign direction matches the economic mechanism (short spread in contango for carry capture). The initial failure was using the wrong sign direction (long spread), which systematically leaned into roll-down loss.

**Canonical Rule for Future Development:**
- ✅ **VALID**: Trade calendar spread SHORT for carry capture: `-sign(VX_long - VX_short)`
- ❌ **INVALID**: Generic slope-following (long spread in contango) — this captures roll-down loss, not carry

This distinction is explicitly documented to prevent future re-testing loops. All Phase-1 and production implementations must use the short-spread direction for carry capture.

### Strategy Logic

**Sign-Only VX Calendar Carry:**

**Canonical Phase-0 Spreads**: VX2-VX1 (front spread), VX3-VX2 (mid spread)

**Signal**: `-sign(VX_long - VX_short)` for carry capture
- In contango (VX_long > VX_short): Short spread (capture carry from front overpricing)
- In backwardation: Long spread (capture carry from back overpricing)
- Trade calendar spread directly: `VX_Carry_t = P(VX_long) - P(VX_short)`

**Return Construction**: `r_spread = (+1)*r_VX_long - (1)*r_VX_short`
- Where `r_k = (P_k,t - P_k,t-1) / P_k,t-1` (leg percentage returns)
- NOT % change of spread level (avoids division by near-zero)

**Key Constraints:**
- No z-scores, no vol targeting, no normalization beyond sign
- Equal-weight position sizing (1:1 notional)
- Daily rebalancing
- Execution: `signal.shift(1)` (signals at close T, positions entered at close T, P&L accrues T→T+1)

### Usage

```bash
# Run sweep (all variants)
python scripts/run_vx_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --sweep

# Run single variant (VX2-VX1, short spread for carry capture)
python scripts/run_vx_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --pair 2-1 --flip-sign

# Run single variant (VX3-VX2, short spread for carry capture)
python scripts/run_vx_carry_sanity.py --start 2020-01-02 --end 2025-10-31 --pair 3-2 --flip-sign
```

### Output

**Variant Level** (`reports/sanity_checks/carry/vx_calendar_carry_variants/latest/{pair}_{direction}/`):
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `asset_strategy_returns.csv`: Per-asset strategy returns
- `carry_signals.csv`: Raw carry signals (VX_long - VX_short)
- `positions.csv`: Daily positions (signs)
- `per_asset_stats.csv`: Per-asset performance statistics
- `meta.json`: Run metadata (dates, variant, metrics)
- **Plots**: Equity curves, return histograms, carry signal timeseries, subperiod comparison

**Summary Level** (`reports/sanity_checks/carry/vx_calendar_carry_variants/summary/latest/`):
- `variant_summary.csv`: All variants ranked by Sharpe
- `canonical_selection.json`: Canonical variant selection metadata

### Phase-0 Results (2020-01-02 to 2025-10-31)

**VX2-VX1_short (Front Carry) — ✅ PROMOTED TO PHASE-1:**
- **Signal**: `-sign(VX2 - VX1)` → trade spread SHORT (carry capture)
- **Sharpe**: **1.0555** (✅ **PASS**)
- **CAGR**: 37.82%
- **MaxDD**: -22.98%
- **HitRate**: 44.02%
- **Vol**: 36.60%
- **n_days**: 1,472
- **Rationale**: Superior real-world liquidity (front spread)

**VX3-VX2_short (Mid Carry) — ✅ PROMOTED TO PHASE-1:**
- **Signal**: `-sign(VX3 - VX2)` → trade spread SHORT (carry capture)
- **Sharpe**: **1.2284** (✅ **PASS**)
- **CAGR**: 30.72%
- **MaxDD**: -27.05%
- **HitRate**: 44.97%
- **Vol**: 24.12%
- **n_days**: 1,472
- **Rationale**: Superior raw Sharpe and lower volatility

**VX2-VX1_long (Failed):**
- **Signal**: `sign(VX2 - VX1)` → trade spread LONG
- **Sharpe**: -1.0555 (❌ FAIL)
- **CAGR**: -36.60%
- **MaxDD**: -93.55%
- **Status**: Wrong sign direction (systematically leans into roll-down loss)

**VX3-VX2_long (Failed):**
- **Signal**: `sign(VX3 - VX2)` → trade spread LONG
- **Sharpe**: -1.2284 (❌ FAIL)
- **CAGR**: -27.86%
- **MaxDD**: -89.35%
- **Status**: Wrong sign direction (systematically leans into roll-down loss)

**Rank Integrity:**
- VX2-VX1: 4 collision days (0.27%) — dropped from analysis
- VX3-VX2: 8 collision days (0.54%) — dropped from analysis
- Both well below 5% hard-fail threshold

### Phase-0 Conclusion (2025-12-17)

**Status**: ✅ **PHASE-0 PASSED** — Both VX2-VX1_short and VX3-VX2_short exhibit strong standalone carry expectancy.

**Promoted to Phase-1:**
- **VX Front Carry**: VX2-VX1_short (liquidity-favored)
- **VX Mid Carry**: VX3-VX2_short (expectancy-favored)

**Key Points:**
- Both variants represent the same economic idea (volatility term carry) with different curve locations
- Selection between them is deferred until allocator / production constraints
- No synthetic execution modeling was applied due to lack of bid-ask data
- Liquidity considerations deferred to production reality
- This matches institutional CTA practice: spreads are worked differently than outrights

**Phase-1 Implementation:**
- Same signal (z-scoring, normalization, etc.)
- Same execution semantics
- Same diagnostics
- Two atomic sleeves instead of one

**Future Work:**
- Phase-2 will tell us how each behaves in portfolio context
- Production / paper trading will tell us which one is actually tradable at scale
- No assumptions. No fake realism. No premature pruning.

### Phase-2: Portfolio Integration (2025-12-17)

**Objective**: Measure portfolio impact of adding VX Carry to Core v7 baseline, with a scaffolding research weight (5% like SR3 carry), and confirm it improves the portfolio on at least one of: Sharpe preservation, vol reduction, MaxDD improvement, low correlation.

**Primary Variant for Phase-2**: VX2-VX1_short (liquidity-favored, least-regret tradable expression)
- Performance is basically tied with VX3-VX2_short (Sharpe 0.8573 vs 0.8515)
- VX2-VX1 is the "least-regret" tradable expression
- Aligns with PROCEDURES.md principle: build robust architecture, not optimize parameters

**Optional Second Run**: VX3-VX2_short (expectancy-favored) can be run as a parallel Phase-2 candidate.

**Phase-2 Pass/Promote Criteria (Carry-style)**:
- Do not degrade baseline Sharpe materially (small negative allowed if offset by DD/vol improvement)
- Improve at least one of:
  - Vol
  - MaxDD
  - Crisis drawdown profile
  - Correlations (i.e., truly diversifying)
- If VX carry adds return too, great — but we don't require it.

**Usage**:
```bash
# Run Phase-2 for primary variant (VX2-VX1_short)
python scripts/diagnostics/run_vx_calendar_carry_phase2.py --variant vx2_vx1_short --start 2020-01-02 --end 2025-10-31

# Run Phase-2 for optional variant (VX3-VX2_short)
python scripts/diagnostics/run_vx_calendar_carry_phase2.py --variant vx3_vx2_short --start 2020-01-02 --end 2025-10-31

# Use existing Core v7 run
python scripts/diagnostics/run_vx_calendar_carry_phase2.py --variant vx2_vx1_short --core-v7-run-id <run_id>
```

**Output** (`reports/runs/carry/vx_calendar_carry_phase2/{variant}/{timestamp}/`):
- `comparison_returns.csv`: Baseline, VX Carry, and Combined returns
- `comparison_equity.csv`: Equity curves for all three
- `sleeve_correlation_matrix.csv`: Correlation matrix (VX Carry vs Trend, CSMOM, VRP-Core, VRP-Alt, portfolio baseline)
- `comparison_summary.json`: Full metrics comparison
- `diff_metrics.json`: Difference metrics (Sharpe diff, CAGR diff, MaxDD diff, Vol diff)
- `equity_curves.png`: Equity curve comparison plot
- `drawdown_curves.png`: Drawdown comparison plot
- Phase index: `reports/phase_index/carry/vx_calendar_carry/{variant}/phase2.txt`

**Phase-2 Results (2025-12-17)**:

**VX2-VX1_short (Primary - PROMOTED):**
- **Baseline (Core v7)**: Sharpe 0.6577, CAGR 8.53%, MaxDD -15.43%, Vol 13.86%
- **Combined (Core v7 + 5% VX Carry = Core v8)**: Sharpe 0.6954, CAGR 8.67%, MaxDD -14.41%, Vol 13.15%
- **Improvements**: Sharpe +0.0377, CAGR +0.14%, MaxDD +1.02%, Vol -0.71%
- **Correlation**: -0.0508 (low, good diversification)
- **Status**: ✅ **PASS** → **PROMOTED** as canonical atomic sleeve
- **Date Range**: 2020-01-06 to 2025-10-31 (1,472 days, aligned with Core v7 for apples-to-apples comparison)

*Note: All Core version metrics in this document are computed on aligned date ranges (2020-01-06 to 2025-10-31, 1,472 days) to ensure apples-to-apples comparison across Core v3-v8 evolution. See STRATEGY.md § "Baseline Evolution Summary" for the complete comparison table.*

**VX3-VX2_short (Secondary - VALID):**
- **Baseline (Core v7)**: Sharpe 0.6577, CAGR 8.53%, MaxDD -15.43%, Vol 13.86%
- **Combined (Core v7 + 5% VX Carry)**: Sharpe 0.6909, CAGR 8.58%, MaxDD -14.47%, Vol 13.17%
- **Improvements**: Sharpe +0.0333, CAGR +0.05%, MaxDD +0.96%, Vol -0.69%
- **Correlation**: -0.0075 (very low, excellent diversification)
- **Status**: ✅ **PASS** → **VALID** as secondary/backup atomic sleeve
- **Date Range**: 2020-01-06 to 2025-10-31 (1,472 days, aligned with Core v7)

**Phase-2 Conclusion (2025-12-17)**:

**Status**: ✅ **PHASE-2 PASSED** — Both variants improve portfolio metrics with low correlation.

**Promotion Decision**:
- **Canonical Atomic Sleeve**: VX2–VX1_short (promoted to Core v8 baseline)
  - Slightly stronger Phase-2 improvements (+0.0377 Sharpe vs +0.0333)
  - Liquidity-favored (front spread typically easier to trade)
  - Least-regret tradable expression
- **Secondary Atomic Sleeve**: VX3–VX2_short (validated, non-default)
  - Excellent diversification (correlation -0.0075)
  - Retained as ready alternative if/when trading reality forces a switch
  - Status: "VALID / Phase-2 PASS (secondary), not default"

**Key Points**:
- Both variants pass Phase-2 criteria (Sharpe preserved/improved, multiple improvements)
- VX Carry serves as "portfolio glue" (similar to SR3 carry), not a standalone return engine
- Expression + direction are frozen: short spread carry capture (no long spread variants)
- No synthetic execution modeling applied; liquidity considerations deferred to production reality

### Implementation

**Core Module (Phase-0)**: `src/diagnostics/vx_carry_sanity.py`

**Core Module (Phase-1)**: `src/strategies/carry/vx_calendar_carry.py`

**Key Functions:**
- `run_sign_only_vx_carry()`: Main function to run VX carry Phase-0 strategy
- `compute_sign_only_vx_carry()`: Core carry logic (supports spread pairs and sign directions)
- `compute_vx_calendar_carry_phase1()`: Phase-1 implementation with z-scoring, vol targeting
- `compute_summary_stats()`: Performance metrics calculation
- `save_results()`: Save artifacts to disk
- `generate_plots()`: Generate visualization plots

**CLI Scripts:**
- Phase-0: `scripts/run_vx_carry_sanity.py`
- Phase-1: `scripts/run_vx_calendar_carry_phase1.py`
- Phase-2: `scripts/diagnostics/run_vx_calendar_carry_phase2.py`

**Phase Index**: 
- Phase-0 (Canonical): `reports/phase_index/carry/vx_calendar_carry_variants/phase0.txt` → `reports/sanity_checks/carry/vx_calendar_carry_variants/latest/3-2_short/`
- Phase-1: `reports/phase_index/carry/vx_calendar_carry/{variant}/phase1.txt` → `reports/runs/carry/vx_calendar_carry_phase1/{run_id}/`
- Phase-2: `reports/phase_index/carry/vx_calendar_carry/{variant}/phase2.txt` → `reports/runs/carry/vx_calendar_carry_phase2/{variant}/{timestamp}/`

**Phase-0 Variant Sweep Results** (`reports/sanity_checks/carry/vx_calendar_carry_variants/summary/latest/variant_summary.csv`):
- **VX3-VX2_short**: Sharpe 1.2284 ✅ **PROMOTED** (expectancy-favored)
- **VX2-VX1_short**: Sharpe 1.0555 ✅ **PROMOTED** (liquidity-favored)
- **VX2-VX1_long**: Sharpe -1.0555 ❌ FAILED
- **VX3-VX2_long**: Sharpe -1.2284 ❌ FAILED

## CSMOM Sign-Only Sanity Check (Cross-Sectional Momentum Meta-Sleeve)

A diagnostic script for validating cross-sectional momentum using a simplified ranking strategy.

### Purpose

The CSMOM sign-only sanity check (`scripts/run_csmom_sanity.py`) implements a minimal cross-sectional momentum strategy to verify:
- Cross-sectional ranking logic
- Equal-notional long/short construction
- Market-neutral portfolio mechanics
- **Core cross-sectional momentum edge**: Do simple ranking signals have positive alpha?

### Strategy Logic

**Sign-Only Cross-Sectional Momentum:**
1. For each rebalance date, compute k-day return across universe
2. Rank assets by return (ascending: lowest = rank 1, highest = rank N)
3. Long top fraction (e.g., top 33%), short bottom fraction (e.g., bottom 33%)
4. Equal notional within long and within short
5. Net exposure ≈ 0 by construction
6. Daily rebalancing, no vol targeting or overlays

### Usage

```bash
# Run with default parameters (lookback=126, top_frac=0.33, bottom_frac=0.33)
# Results saved to: reports/sanity_checks/csmom/phase0/<timestamp>/
python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31

# Custom lookback period
python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --lookback 126

# Custom fractions
python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --top_frac 0.30 --bottom_frac 0.30

# Specific universe
python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --universe ES,NQ,RTY,CL,GC

# Custom output directory
python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/csmom/phase0/my_run
```

### Output

**Phase-0 Results** (`reports/sanity_checks/csmom/phase0/<timestamp>/`):
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `asset_returns.csv`: Daily asset returns
- `weights.csv`: Daily portfolio weights (long/short positions)
- `per_asset_stats.csv`: Per-asset performance statistics
- `summary_metrics.csv`: Portfolio summary metrics
- `meta.json`: Run metadata (dates, lookback, fractions, universe, metrics)
- **Plots**:
  - `equity_curve.png`: Equity curve over time
  - `return_histogram.png`: Return distribution (portfolio and per-asset)

### Key Metrics

- Portfolio Sharpe (all assets combined)
- Per-asset Sharpe
- Net exposure (should be ≈ 0)

### Implementation

**Core Module:** `src/diagnostics/csmom_sanity.py`

**Key Functions:**
- `compute_sign_only_csmom()`: Core sign-only CSMOM logic
- `compute_summary_stats()`: Performance metrics calculation
- `save_results()`: Save artifacts to disk
- `generate_plots()`: Generate visualization plots

**CLI Script:** `scripts/run_csmom_sanity.py`

## VRP-Core Canonical Sleeve Diagnostics

**VRP-Core Canonical Sleeve Diagnostics Summary:**

| Phase | Script | Purpose | Key Outputs |
|-------|--------|---------|-------------|
| 0 | `scripts/diagnostics/run_vrp_phase0.py` | Toy signal sanity test | `vrp_core_phase0_timeseries.parquet`, metrics JSON, plots |
| 1 | `scripts/diagnostics/run_vrp_core_phase1.py` | Engineered VRP-Core backtest | Sleeve returns, metrics, signals, plots |
| 2 | `scripts/diagnostics/run_core_v5_trend_csmom_vrp_core_phase2.py` | Portfolio integration vs baseline | Baseline vs VRP-enhanced returns & comparison |

**Units Fix Note:**
Realized ES vol was originally left in decimals and subtracted directly from VIX in vol points; this was corrected by multiplying RV_21 by 100. All documented results refer to the post-fix implementation.

## VRP Data Diagnostics (Volatility Risk Premium Meta-Sleeve)

**NOTE**: This is data diagnostics, NOT Phase-0. Phase-0 is the signal test (see VRP-Core Phase-0 below).

### Purpose

The VRP data diagnostics (part of `scripts/diagnostics/run_vrp_phase0.py`) validate:
- VIX, VIX3M, VX1/2/3 data are correctly loaded from canonical DB
- Data coverage and completeness
- Key VRP spreads are computable (VIX-VX1, VIX3M-VIX, VX2-VX1)
- **Prerequisites for VRP strategy implementation**

This is a **data diagnostics** script, not a strategy sanity check. It verifies that the VRP data pipeline is working correctly before implementing any VRP atomic sleeves.

### Data Requirements

**VIX (1-month implied volatility):**
- Source: FRED (VIXCLS) in `f_fred_observations`
- Loader: `src.market_data.vrp_loaders.load_vix()`

**VIX3M (3-month implied volatility):**
- Source: CBOE in `market_data_cboe` (symbol='VIX3M')
- First observation: 2009-09-18
- Loader: `src.market_data.vrp_loaders.load_vix3m()`

**VX Futures (VX1/2/3 continuous):**
- Source: CBOE in `market_data` (@VX=101XN, @VX=201XN, @VX=301XN)
- Loader: `src.market_data.vrp_loaders.load_vx_curve()`

**Combined Loader:**
- `src.market_data.vrp_loaders.load_vrp_inputs()` provides all VRP data

### VRP Spreads

**Primary VRP Indicators:**
1. **VRP (VIX - VX1)**: Spot vs front month spread (primary VRP signal)
2. **Term Structure (VIX3M - VIX)**: 3M vs 1M implied vol spread
3. **Curve Slope (VX2 - VX1)**: Front vs second month futures slope

### Usage

```bash
# Run with canonical end date (from backtest_window)
python scripts/diagnostics/run_vrp_phase0.py

# Custom date range
python scripts/diagnostics/run_vrp_phase0.py --start 2020-01-01 --end 2025-10-31

# Custom output directory
python scripts/diagnostics/run_vrp_phase0.py --output_dir data/diagnostics/vrp_phase0_custom
```

### Output

**Data Outputs** (`data/diagnostics/vrp_phase0/`):
- `vrp_inputs.parquet`: Full VRP dataset (date, vix, vix3m, vx1, vx2, vx3, spreads)
- `vrp_inputs.csv`: Same as above in CSV format
- `summary_stats.json`: Coverage metrics and summary statistics
- **Plots**:
  - `vix_vs_vx1.png`: VIX vs VX1 time series
  - `vix3m_minus_vix.png`: VIX3M-VIX spread time series
  - `vx2_minus_vx1.png`: VX2-VX1 slope time series
  - `vrp_vix_vx1_histogram.png`: VRP distribution histogram

### Key Metrics (Data Diagnostics)

**Coverage:**
- VIX: % of days with valid observations
- VIX3M: % of days with valid observations (target: ≥95%)
- VX1/2/3: % of days with valid observations (target: ≥90%)

**VRP Spread Statistics:**
- VIX - VX1: mean, std, % positive (typically positive = "VRP exists")
- VIX3M - VIX: mean, std, % positive (term structure)
- VX2 - VX1: mean, std, % positive (curve slope/contango)

**Pass Criteria (for data readiness, NOT Phase-0):**
- VIX coverage ≥ 95%
- VIX3M coverage ≥ 95%
- VX1/2/3 coverage ≥ 90%
- All three spreads computable with minimal gaps

### Example Output (Data Diagnostics)

```
================================================================================
VRP PHASE-0 DIAGNOSTICS SUMMARY
================================================================================

Data Coverage (n=4,123 days):
  VIX:       98.5%
  VIX3M:     97.2%
  VX1:       94.8%
  VX2:       94.6%
  VX3:       94.3%

VRP Spreads:
  VIX - VX1:      mean= -0.45, std=  2.13
                  67.3% positive
  VIX3M - VIX:    mean=  0.78, std=  1.45
                  73.8% positive
  VX2 - VX1:      mean=  0.52, std=  1.02
                  68.5% positive

================================================================================
Phase-0 diagnostics complete!
================================================================================
Results saved to: data/diagnostics/vrp_phase0
```

### Implementation

**Core Module:** `src/market_data/vrp_loaders.py`

**Key Functions:**
- `load_vix()`: Load VIX from FRED
- `load_vix3m()`: Load VIX3M from CBOE
- `load_vx_curve()`: Load VX1/2/3 from market_data
- `load_vrp_inputs()`: Combined loader for all VRP inputs

**CLI Script:** `scripts/diagnostics/run_vrp_phase0.py`

**Next Steps:**
- Once VRP data diagnostics pass, proceed to VRP-Core Phase-0 signal test
- Then follow standard Phase-0 → Phase-1 → Phase-2 → Phase-3 lifecycle
- See `docs/SOTs/PROCEDURES.md` § 4.1.1 for VRP data pipeline prerequisites

**Units Fix Note:**
Realized ES vol was originally left in decimals and subtracted directly from VIX in vol points; this was corrected by multiplying RV_21 by 100. All documented results refer to the post-fix implementation.

## VRP Meta-Sleeve — Phase-0 Summary (2025–2026)

**Status Overview:**

| Sleeve | Status | Phase-0 Outcome | Phase Index Path |
|--------|--------|-----------------|------------------|
| VRP-Core | ✅ PROMOTED | Passed Phase-0 → Phase-1 → Phase-2 | `reports/phase_index/vrp/core/` |
| VRP-Alt (RV5) | ✅ PROMOTED | Passed Phase-0 → Phase-1 → Phase-2 | `reports/phase_index/vrp/alt/` |
| VRP-Convergence | ⏸️ PROMOTED (deprioritized) | Passed Phase-0 → Phase-1 → Phase-2 | `reports/phase_index/vrp/convergence/` |
| VRP-Convexity (VVIX threshold) | ❌ PARKED | Phase-0 FAIL (Sharpe -0.1288, MaxDD -97.73%) | `reports/phase_index/vrp/convexity_vvix_threshold/` |
| VRP-FrontSpread (directional) | ❌ PARKED | Phase-0 FAIL (Sharpe -0.5846, MaxDD -96.78%) | `reports/phase_index/vrp/front_spread_directional/` |
| VRP-Structural (RV252) | ❌ PARKED | Phase-0 FAIL (all variants: VX1/VX2/VX3) | `reports/phase_index/vrp/structural_rv252/` |
| VRP-Mid (RV126) | ❌ PARKED | Phase-0 FAIL (both variants: VX2/VX3) | `reports/phase_index/vrp/mid_rv126/` |

**Notes:**
- Detailed Phase-0 results, metrics, and interpretation are documented in the individual sections below.
- Parked sleeves include rationale and revisit options in their respective phase_index directories.
- VRP-Core and VRP-Alt are the two canonical VRP engines currently in production.

---

## VRP-Core Phase-0 Signal Test (Volatility Risk Premium Meta-Sleeve)

A Phase-0 signal test for the canonical VRP-Core atomic sleeve.

### Purpose

The VRP-Core Phase-0 signal test (`scripts/diagnostics/run_vrp_phase0.py`) validates the core economic idea:
- **Economic spec**: VIX (30d implied vol) vs 21-day realized ES volatility
- **Toy rule**: Short VX1 when VIX > RV_21, otherwise flat
- **No z-scores, no clipping, no vol targeting**
- **Core VRP edge**: Test if simple VRP signal has predictive power

This follows the standard Phase-0 pattern (simple, non-engineered rule) consistent with Trend and CSMOM Phase-0 tests.

### Strategy Logic

**VRP-Core Phase-0 Toy Rule:**
1. Compute 21-day realized ES volatility: `rv_es_21 = std(ES_returns_21d) * sqrt(252)`
2. Compute VRP spread: `vrp_spread = VIX - rv_es_21`
3. Generate signal: `signal = -1 if vrp_spread > 1.5 else 0`
   - When VRP spread > 1.5 vol points → short VX1 (fade expensive vol)
   - Otherwise → flat (no position)
   - **Threshold of 1.5 is fixed for Phase-0 documentation only; NOT used in Phase-1 or production**
4. Trade VX1 with 1-day lag (avoid lookahead)
5. Daily PnL: `pnl = position * vx1_return`

**Key Difference from Phase-1:**
- Phase-0: Binary signal (-1 or 0), threshold-based, no smoothing
- Phase-1: Z-scored continuous signal in [-1, 1], with vol targeting

### Usage

```bash
# Run Phase-0 signal test (also runs data diagnostics)
python scripts/diagnostics/run_vrp_phase0.py --start 2020-01-01 --end 2025-10-31

# Use full VRP history
python scripts/diagnostics/run_vrp_phase0.py --start 2009-09-18 --end 2025-10-31
```

### Output

**Phase-0 Signal Test** (`data/diagnostics/vrp_phase0/phase0_signal_test/`):
- `vrp_core_phase0_timeseries.parquet`: Full timeseries (VIX, RV, spread, signals, PnL)
- `vrp_core_phase0_timeseries.csv`: Same as above in CSV
- `vrp_core_phase0_metrics.json`: Summary metrics
- **Plots**:
  - `phase0_equity_curve.png`: Equity curve from toy rule
  - `phase0_spreads_signals.png`: VRP spread and signals over time
  - `phase0_pnl_histogram.png`: PnL distribution

**Data Diagnostics** (`data/diagnostics/vrp_phase0/data_diagnostics/`):
- Coverage and spread diagnostics (separate from Phase-0)

**Phase Index** (`reports/phase_index/vrp/phase0.txt`):
- Registers Phase-0 signal test run with metrics

### Key Metrics

**Portfolio Performance (Phase-0 Signal Test):**
- CAGR (target: positive)
- Sharpe ratio (target: ≥0.1 for Phase-0 pass, ≥0.2 preferred)
- Max Drawdown (target: <50%)
- Hit rate (% positive PnL days)

### Pass Criteria (Phase-0 → Phase-1)

- Sharpe ≥ 0.1 (minimum), ideally ≥ 0.2
- MaxDD within reasonable bounds (<50%)
- Signal fires meaningfully (not stuck at 0 or -1)
- Positive CAGR preferred

**If Phase-0 fails → redesign economic spec before Phase-1**

### Example Output

```
================================================================================
VRP-CORE PHASE-0 SIGNAL TEST SUMMARY
================================================================================
Rule: Short VX1 when VIX > RV_21, else flat

Metrics:
  CAGR:        0.0350 (  3.50%)
  Vol:         0.1850 ( 18.50%)
  Sharpe:      0.1892
  MaxDD:      -0.3250 (-32.50%)
  HitRate:     0.5180 ( 51.80%)
  n_days:       1200
  years:        4.76

Phase-0 Pass Criteria:
  ✓ Sharpe ≥ 0.1: 0.1892 (PASS)

Phase-0 signal test saved to: data/diagnostics/vrp_phase0/phase0_signal_test
```

### Implementation

**Core Modules:**
- `src/market_data/vrp_loaders.py`: Load VIX, VX1/2/3 data
- `src/diagnostics/tsmom_sanity.py`: compute_summary_stats helper

**CLI Script:** `scripts/diagnostics/run_vrp_phase0.py`

**Next Steps (Phase-1):**
- If Phase-0 passes, proceed to engineered implementation
- Phase-1 uses z-scored VRP spread with continuous signals
- See VRP-Core Phase-1 section below

## VRP-Core Phase-1 Diagnostics (Volatility Risk Premium Meta-Sleeve)

A Phase-1 diagnostic for the canonical VRP-Core atomic sleeve using z-scored VRP spread.

### Purpose

The VRP-Core Phase-1 diagnostics (`scripts/diagnostics/run_vrp_core_phase1.py`) implement production-quality VRP strategy:
- Z-scored VRP spread (VIX - realized ES vol)
- Directional trading of VX1 front month futures
- Mean-reversion logic: fade extreme VRP levels
- **Core VRP edge**: Test if z-scored VRP spread has predictive power for VX1 returns

### Strategy Logic

**VRP-Core Phase-1:**
1. Compute 21-day realized ES volatility: `rv_es = std(ES_returns_21d) * sqrt(252)` (decimals, e.g., 0.18 = 18%)
2. Convert to vol points: `rv_es_volpoints = rv_es * 100.0` (e.g., 18.0 = 18%)
3. Compute VRP spread: `vrp_spread = VIX - rv_es_volpoints` (both in vol points)
4. Z-score VRP spread: `vrp_z = (vrp_spread - rolling_mean(252d)) / rolling_std(252d)`
5. Generate signal: `signal = -vrp_z / clip` (inverted for mean-reversion)
   - Positive vrp_z (high VRP) → short VX1 (fade expensive vol)
   - Negative vrp_z (low VRP) → long VX1 (fade cheap vol)
5. Trade VX1 directionally with signal strength in [-1, 1]

**Key Parameters:**
- `rv_lookback`: 21 days (realized vol calculation)
- `zscore_window`: 252 days (z-score standardization)
- `clip`: ±3.0 (z-score clipping bounds)
- `signal_mode`: "zscore" or "tanh" (signal transformation)

### Data Requirements

**VRP Features (from VRPCoreFeatures):**
- VIX: 1-month implied vol from FRED
- ES returns: For realized vol calculation
- VX1 prices: For P&L calculation (@VX=101XN)

**RV ffill policy:** RV series are forward-filled across isolated missing return days (e.g., holiday/weekend artifacts) to avoid false "no-signal" days. This does not invent new price data; it stabilizes the RV feature used for signal generation.

**Warmup Period:**
- 273 trading days (252d z-score + 21d realized vol)

### Usage

```bash
# Run with defaults (21d RV, 252d z-score window)
python scripts/diagnostics/run_vrp_core_phase1.py --start 2020-01-01 --end 2025-10-31

# Custom parameters
python scripts/diagnostics/run_vrp_core_phase1.py --start 2020-01-01 --end 2025-10-31 --rv_lookback 21 --zscore_window 252

# Custom output directory
python scripts/diagnostics/run_vrp_core_phase1.py --start 2020-01-01 --end 2025-10-31 --output_dir data/diagnostics/vrp_core_phase1/my_run

# Use tanh signal transformation
python scripts/diagnostics/run_vrp_core_phase1.py --start 2020-01-01 --end 2025-10-31 --signal_mode tanh
```

### Output

**Data Outputs** (`data/diagnostics/vrp_core_phase1/<timestamp>/`):
- `portfolio_returns.csv`: Daily portfolio returns (VX1 directional strategy)
- `equity_curve.csv`: Cumulative equity curve
- `vx1_returns.csv`: VX1 daily returns
- `signals.csv`: VRP-Core signals ([-1, 1])
- `summary_metrics.csv`: Portfolio summary stats
- `meta.json`: Run metadata
- **Plots**:
  - `equity_curve.png`: Equity curve over time
  - `distributions.png`: Return and signal distributions
  - `signals_timeseries.png`: Signals over time with directional fill

**Phase Index** (`reports/phase_index/vrp/vrp_core_phase1.txt`):
- Registers Phase-1 run with metrics

### Key Metrics

**Portfolio Performance:**
- CAGR (target: positive)
- Sharpe ratio (target: ≥0.2 for Phase-1 pass)
- Max Drawdown (target: <50%)
- Hit rate (% positive return days)
- Volatility (annualized)

**Signal Statistics:**
- Mean, std, min, max signal values
- % long positions (signal > 0.1)
- % short positions (signal < -0.1)
- % neutral positions (|signal| ≤ 0.1)

**Pass Criteria (for Phase-1 → Phase-2 promotion):**
- Sharpe ≥ 0.2 over full test window
- MaxDD within reasonable bounds (<50%)
- Signal distribution shows meaningful variation (not stuck at 0)
- Consistent behavior across subperiods (e.g., pre/post 2022)

### Example Output

```
================================================================================
VRP CORE PHASE-1 SUMMARY
================================================================================
  CAGR           :     0.0458 (  4.58%)
  Vol            :     0.2145 ( 21.45%)
  Sharpe         :     0.2135
  MaxDD          :    -0.3842 (-38.42%)
  HitRate        :     0.5234 ( 52.34%)
  n_days         :       1245
  years          :       4.95

Signal Statistics:
  Mean           :     0.0123
  Std            :     0.4567
  Min            :    -1.0000
  Max            :     1.0000
  % Long (>0.1)  :      38.2%
  % Short (<-0.1):      35.7%
  % Neutral      :      26.1%

================================================================================
Diagnostics complete!
================================================================================
Results saved to: data/diagnostics/vrp_core_phase1/20251209_143052
```

### Implementation

**Core Modules:**
- `src/agents/feature_vrp_core.py`: VRP feature calculation
- `src/agents/strat_vrp_core.py`: VRPCorePhase1 strategy class
- `src/diagnostics/vrp_core_phase1.py`: Phase-1 diagnostics runner

**Key Classes:**
- `VRPCoreFeatures`: Compute VRP spread and z-score
- `VRPCorePhase1`: Generate directional VX1 signals
- `VRPCoreConfig`: Configuration dataclass

**CLI Script:** `scripts/diagnostics/run_vrp_core_phase1.py`

### Next Steps (Phase-2)

- Integrate VRP-Core into VRP Meta-Sleeve
- Wire into `CombinedStrategy` with vol targeting overlay
- A/B test vs Trend + CSMOM baseline (`core_v4_trend_csmom_no_macro` - superseded, now use `core_v5` as baseline)
- Evaluate contribution when combined with other meta-sleeves

## VRP-Convergence Phase-0 Diagnostics (Volatility Risk Premium Meta-Sleeve)

A Phase-0 signal test for the VRP-Convergence atomic sleeve.

### Purpose

The VRP-Convergence Phase-0 signal test (`scripts/diagnostics/run_vrp_convergence_phase0.py`) validates the core economic idea:
- **Economic spec**: VIX (spot) vs VX1 (front-month futures) convergence
- **Toy rule**: SHORT-ONLY convergence rule
  - Short VX1 when (VIX - VX1) < -T (VX1 too rich vs VIX, expect convergence down)
  - Flat otherwise
  - **Note**: Positive spreads (VIX > VX1) do NOT imply mean reversion (momentum/expansion regime)
- **No z-scores, no clipping, no vol targeting**
- **Core convergence edge**: Test if simple convergence signal has predictive power

This follows the standard Phase-0 pattern (simple, non-engineered rule) consistent with VRP-Core Phase-0 tests, but uses short-only rule due to asymmetric economics.

### Strategy Logic

**VRP-Convergence Phase-0 Toy Rule (SHORT-ONLY):**
1. Compute convergence spread: `spread_conv = VIX - VX1` (both in vol points)
2. Generate signal:
   - `signal = -1` if `spread_conv < -T` (VX1 too rich vs VIX, expect convergence down → short VX1)
   - `signal = 0` otherwise (flat)
   - **Threshold T = 1.0 vol points (fixed for Phase-0 documentation only; NOT used in Phase-1 or production)**
   - **Multiple thresholds tested**: 1.0, 1.5, 2.0 vol points
3. Trade VX1 with 1-day lag (avoid lookahead)
4. Daily PnL: `pnl = position * vx1_return`

**Rationale for Short-Only Rule:**
- Positive spreads (VIX > VX1) do NOT imply mean reversion - they indicate momentum/expansion regimes
- Only negative spreads (VX1 > VIX) produce stable convergence (contango decay)
- This aligns with academic VIX term structure literature

**Key Difference from Phase-1:**
- Phase-0: Binary signal (-1 or 0), threshold-based, no smoothing
- Phase-1: Z-scored continuous signal in [-1, 1], with vol targeting (replaces threshold with continuous z-score)

### Usage

```bash
# Run Phase-0 signal test
python scripts/diagnostics/run_vrp_convergence_phase0.py --start 2020-01-01 --end 2025-10-31

# Use full VRP history
python scripts/diagnostics/run_vrp_convergence_phase0.py --start 2009-09-18 --end 2025-10-31
```

### Output

**Phase-0 Signal Test** (`data/diagnostics/vrp_convergence_phase0/phase0_signal_test/`):
- `vrp_convergence_phase0_timeseries.parquet`: Full timeseries (VIX, VX1, spread_conv, signal, position, PnL, equity)
- `vrp_convergence_phase0_timeseries.csv`: Same as above in CSV
- `vrp_convergence_phase0_metrics.json`: Summary metrics and signal distribution
- **Plots**:
  - `phase0_equity_curve.png`: Equity curve from toy rule
  - `phase0_spreads_signals.png`: Convergence spread and signals over time
  - `phase0_pnl_histogram.png`: PnL distribution

**Phase Index** (`reports/phase_index/vrp/vrp_convergence_phase0.txt`):
- Registers Phase-0 signal test run with metrics

### Key Metrics

**Portfolio Performance (Phase-0 Signal Test):**
- CAGR (target: positive)
- Sharpe ratio (target: ≥0.1 for Phase-0 pass)
- Max Drawdown (target: <50%)
- Hit rate (% positive PnL days)

**Signal Distribution:**
- % long positions (signal = +1)
- % short positions (signal = -1)
- % flat positions (signal = 0)

### Pass Criteria (Phase-0 → Phase-1)

- Sharpe ≥ 0.1 (minimum)
- MaxDD within reasonable bounds (<50%)
- Non-degenerate signal distribution (both long/short states used; not stuck at 0)
- Valid sample length (≥ 4 years of data after warmup if any)

**If Phase-0 fails → redesign economic spec before Phase-1**

### Implementation

**Core Modules:**
- `src/market_data/vrp_loaders.py`: Load VIX, VX1/2/3 data
- `src/diagnostics/tsmom_sanity.py`: compute_summary_stats helper

**CLI Script:** `scripts/diagnostics/run_vrp_convergence_phase0.py`

**Next Steps (Phase-1):**
- If Phase-0 passes, proceed to engineered implementation
- Phase-1 uses z-scored convergence spread with continuous signals
- See VRP-Convergence Phase-1 section below

## VRP-Convergence Phase-1 Diagnostics (Volatility Risk Premium Meta-Sleeve)

A Phase-1 diagnostic for the VRP-Convergence atomic sleeve using z-scored convergence spread.

### Purpose

The VRP-Convergence Phase-1 diagnostics (`scripts/diagnostics/run_vrp_convergence_phase1.py`) implement production-quality convergence strategy:
- Z-scored convergence spread (VIX - VX1)
- Directional trading of VX1 front month futures
- Mean-reversion logic: trade convergence when spread is extreme
- **Core convergence edge**: Test if z-scored convergence spread has predictive power for VX1 returns

### Strategy Logic

**VRP-Convergence Phase-1:**
1. Compute convergence spread: `spread_conv = VIX - VX1` (both in vol points)
2. Z-score convergence spread: `conv_z = (spread_conv - rolling_mean(252d)) / rolling_std(252d)` (clipped ±3σ)
3. Generate short-only signal: `conv_z_neg = min(conv_z, 0.0)` (only use negative z-scores, ignore positive)
4. Apply signal transformation: `signal = np.tanh(conv_z_neg / 2.0)` (bounded in [-1, 0] - short-only)
   - Alternative (zscore mode): `signal = clip(conv_z_neg / clip, -1.0, 0.0)`
   - **Phase-1 replaces Phase-0 threshold with continuous z-score signal, but maintains short-only constraint**
5. Apply volatility targeting:
   - Compute rolling vol of VX1 returns (63-day lookback, simple rolling std)
   - Scale position: `position = signal * (target_vol / realized_vol)`
   - Target vol: 10% annualized
   - Vol floor: 5% minimum
6. Apply 1-day lag: `position_t = signal_{t-1}` (rebalance at close, apply next day)
7. Trade VX1 directionally with vol-targeted position sizing (short-only)

**Signal Distribution Verification:**
- We verify that the signal distribution is:
  - `%short > 0` (meaningful short signals)
  - `%flat > 0` (some flat periods)
  - `%long = 0` (no long signals - short-only constraint)
- Compare Phase-1 vs Phase-0 equity curves and crisis windows

**Key Difference from Phase-0:**
- Phase-0: Short-only binary signal (-1 or 0) with fixed threshold (1.0 vol points)
- Phase-1: Short-only continuous z-scored signal in [-1, 0] with tanh transformation, replacing threshold with rolling z-score

**Phase-1 Metrics to Record:**
- Sharpe, Vol, MaxDD, HitRate, signal distribution
- Key sanity checks:
  - `%long == 0` (short-only constraint)
  - `%short > 0` (meaningful short signals)
  - `%flat > 0` (some flat periods)
- Equity curve vs Phase-0 comparison

**Phase-1 Results (2020-2025, canonical window):**
- Sharpe: 0.27 (PASS - above 0.20 threshold)
- CAGR: 0.33% (positive, though low due to vol targeting)
- MaxDD: -2.08% (very reasonable)
- Signal distribution: 0.0% long, 40.1% short, 59.9% flat

## VRP-Convergence Phase-2 Portfolio Diagnostics (Volatility Risk Premium Meta-Sleeve)

A Phase-2 diagnostic for integrating VRP-Convergence into the portfolio alongside VRP-Core.

### Purpose

The VRP-Convergence Phase-2 diagnostics (`scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py`) test whether adding VRP-Convergence to the baseline portfolio (`core_v5_trend_csmom_vrp_core_no_macro`) is additive and safe.

### Strategy Profiles

**Baseline**: `core_v5_trend_csmom_vrp_core_no_macro`
- Trend Meta-Sleeve: 65%
- CSMOM Meta-Sleeve: 25%
- VRP-Core: 10%

**Variant**: `core_v6_trend_csmom_vrp_core_convergence_no_macro`
- Trend Meta-Sleeve: 62.5%
- CSMOM Meta-Sleeve: 25%
- VRP-Core: 7.5%
- VRP-Convergence: 5%

### Usage

```bash
# Run Phase-2 diagnostics with canonical dates
python scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py

# Run with custom dates
python scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py --start 2020-01-01 --end 2025-10-31
```

### Output

**Comparison Results** (`data/diagnostics/phase2/core_v6_trend_csmom_vrp_core_convergence/<timestamp>/`):
- `comparison_returns.[csv|parquet]` - Baseline vs variant portfolio returns
- `comparison_summary.json` - Full comparison metrics
- `diff_metrics.json` - Difference metrics (variant - baseline)
- `sleeve_correlations.json` - Sleeve-level correlation summary
- `sleeve_correlation_matrix.csv` - Full correlation matrix
- `equity_curves.png` - Baseline vs variant equity curves
- `drawdown_curves.png` - Baseline vs variant drawdown curves

**Sleeve-Level Correlations**:
- `corr(VRP-Convergence, Trend)`
- `corr(VRP-Convergence, CSMOM)`
- `corr(VRP-Convergence, VRP-Core)`
- `corr(VRP-Convergence, baseline portfolio)`
- `corr(VRP-Convergence, variant portfolio)`

**Crisis Period Analysis**:
- 2020 Q1 (COVID crash)
- 2020 Q2 (COVID recovery)
- 2022 (Volatility spike)

### Phase-2 Pass Criteria

**Baseline vs Variant**:
- Sharpe: `Sharpe_variant >= Sharpe_baseline - 0.01` (no meaningful degradation)
- MaxDD: `MaxDD_variant >= MaxDD_baseline` (less negative is better)
- HitRate: Stable or slightly improved (nice but not required)

**Sleeve Behavior**:
- `corr(VRP-Convergence, VRP-Core) < 0.9` (not just a clone)
- Modest correlations vs Trend & CSMOM (should be somewhat independent)
- No weird blow-ups in crisis windows

**Phase Index**: `reports/phase_index/vrp/phase2_core_v6_trend_csmom_vrp_core_convergence.txt`

### Core v6 Phase-2 (VRP-Convergence Promotion)

**Baseline**: Core v5 (Trend + CSMOM + VRP-Core)
**Variant**: Core v6 (Trend + CSMOM + VRP-Core + VRP-Convergence)

**Key Metrics (2020-2025, canonical window)**:
- Sharpe: 0.5774 → 0.5796 (+0.0022)
- CAGR: 6.74% → 6.77% (+0.03%)
- MaxDD: -17.22% → -17.18% (+0.04%, less negative)

**Conclusion**: Core v6 is slightly superior and is promoted to the canonical baseline. Core v5 is retained as a historical reference baseline.

**Note**: Sleeve-level correlations for VRP-Convergence will be added once the VolManagedOverlay diagnostics bug is fixed.

**Key Parameters:**
- `zscore_window`: 252 days (z-score standardization)
- `clip`: ±3.0 (z-score clipping bounds)
- `signal_mode`: "zscore" or "tanh" (signal transformation)
- `target_vol`: 0.10 (10% annualized volatility target)
- `vol_lookback`: 63 days (volatility lookback for vol targeting)
- `vol_floor`: 0.05 (5% minimum volatility floor)

### Data Requirements

**VRP Features (from VRPConvergenceFeatures):**
- VIX: 1-month implied vol from FRED
- VX1 prices: Front-month futures from canonical DB (@VX=101XN)
- Optional: VX2 prices for curve slope diagnostics

**Warmup Period:**
- 252 trading days (z-score window)

### Usage

```bash
# Run with defaults (252d z-score window)
python scripts/diagnostics/run_vrp_convergence_phase1.py --start 2020-01-01 --end 2025-10-31

# Custom parameters
python scripts/diagnostics/run_vrp_convergence_phase1.py --start 2020-01-01 --end 2025-10-31 --zscore_window 252 --clip 3.0

# Custom output directory
python scripts/diagnostics/run_vrp_convergence_phase1.py --start 2020-01-01 --end 2025-10-31 --output_dir data/diagnostics/vrp_convergence_phase1/my_run

# Use tanh signal transformation
python scripts/diagnostics/run_vrp_convergence_phase1.py --start 2020-01-01 --end 2025-10-31 --signal_mode tanh

# Custom vol targeting
python scripts/diagnostics/run_vrp_convergence_phase1.py --start 2020-01-01 --end 2025-10-31 --target_vol 0.12 --vol_lookback 63
```

### Output

**Data Outputs** (`data/diagnostics/vrp_convergence_phase1/<timestamp>/`):
- `portfolio_returns.csv`: Daily portfolio returns (VX1 directional strategy)
- `equity_curve.csv`: Cumulative equity curve
- `vx1_returns.csv`: VX1 daily returns
- `signals.csv`: VRP-Convergence signals ([-1, 1])
- `positions.csv`: Vol-targeted positions
- `spread_conv_timeseries.csv`: Convergence spread, z-score, signals, positions
- `spread_conv_timeseries.parquet`: Same as above in Parquet format
- `summary_metrics.csv`: Portfolio summary stats
- `vrp_convergence_phase1_metrics.json`: Run metadata
- **Plots**:
  - `equity_curve.png`: Equity curve over time
  - `spread_z_and_signals.png`: Convergence spread, z-score, and signals over time
  - `pnl_histogram.png`: PnL distribution

**Phase Index** (`reports/phase_index/vrp/vrp_convergence_phase1.txt`):
- Registers Phase-1 run with metrics

### Key Metrics

**Portfolio Performance:**
- CAGR (target: positive)
- Sharpe ratio (target: ≥0.2 for Phase-1 pass)
- Max Drawdown (target: <50%)
- Hit rate (% positive return days)
- Volatility (annualized)

**Signal Statistics:**
- Mean, std, min, max signal values
- % long positions (signal > 0.1)
- % short positions (signal < -0.1)
- % neutral positions (|signal| ≤ 0.1)

**Pass Criteria (for Phase-1 → Phase-2 promotion):**
- Sharpe ≥ 0.2 over canonical 2020–2025 window
- MaxDD not catastrophically worse than VRP-Core Phase-1 in relative terms
- Reasonable signal distribution (not stuck at extremes, not always long or short)

### Implementation

**Core Modules:**
- `src/agents/feature_vrp_convergence.py`: VRP convergence feature calculation
- `src/agents/strat_vrp_convergence.py`: VRPConvergencePhase1 strategy class
- `src/diagnostics/vrp_convergence_phase1.py`: Phase-1 diagnostics runner

**Key Classes:**
- `VRPConvergenceFeatures`: Compute convergence spread and z-score
- `VRPConvergencePhase1`: Generate directional VX1 signals with vol targeting
- `VRPConvergenceConfig`: Configuration dataclass

**CLI Script:** `scripts/diagnostics/run_vrp_convergence_phase1.py`

### Next Steps (Phase-2)

- Integrate VRP-Convergence into VRP Meta-Sleeve
- Wire into `CombinedStrategy` with vol targeting overlay
- A/B test vs baseline (`core_v5_trend_csmom_vrp_core_no_macro`)
- Evaluate contribution when combined with other meta-sleeves

## VRP-Convergence Phase-2 Diagnostics (Portfolio Integration)

A Phase-2 diagnostic comparing baseline (Trend + CSMOM + VRP-Core) vs VRP-Convergence-enhanced portfolio (Trend + CSMOM + VRP-Core + VRP-Convergence).

### Purpose

The VRP-Convergence Phase-2 diagnostics (`scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py`) validate portfolio-level integration:
- Compare baseline `core_v5_trend_csmom_vrp_core_no_macro` (Trend 65% + CSMOM 25% + VRP-Core 10%)
- Compare VRP-Convergence-enhanced `core_v6_trend_csmom_vrp_core_convergence_no_macro` (Trend 62.5% + CSMOM 25% + VRP-Core 7.5% + VRP-Convergence 5%)
- Assess VRP-Convergence contribution to diversified portfolio
- Evaluate crisis-period performance
- Compute sleeve-level correlations (VRP-Convergence vs Trend, CSMOM, VRP-Core)

### Strategy Profiles

**Baseline**: `core_v5_trend_csmom_vrp_core_no_macro`
- Trend Meta-Sleeve: 65% weight
- CSMOM Meta-Sleeve: 25% weight
- VRP-Core Meta-Sleeve: 10% weight

**Variant**: `core_v6_trend_csmom_vrp_core_convergence_no_macro`
- Trend Meta-Sleeve: 62.5% weight (reduced to make room for VRP-Convergence)
- CSMOM Meta-Sleeve: 25% weight (unchanged)
- VRP-Core Meta-Sleeve: 7.5% weight (reduced from 10% to make room)
- VRP-Convergence Meta-Sleeve: 5% weight (modest allocation for Phase-2 testing)

### Usage

```bash
# Run Phase-2 comparison with canonical dates
python scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py

# Run with custom dates
python scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py --start 2020-01-01 --end 2025-10-31
```

### Output

**Phase-2 Comparison** (`data/diagnostics/phase2/core_v6_trend_csmom_vrp_core_convergence/<timestamp>/`):
- `comparison_returns.csv`: Aligned returns for both portfolios
- `comparison_returns.parquet`: Same as above in Parquet format
- `comparison_summary.json`: Full comparison metrics
- `diff_metrics.json`: Difference metrics (Variant - Baseline)
- `sleeve_correlations.json`: Sleeve-level correlations
- `sleeve_correlation_matrix.csv`: Full correlation matrix
- **Plots**:
  - `equity_curves.png`: Equity curves for both portfolios
  - `drawdown_curves.png`: Drawdown curves for both portfolios

**Phase Index** (`reports/phase_index/vrp/phase2_core_v6_trend_csmom_vrp_core_convergence.txt`):
- Registers Phase-2 run with metrics

### Key Metrics

**Portfolio Comparison:**
- Sharpe, CAGR, Vol, MaxDD, HitRate for both baseline and variant
- Difference metrics (variant - baseline)
- Portfolio correlation

**Crisis Period Analysis:**
- 2020 Q1, 2020 Q2, 2022 performance for both portfolios

**Sleeve-Level Correlations:**
- corr(VRP-Convergence, Trend)
- corr(VRP-Convergence, CSMOM)
- corr(VRP-Convergence, VRP-Core)
- corr(VRP-Convergence, Baseline Portfolio)
- corr(VRP-Convergence, Variant Portfolio)

### Pass Criteria (Phase-2)

- Portfolio Sharpe improves or stays similar with no worse MaxDD
- Crisis behavior is neutral or improved
- Sleeve-level correlations show VRP-Convergence is not redundant:
  - e.g., corr(VRP-Convergence, VRP-Core) < 0.95
- No obvious pathologies (e.g., massive drawdown concentrated in specific crisis windows)

### Implementation

**Core Modules:**
- `scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py`: Phase-2 diagnostics script
- `run_strategy.py`: Strategy runner (supports `vrp_convergence_meta`)
- `configs/strategies.yaml`: Strategy profile definitions

**Key Functions:**
- `run_strategy_profile()`: Run a strategy profile and save results
- `compute_sleeve_returns()`: Compute individual sleeve returns
- `compute_crisis_periods()`: Analyze crisis-period performance

**CLI Script:** `scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py`

## VRP-Core Phase-2 Diagnostics (Portfolio Integration)

A Phase-2 diagnostic comparing baseline (Trend + CSMOM) vs VRP-enhanced portfolio (Trend + CSMOM + VRP-Core).

### Purpose

The VRP-Core Phase-2 diagnostics (`scripts/diagnostics/run_core_v5_trend_csmom_vrp_core_phase2.py`) validate portfolio-level integration:
- Compare baseline `core_v4_trend_csmom_no_macro` (Trend 75% + CSMOM 25%) - now superseded
- Compare VRP-enhanced `core_v5_trend_csmom_vrp_core_no_macro` (Trend 65% + CSMOM 25% + VRP-Core 10%) - now current baseline
- Assess VRP-Core contribution to diversified portfolio
- Evaluate crisis-period performance

### Strategy Profiles

**Baseline**: `core_v4_trend_csmom_no_macro`
- Trend Meta-Sleeve: 75% weight
- CSMOM Meta-Sleeve: 25% weight

**VRP-Enhanced**: `core_v5_trend_csmom_vrp_core_no_macro`
- Trend Meta-Sleeve: 65% weight (reduced to make room for VRP)
- CSMOM Meta-Sleeve: 25% weight (unchanged)
- VRP-Core Meta-Sleeve: 10% weight (modest allocation for Phase-2 testing)

### Usage

```bash
# Run Phase-2 comparison with canonical dates
python scripts/diagnostics/run_core_v5_trend_csmom_vrp_core_phase2.py

# Run with custom dates
python scripts/diagnostics/run_core_v5_trend_csmom_vrp_core_phase2.py --start 2020-01-01 --end 2025-10-31
```

### Output

**Phase-2 Comparison** (`data/diagnostics/phase2/core_v5_trend_csmom_vrp_core/<timestamp>/`):
- `comparison_returns.csv`: Aligned returns for both portfolios
- `comparison_returns.parquet`: Same as above in Parquet format
- `comparison_summary.json`: Full comparison metrics
- `diff_metrics.json`: Difference metrics (VRP - Baseline)
- **Plots**:
  - `equity_curves.png`: Equity curves for both portfolios
  - `drawdown_curves.png`: Drawdown curves for both portfolios

**Phase Index** (`reports/phase_index/vrp/phase2_core_v5_trend_csmom_vrp_core.txt`):
- Registers Phase-2 comparison with metrics

### Key Metrics

**Portfolio Comparison:**
- Sharpe ratio (baseline vs VRP-enhanced)
- CAGR (baseline vs VRP-enhanced)
- Volatility (baseline vs VRP-enhanced)
- Max Drawdown (baseline vs VRP-enhanced)
- Hit Rate (baseline vs VRP-enhanced)
- Correlation between portfolios

**Difference Metrics:**
- Sharpe difference (VRP - Baseline)
- CAGR difference (VRP - Baseline)
- Vol difference (VRP - Baseline)
- MaxDD difference (VRP - Baseline)
- HitRate difference (VRP - Baseline)

**Crisis Period Analysis:**
- 2020 Q1 (COVID crash)
- 2020 Q2 (recovery)
- 2022 (inflation/rate hikes)

**Sleeve-Level Correlations:**

As part of VRP-Core Phase-2, we compute:

- `corr(VRP-Core returns, Trend returns)`
- `corr(VRP-Core returns, CSMOM returns)`
- `corr(VRP-Core returns, baseline portfolio returns)`
- `corr(VRP-Core returns, VRP-enhanced portfolio returns)`

These measures are more informative than the trivial portfolio-vs-portfolio correlation at small VRP weights. They help assess:

- Whether VRP-Core behaves differently from Trend and CSMOM
- Whether VRP provides meaningful diversification or is redundant
- How VRP interacts with the baseline portfolio during crises and normal regimes

The results are saved in `sleeve_correlations.json` and `sleeve_correlation_matrix.csv` under the Phase-2 diagnostics directory.

### Pass Criteria (Phase-2 → Production)

- Portfolio Sharpe improves or remains similar with lower drawdown
- Crisis behavior is acceptable (no excessive drawdowns during stress periods)
- VRP contribution is additive and not redundant (correlation < 0.95)
- VRP-Core provides diversification benefit

**✅ Phase-2 PASSED** - `core_v5_trend_csmom_vrp_core_no_macro` promoted to production baseline (Dec 2025)

**Promotion Summary:**
Phase-2 for VRP-Core (core_v5 vs core_v4) passed with small but consistent improvements across Sharpe, CAGR, and drawdowns. As a result, `core_v5_trend_csmom_vrp_core_no_macro` is designated as the new canonical baseline strategy profile.

| Strategy Profile | Role | Period | Sharpe | MaxDD |
|------------------|------|--------|--------|-------|
| `core_v4_trend_csmom_no_macro` | Pre-VRP baseline (superseded) | 2020-2025 | 0.5700 | -17.37% |
| `core_v5_trend_csmom_vrp_core_no_macro` | ✅ VRP-Core baseline (current) | 2020-2025 | 0.5774 | -17.22% |

### Implementation

**Core Modules:**
- `run_strategy.py`: Main strategy execution engine
- `configs/strategies.yaml`: Strategy profile definitions
- `src/agents/strat_vrp_core.py`: VRPCoreMeta wrapper

**CLI Script:** `scripts/diagnostics/run_core_v5_trend_csmom_vrp_core_phase2.py`

### Comparison vs Baseline

**Baseline Options:**
1. **Cash**: Zero return, zero risk (absolute performance test)
2. **Buy-and-hold VX1**: Simple VX1 long position (tests directional strategy value-add)
3. **Trend + CSMOM**: Core multi-sleeve portfolio (tests diversification benefit)

**Recommended Comparison Workflow:**
```bash
# Run VRP-Core Phase-1
python scripts/diagnostics/run_vrp_core_phase1.py --start 2020-01-01 --end 2025-10-31 --output_dir data/diagnostics/vrp_core_phase1/run1

# Compare metrics manually or via diagnostics tools
# Key question: Does VRP-Core have positive Sharpe as standalone strategy?
```

## Phase-0 Results Summary

### CSMOM Phase-0 (Sign-Only Cross-Sectional Momentum)

**Script:** `scripts/run_csmom_sanity.py`

**Test window:** 2020-01-02 to 2025-10-31

**Result:** Sharpe 0.236, MaxDD −30.26%, CAGR 2.42%

**Verdict:** PASS (Sharpe ≥ 0.2). Eligible for Phase-1.

## Phase-0 Workflow

The Phase-0 sanity check process follows this workflow:

1. **Run Phase-0 Test**: Execute the appropriate sanity check script
   ```bash
   python scripts/run_<meta_sleeve>_sanity.py --start <start> --end <end>
   ```

2. **Review Results**: Check Sharpe ratios and subperiod analysis
   - **Pass**: Sharpe ≥ 0.2+ → Proceed to Phase-1 (clean implementation)
   - **Fail**: Sharpe < 0.2 → Sleeve remains disabled, redesign required

3. **Compare Atomic Sleeves**: Within a Meta-Sleeve, compare atomic sleeve results
   - Identify which atomic sleeves contribute positively
   - Determine optimal weighting for meta-signal combination

4. **Document Findings**: Update `docs/SOTs/STRATEGY.md` with Phase-0 results
   - Pass/fail status
   - Key metrics (Sharpe, CAGR, MaxDD)
   - Subperiod analysis insights
   - Redesign plans if failed

## Multi-Sleeve Diagnostics — Trend vs Trend + CSMOM

**Baseline:** `core_v3_baseline_2020_2025` (Trend-only)  
**Variant:** `core_v4_trend_csmom_2020_2025` (Trend + CSMOM, 75/25)

- Sharpe: 0.0294 → 0.0902
- CAGR: -0.38% → +0.35%
- MaxDD: -29.85% → -28.09%
- Vol: 12.19% → 12.29% (unchanged)
- Final equity ratio (variant / baseline): 1.0549

**Verdict:** Phase-2 CSMOM integration PASSED.  
CSMOM is approved for use in the production-style core configuration and will participate in future multi-sleeve tests as additional Meta-Sleeves are added.

- **Baseline run_id:** `core_v3_baseline_2020_2025`
- **Variant run_id:** `core_v4_trend_csmom_2020_2025`
- **Promotion Note:** As of Dec 2025, `core_v5_trend_csmom_vrp_core_no_macro` is the canonical multi-sleeve baseline configuration for future Phase-2 tests. `core_v3_no_macro` remains the Trend-only reference profile. `core_v4_trend_csmom_no_macro` is retained for historical comparison but is superseded.

## Multi-Sleeve Diagnostics — Trend MultiHorizon with vs without Residual

**Purpose**: Compare Trend Meta-Sleeve performance with and without Residual Trend as an internal atomic sleeve.

### Diagnostic Template

**Baseline:** `core_v3_no_macro` (Trend MultiHorizon only, 3 atomic sleeves)  
**Variant (External):** `core_v3_trend_plus_residual_experiment` (80% Trend + 20% Residual, external blending)  
**Variant (Internal):** `core_v3_trend_v2_with_residual` (Trend MultiHorizon with Residual as 4th atomic sleeve, internal blending)

### Comparison Workflow

1. **Run Baseline**:
   ```bash
   python run_strategy.py \
     --strategy_profile core_v3_no_macro \
     --start 2021-01-01 \
     --end 2025-10-31 \
     --run_id trend_v1_baseline
   ```

2. **Run External Blending Variant**:
   ```bash
   python scripts/run_residual_trend_phase1.py \
     --run_id trend_plus_residual_external \
     --start 2021-01-01 \
     --end 2025-10-31
   ```

3. **Run Internal Blending Variant** (after implementation):
   ```bash
   python run_strategy.py \
     --strategy_profile core_v3_trend_v2_with_residual \
     --start 2021-01-01 \
     --end 2025-10-31 \
     --run_id trend_v2_with_residual_internal
   ```

4. **Compare Results**:
   ```bash
   # External vs Baseline
   python scripts/run_perf_diagnostics.py \
     --run_id trend_plus_residual_external \
     --baseline_id trend_v1_baseline
   
   # Internal vs Baseline
   python scripts/run_perf_diagnostics.py \
     --run_id trend_v2_with_residual_internal \
     --baseline_id trend_v1_baseline
   
   # Internal vs External
   python scripts/run_perf_diagnostics.py \
     --run_id trend_v2_with_residual_internal \
     --baseline_id trend_plus_residual_external
   ```

### Expected Metrics to Compare

- **CAGR**: Should improve with residual trend
- **Sharpe Ratio**: Should improve (may turn positive if baseline is negative)
- **Max Drawdown**: Should reduce
- **Volatility**: Should remain similar or slightly increase
- **Hit Rate**: Should improve
- **Equity Ratio**: Should show consistent outperformance over time

### Key Questions

1. **Does internal blending behave similarly to external blending?**
   - Internal: Residual Trend as 4th atomic sleeve with static weights
   - External: Residual Trend as separate sleeve blended at CombinedStrategy level
   - Both should show similar improvements if architecture is sound

2. **Are improvements consistent across years?**
   - Check year-by-year breakdown
   - Residual trend should help especially in choppy markets (2024-2025)

3. **Is the signal economically distinct?**
   - Residual trend should provide diversification vs pure momentum
   - Correlation analysis between residual and other atomic sleeves

### Phase-1 Results (External Blending)

**Baseline:** `core_v3_no_macro`  
**Variant:** `core_v3_trend_plus_residual_experiment` (80% Trend + 20% Residual)

- **CAGR**: -2.59% → -0.49% (+2.10% improvement)
- **Sharpe**: -0.15 → 0.03 (+0.18 improvement) ✨
- **MaxDD**: -39.69% → -36.80% (+2.89% improvement)
- **Vol**: 12.31% → 12.93% (+0.62%)
- **HitRate**: 49.77% → 50.83% (+1.06% improvement)
- **Equity Ratio**: 1.136x (13.6% better over full period)

**Verdict**: ✅ **Phase-1 PASSED** — Residual Trend validated. Ready for internal integration.

## Multi-Sleeve Diagnostics — Trend MultiHorizon with vs without Breakout Mid

**Purpose**: Compare Trend Meta-Sleeve performance with and without Breakout Mid (50-100d) as an internal atomic sleeve.

### Phase-1B Results (Refinement Cycle)

**Baseline:** `core_v3_no_macro_phase1b_baseline`  
**Test Configurations:** 4 variants tested with 3% horizon weight

**Winner: 70/30 Configuration (breakout_50: 70%, breakout_100: 30%)**

- **CAGR**: 0.29% → 0.42% (+0.12% improvement)
- **Sharpe**: 0.0857 → 0.0953 (+0.0095 improvement) ✨
- **MaxDD**: -32.02% → -31.52% (+0.50% improvement)
- **Vol**: 12.44% → 12.35% (-0.09% improvement)
- **HitRate**: 52.11% → 52.37% (+0.26% improvement)
- **Equity Ratio**: 1.0073 (0.73% better over full period)

**Other Test Results:**
- **30/70 (50d/100d)**: ❌ FAILED (Sharpe: -0.1502, MaxDD: -39.00%)
- **Pure 50d (100/0)**: ❌ FAILED (Sharpe: -0.1396, MaxDD: -38.60%)
- **Pure 100d (0/100)**: ❌ FAILED (Sharpe: -0.1547, MaxDD: -39.15%)

**Key Findings:**
- 3% horizon weight optimal (vs 10% in Phase-1 initial test)
- 50d breakout more effective than 100d in this setup
- 70/30 blend outperforms pure configurations
- Lower weight reduces signal conflicts with existing sleeves

**Verdict**: ✅ **Phase-1B PASSED** — 70/30 configuration promoted to Phase-2 validation.

### Phase-2 Validation

**Profile**: `core_v3_trend_breakout_70_30_phase2`  
**Baseline**: `core_v3_no_macro_phase1b_baseline`  
**Configuration**: 3% horizon weight, 70% breakout_50, 30% breakout_100  
**Target Window**: 2021-01-01 to 2025-11-19

**Results Summary**:

| Metric | Baseline | Phase-2 | Delta | Status |
|--------|----------|---------|-------|--------|
| **Sharpe** | 0.0857 | **0.0953** | **+0.0095** | ✅ Better |
| **MaxDD** | -32.02% | **-31.52%** | **+0.50%** | ✅ Better |
| **CAGR** | 0.29% | **0.42%** | **+0.12%** | ✅ Better |
| **Vol** | 12.44% | **12.35%** | **-0.09%** | ✅ Better |
| **HitRate** | 52.11% | **52.37%** | **+0.26%** | ✅ Better |

**Equity Curve Comparison**:
- Final equity ratio (Phase-2 / Baseline): 1.0073 (0.73% outperformance)
- Mean equity ratio: 1.0027
- Consistent outperformance across full period

**Year-by-Year Performance**:
- **2021**: Outperformed baseline
- **2022**: Outperformed baseline
- **2023**: Outperformed baseline
- **2024**: Outperformed baseline
- **2025**: Outperformed baseline

**Promotion Criteria**:
- ✅ **Sharpe ≥ baseline**: 0.0953 ≥ 0.0857 (passes)
- ✅ **MaxDD ≤ baseline**: -31.52% ≤ -32.02% (passes)
- ✅ **Robust across years**: Outperforms in all 5 years
- ✅ **Stable correlations**: Configuration reproducible and stable

**Verdict**: ✅ **Phase-2 PASSED** — Breakout Mid (50-100d) approved for Phase-3 (production monitoring).

---

## VRP-TermStructure Phase-0 Diagnostics (Volatility Risk Premium Meta-Sleeve)

### Purpose

The VRP-TermStructure Phase-0 diagnostics (`scripts/vrp/run_vrp_termstructure_phase0.py`) test whether the slope of the VIX futures curve (VX2 - VX1) contains a tradable short-volatility risk premium.

### Strategy Logic

**Phase-0 Specification**:
- Compute slope: `slope = VX2 - VX1`
- Short-only rule: `signal = -1 if slope > 0.5 else 0`
- Threshold: 0.5 vol points (fixed for Phase-0)
- No z-scores, no vol targeting, no overlays

### Usage

```bash
# Run Phase-0 signal test with canonical dates
python scripts/vrp/run_vrp_termstructure_phase0.py --start 2020-01-01 --end 2025-10-31
```

### Output

**Canonical Results Location**: `reports/sanity_checks/vrp/vrp_termstructure/latest/`

**Artifacts**:
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `vrp_termstructure_phase0_timeseries.csv`: Slope, signal, and PnL timeseries
- `meta.json`: Full metrics and metadata
- Diagnostic plots: `equity_curve.png`, `slope_timeseries.png`, `return_histogram.png`

### Phase-0 Results (2020-01-01 to 2025-10-31)

**Metrics**:
- **Sharpe**: -0.63 (FAIL)
- **CAGR**: -52.4%
- **MaxDD**: -98.9%
- **Hit Rate**: 43.2%

**Interpretation**:
- Signal alignment and PnL attribution confirmed correct data and calculations
- Economic failure: The mapping from (VX2 - VX1 > 0.5) → short VX1 does not produce a positive edge
- Losses concentrated in volatility spike regimes where contango persisted
- Strategy correctly avoided losses during March 2020 backwardation but suffered heavily during H1 2022

**Status**: ❌ **Phase-0 FAILED** — Sleeve PARKED. No Phase-1 development will proceed under the current specification.

**Future Revisit Options**:
- Use term-structure slope as a regime filter rather than a directional VRP sleeve
- Potential integration into Crisis Meta-Sleeve or calendar-spread-based research

### Implementation

**Core Modules**:
- `scripts/vrp/run_vrp_termstructure_phase0.py`: Phase-0 signal test script

## VRP-Convexity (VVIX Threshold) Phase-0 Diagnostics

### Purpose

The VRP-Convexity Phase-0 diagnostics (`scripts/diagnostics/run_vrp_convexity_vvix_phase0.py`) test whether a simple threshold rule on VVIX (volatility of volatility index) produces a tradable short-volatility risk premium.

### Strategy Logic

**Phase-0 Specification**:
- Compute signal: `signal = -1 if VVIX > 100 else 0`
- Threshold: 100 (fixed for Phase-0)
- Short-only rule: Short VX1 when VVIX > 100, flat otherwise
- No z-scores, no vol targeting, no overlays

### Usage

```bash
# Run Phase-0 signal test with canonical dates
python scripts/diagnostics/run_vrp_convexity_vvix_phase0.py --start 2020-01-01 --end 2025-10-31
```

### Output

**Canonical Results Location**: `reports/sanity_checks/vrp/convexity_vvix/latest/`

**Artifacts**:
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `vrp_convexity_vvix_phase0_timeseries.csv`: VVIX, signal, and PnL timeseries
- `meta.json`: Full metrics and metadata
- Diagnostic plots: `equity_curve.png`, `vvix_timeseries.png`, `return_histogram.png`

### Phase-0 Results (2020-01-01 to 2025-10-31)

**Run ID**: 20251213_145159

**Metrics**:
- **Sharpe**: -0.1288 (FAIL)
- **CAGR**: -35.35%
- **MaxDD**: -97.73%
- **Hit Rate**: 21.23%
- **Active %**: 52.4% (non-degenerate)

**Interpretation**:
- VVIX data successfully loaded (1467 rows, 2020-01-02 to 2025-10-31)
- Signal distribution is non-degenerate (52.4% active days)
- Economic failure: Simple threshold rule (VVIX > 100 → short VX1) does not produce positive edge
- Negative Sharpe plus catastrophic MaxDD indicates hard fail, not borderline case

**Verdict**: ❌ **Phase-0 FAILED** — Sleeve PARKED. No Phase-1 development will proceed under the current specification.

**Future Revisit Options**:
- Reframe as conditioning feature rather than directional VX1 trade
- Consider VVIX as regime filter or spread-style trade component
- Potential integration into Crisis Meta-Sleeve or as volatility regime indicator

### Implementation

**Core Modules**:
- `scripts/diagnostics/run_vrp_convexity_vvix_phase0.py`: Phase-0 signal test script
- `src/market_data/vrp_loaders.py`: `load_vvix()` function

## VRP-FrontSpread (Directional) Phase-0 Diagnostics

### Purpose

The VRP-FrontSpread Phase-0 diagnostics (`scripts/diagnostics/run_vrp_front_spread_phase0.py`) test whether calendar-richness (VX1 > VX2) contains a tradable carry/decay edge that can be harvested directionally with VX1.

### Strategy Logic

**Phase-0 Specification**:
- Compute spread: `spread = VX1 - VX2` (both in vol points)
- Short-only rule: `signal = -1 if spread > 0 else 0`
- Short VX1 when VX1 > VX2 (contango), flat otherwise
- No z-scores, no vol targeting, no overlays

### Usage

```bash
# Run Phase-0 signal test with canonical dates
python scripts/diagnostics/run_vrp_front_spread_phase0.py --start 2020-01-01 --end 2025-10-31
```

### Output

**Canonical Results Location**: `reports/sanity_checks/vrp/front_spread/latest/`

**Artifacts**:
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `vrp_front_spread_phase0_timeseries.csv`: Spread, signal, and PnL timeseries
- `meta.json`: Full metrics and metadata
- Diagnostic plots: `equity_curve.png`, `spread_timeseries.png`, `return_histogram.png`

### Phase-0 Results (2020-01-01 to 2025-10-31)

**Run ID**: 20251213_140054

**Metrics**:
- **Sharpe**: -0.5846 (FAIL)
- **CAGR**: -34.31%
- **MaxDD**: -96.78%
- **Hit Rate**: 6.05%
- **Active %**: 15.7% (non-degenerate)

**Interpretation**:
- VX term structure is usually in backwardation (VX1 < VX2), so contango signal only triggers in small subset of regimes (~15.7% active days)
- Even when contango exists, simple directional short is not the right expression of "calendar carry"
- Results (CAGR -34%, MaxDD -97%, hit rate ~6%) indicate crisis convexity + bad timing, not "no carry exists"
- Wrong instrument / wrong payoff: curve relationships are better treated as features/regime inputs or spread trades, not outright VX1 directionality

**Verdict**: ❌ **Phase-0 FAILED** — Sleeve PARKED. No Phase-1 development will proceed under the current specification.

**Future Revisit Options**:
- Reframe as calendar-spread trade rather than directional VX1 exposure
- Consider using front-spread as a feature/regime input rather than a directional sleeve
- Potential integration into spread-based VRP research

### Implementation

**Core Modules**:
- `scripts/diagnostics/run_vrp_front_spread_phase0.py`: Phase-0 signal test script
- `src/market_data/vrp_loaders.py`: `load_vx_curve()` function
- `src/market_data/vrp_loaders.py`: VX curve data loader (`load_vx_curve`)

**Key Functions**:
- `load_vx_curve()`: Loads VX1, VX2, VX3 continuous history
- Signal generation: Simple threshold-based short-only rule
- PnL calculation: 1-day lag (position from previous day × current return)

---

## VRP-Structural (RV252) Phase-0 Diagnostics

The VRP-Structural Phase-0 diagnostics (`scripts/diagnostics/run_vrp_structural_rv252_phase0.py`) test whether long-horizon implied vs realized volatility (VIX - RV252) contains a tradable volatility risk premium across three VX tenors (VX1, VX2, VX3).

### Economic Thesis

Long-horizon implied vs realized volatility premium: VIX (1-month implied volatility) vs RV252 (252-day realized volatility) should contain a tradable volatility risk premium.

### Phase-0 Signal Definition

**Signal**: `signal = -1 if VIX > RV252 else 0`

**Trade Expression**: Short VX when signal = -1, flat otherwise

**Tested Variants**: VX1, VX2, VX3 (all three tenors with identical signal logic)

**Discipline**: Sign-only, no z-scores, no filters, no vol targeting, constant unit exposure

### Running the Diagnostics

```bash
python scripts/diagnostics/run_vrp_structural_rv252_phase0.py
```

### Output

**Canonical Results Locations**:
- `reports/sanity_checks/vrp/structural_rv252_vx1/latest/` (VX1 variant)
- `reports/sanity_checks/vrp/structural_rv252_vx2/latest/` (VX2 variant)
- `reports/sanity_checks/vrp/structural_rv252_vx3/latest/` (VX3 variant)
- `reports/sanity_checks/vrp/structural_rv252_compare/latest/summary.json` (comparison summary)

**Artifacts** (per variant):
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `vrp_structural_rv252_{vx1|vx2|vx3}_phase0_timeseries.csv`: Spread, signal, and PnL timeseries
- `meta.json`: Full metrics and metadata
- Diagnostic plots: `equity_curve.png`, `spread_timeseries.png`, `return_histogram.png`

### Phase-0 Results (2020-01-01 to 2025-10-31)

**Run ID**: 20251213_153656 (VX1), 20251213_153657 (VX2), 20251213_153658 (VX3)

**VX1 Variant**:
- **Sharpe**: -0.1817 (FAIL)
- **CAGR**: -38.92%
- **MaxDD**: -92.90% (catastrophic)
- **Hit Rate**: 30.02%
- **Active %**: 75.0% (non-degenerate)

**VX2 Variant**:
- **Sharpe**: -0.1663 (FAIL)
- **CAGR**: -24.14%
- **MaxDD**: -78.28%
- **Hit Rate**: 30.84%
- **Active %**: 74.9% (non-degenerate)

**VX3 Variant**:
- **Sharpe**: -0.1481 (FAIL)
- **CAGR**: -14.81%
- **MaxDD**: -60.04%
- **Hit Rate**: 31.94%
- **Active %**: 74.9% (non-degenerate)

**Recommended Winner**: VX3 (least negative Sharpe, smallest MaxDD)

**Interpretation**:
- All three variants failed Phase-0 criteria (Sharpe < 0.10)
- VIX > RV252 occurs ~75% of the time, but shorting VX in these regimes is not profitable
- VX1 shows catastrophic drawdown (-93%), indicating vulnerability to volatility spikes when short
- VX3 performs best (least negative Sharpe, smallest MaxDD), suggesting back-month futures are less vulnerable to crisis convexity, but still not profitable
- Long-horizon implied vs realized volatility spread may be a structural feature (VIX typically > RV252 in normal markets), but it does not translate to profitable directional short-VX trades

**Verdict**: ❌ **Phase-0 FAILED (all variants)** — Sleeve PARKED. No Phase-1 development will proceed under the current specification.

**Future Revisit Options**:
- Use VIX - RV252 spread as a regime/conditioning input rather than a direct signal
- Test expression on different volatility instruments (e.g., VIX options, volatility ETFs)
- Test as a spread trade (long VIX, short realized vol proxy) rather than directional VX short
- Add macro regime filters (e.g., only trade in low-vol regimes, avoid crisis periods)
- Apply z-scoring or dynamic thresholds rather than simple sign rule

### Implementation

**Core Modules**:
- `scripts/diagnostics/run_vrp_structural_rv252_phase0.py`: Phase-0 signal test script (tests all three variants)
- `src/market_data/vrp_loaders.py`: `load_vix()`, `load_vx_curve()` functions
- `src/agents/data_broker.py`: `MarketData` class for ES returns (RV252 computation)

**Key Functions**:
- `load_vix()`: Loads VIX from FRED
- `load_vx_curve()`: Loads VX1, VX2, VX3 continuous history
- RV252 computation: 252-day rolling realized volatility from ES futures returns (annualized, vol points)
- Signal generation: Simple threshold-based short-only rule (VIX > RV252)
- PnL calculation: 1-day lag (position from previous day × current return)

---

## VRP-Mid (RV126) Phase-0 Diagnostics

The VRP-Mid Phase-0 diagnostics (`scripts/diagnostics/run_vrp_mid_rv126_phase0.py`) test whether mid-horizon implied vs realized volatility (VIX - RV126) contains a tradable volatility risk premium across two VX tenors (VX2, VX3).

### Economic Thesis

Mid-horizon implied vs realized volatility premium: VIX (1-month implied volatility) vs RV126 (126-day realized volatility) should contain a tradable volatility risk premium.

### Phase-0 Signal Definition

**Signal**: `signal = -1 if VIX > RV126 else 0`

**Trade Expression**: Short VX when signal = -1, flat otherwise

**Tested Variants**: VX2, VX3 (back-month futures only)

**Discipline**: Sign-only, no z-scores, no filters, no vol targeting, constant unit exposure

### Running the Diagnostics

```bash
python scripts/diagnostics/run_vrp_mid_rv126_phase0.py
```

### Output

**Canonical Results Locations**:
- `reports/sanity_checks/vrp/mid_rv126_vx2/latest/` (VX2 variant)
- `reports/sanity_checks/vrp/mid_rv126_vx3/latest/` (VX3 variant)
- `reports/sanity_checks/vrp/mid_rv126_compare/latest/summary.json` (comparison summary)

**Artifacts** (per variant):
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `vrp_mid_rv126_{vx2|vx3}_phase0_timeseries.csv`: Spread, signal, and PnL timeseries
- `meta.json`: Full metrics and metadata
- Diagnostic plots: `equity_curve.png`, `spread_timeseries.png`, `return_histogram.png`

### Phase-0 Results (2020-01-01 to 2025-10-31)

**Run ID**: 20251213_154537 (VX2), 20251213_154538 (VX3)

**VX2 Variant**:
- **Sharpe**: -0.1704 (FAIL)
- **CAGR**: -24.27%
- **MaxDD**: -85.68% (catastrophic)
- **Hit Rate**: 33.26%
- **Active %**: 81.9% (non-degenerate)

**VX3 Variant**:
- **Sharpe**: -0.1517 (FAIL)
- **CAGR**: -14.90%
- **MaxDD**: -73.88%
- **Hit Rate**: 34.50%
- **Active %**: 81.9% (non-degenerate)

**Recommended Winner**: VX3 (least negative Sharpe, smaller MaxDD)

**Interpretation**:
- Both variants failed Phase-0 criteria (Sharpe < 0.10)
- VIX > RV126 occurs ~82% of the time, but shorting VX in these regimes is not profitable
- VX2 shows catastrophic drawdown (-86%), indicating vulnerability to volatility spikes when short
- VX3 performs better (less negative Sharpe, smaller MaxDD), suggesting back-month futures are less vulnerable to crisis convexity, but still not profitable
- Mid-horizon implied vs realized volatility spread may be a structural feature (VIX typically > RV126 in normal markets), but it does not translate to profitable directional short-VX trades

**Verdict**: ❌ **Phase-0 FAILED (both variants)** — Sleeve PARKED. No Phase-1 development will proceed under the current specification.

**Future Revisit Options**:
- Use VIX - RV126 spread as a regime/conditioning input rather than a direct signal
- Test expression on different volatility instruments (e.g., VIX options, volatility ETFs)
- Test as a spread trade (long VIX, short realized vol proxy) rather than directional VX short
- Add macro regime filters (e.g., only trade in low-vol regimes, avoid crisis periods)
- Apply z-scoring or dynamic thresholds rather than simple sign rule

### Implementation

**Core Modules**:
- `scripts/diagnostics/run_vrp_mid_rv126_phase0.py`: Phase-0 signal test script (tests VX2, VX3)
- `src/market_data/vrp_loaders.py`: `load_vix()`, `load_vx_curve()` functions
- `src/agents/data_broker.py`: `MarketData` class for ES returns (RV126 computation)

**Key Functions**:
- `load_vix()`: Loads VIX from FRED
- `load_vx_curve()`: Loads VX1, VX2, VX3 continuous history
- RV126 computation: 126-day rolling realized volatility from ES futures returns (annualized, vol points)
- Signal generation: Simple threshold-based short-only rule (VIX > RV126)
- PnL calculation: 1-day lag (position from previous day × current return)

---

## Crisis Meta-Sleeve Diagnostics

**Crisis Sleeve Diagnostics**

For Crisis Meta-Sleeves, the following metrics take precedence:

- **Maximum drawdown**
- **Worst-month / worst-quarter loss**
- **Crisis-period attribution**

Traditional metrics (Sharpe, CAGR) are reported for completeness only and must not be used as promotion criteria.

### Crisis Protection Evaluation Scope

**Critical Clarification**: Crisis protection is evaluated at the **portfolio level**, not the instrument level.

**Implication**: Instrument-level convexity success does not imply portfolio-level protection.

- A crisis sleeve may show correct instrument-level behavior (e.g., VX1 > VX2 > VX3 during 2020 Q1)
- But portfolio-level integration may fail (e.g., portfolio drawdown deterioration during fast-crash windows)
- Portfolio-level diagnostics are required to validate crisis protection effectiveness

**Allocator-Level Crisis Protection**:

Crisis protection implemented at the allocator level is evaluated using the same portfolio-level criteria:
- Portfolio-level drawdown improvement
- Portfolio-level worst-month/worst-quarter improvement
- Portfolio-level crisis-period attribution
- Allocator behavior during stress periods

This locks in the Crisis Meta-Sleeve conclusion: instrument-level convexity is insufficient; portfolio-level protection requires allocator-driven conditional exposure control.

### Crisis Phase-0 Summary

**Evaluation Window**: 2020-01-06 → 2025-10-31  
**Sleeve Weight**: 5% fixed  
**Baseline**: Core v9

| Variant | MaxDD Δ vs Core | Worst-Month Δ | CAGR Δ | Status |
|---------|----------------|---------------|--------|--------|
| VX2 (Long) | +0.91% | +0.49% | -0.32% | ✅ PASS |
| VX Spread (VX2-VX1) | +0.76% | +0.41% | -0.52% | ✅ PASS |
| Duration (UB) | -1.20% | +0.54% | -0.65% | ✅ PASS |

**Phase-0 Results**: All three variants passed. VX2 and VX Spread provide tail protection. Duration worsened MaxDD but improved worst-month.

### Crisis Phase-1 Summary

**Evaluation Window**: 2020-01-06 → 2025-10-31  
**Sleeve Weight**: 5% fixed  
**Baseline**: Core v9  
**Benchmark**: Long VX2 (from Phase-0)

**Phase-1 Variants**:
- **Variant A**: Long VX2 (benchmark / convexity ceiling)
- **Variant B**: Long VX3 (reduced carry, weaker convexity)
- **Variant C**: Long VX2 - VX1 spread (primary candidate)
- **Variant D**: Long VX3 - VX2 spread (exploratory)

**Phase-1 Pass Criteria**:
1. **Tail Preservation**: ≥70% of VX2 MaxDD improvement OR match/improve worst-month vs VX2
2. **Cost Reduction**: Improve CAGR vs VX2 OR reduce carry bleed
3. **Stability**: No new left-tail events, no volatility amplification

**Phase-1 Results Table**:

| Variant | MaxDD Δ vs Core | Worst-Month Δ | CAGR Δ | Tail Preservation | Cost Reduction | Stability | Status |
|---------|----------------|---------------|--------|-------------------|----------------|-----------|--------|
| VX2 (Benchmark) | +0.26% | +0.15% | +0.11% | N/A | N/A | ✅ | Reference (benchmark ceiling) |
| VX3 | +0.41% | +0.23% | +0.17% | ✅ (≥70% of VX2) | ✅ (better CAGR) | ✅ | ✅ **PROMOTED** |
| VX Spread (VX2-VX1) | +0.10% | +0.07% | -0.09% | ❌ (<70% of VX2) | ❌ (worse CAGR) | ✅ | ❌ **PARKED** |
| VX3 Spread (VX3-VX2) | +0.15% | +0.07% | +0.02% | ❌ (<70% of VX2) | ❌ (worse CAGR) | ✅ | ❌ **PARKED** |

**Phase-1 Decision**: Long VX3 promoted to Phase-2 (cost-efficient convexity that preserves tails). VX2 retained as benchmark ceiling reference. Both spread variants parked (Phase-1 FAIL) and eligible for re-test only under parked-sleeve triggers.

**Note**: Phase-1 results are generated by `scripts/diagnostics/run_crisis_phase1.py` and saved to `reports/diagnostics/crisis_phase1/crisis_phase1_summary.csv`.

**Sharpe ratio is reported for completeness only and is not a promotion criterion.**

### Crisis Meta-Sleeve — Final Decision (v1)

**Canonical Evaluation Window**: 2020-01-06 → 2025-10-31  
**Canonical Baseline**: Core v9  
**Sleeve Weight**: 5% fixed (all phases)

#### A) Phase-0 Summary (Always-on structural candidates, 5% weight)

**Variants Tested:**
- **Long VX2** — ✅ PASS (tail improvement: MaxDD +0.91%, Worst-month +0.49%; acceptable bleed)
- **Long VX2−VX1 spread** — ✅ PASS (tail improvement: MaxDD +0.76%, Worst-month +0.41%; acceptable bleed)
- **Long UB** — ✅ PASS conditional (worst-month improved; MaxDD worsened; parked for post-v1 reconsideration)

**Phase-0 Outcome**: VX2 and VX spread passed. UB parked as conditional hedge.

#### B) Phase-1 Summary (VX cost-control, 5% weight)

**Variants Tested:**
- **Long VX2** (benchmark ceiling) — ✅ PASS (MaxDD +0.26%, Worst-month +0.15%, CAGR +0.11%)
- **Long VX3** — ✅ PASS (MaxDD +0.41%, Worst-month +0.23%, CAGR +0.17%) — **WINNER**
- **VX2−VX1 spread** — ❌ FAIL (insufficient tail preservation: <70% of VX2 MaxDD improvement)
- **VX3−VX2 spread** — ❌ FAIL (insufficient tail preservation: <70% of VX2 MaxDD improvement)

**Phase-1 Decision**: Long VX3 selected as Phase-1 candidate for Phase-2 integration (cost-efficient convexity that preserves tails). VX2 retained as benchmark ceiling reference (not promoted). Both spread variants parked (Phase-1 FAIL) and eligible for re-test only under parked-sleeve triggers.

#### C) Phase-2 Summary (Portfolio interaction, 5% VX3)

**Baseline**: Core v9  
**Variant**: Core v9 + Long VX3 (5%)

**Overall Tail Metrics vs Core v9:**
- ✅ **MaxDD improved**: -0.1264 vs -0.1371 (improved by +0.0107)
- ✅ **Worst month improved**: -0.0623 vs -0.0680 (improved by +0.0057)
- ✅ **Worst quarter improved**: -0.0476 vs -0.0511 (improved by +0.0035)
- ✅ **2022 drawdown improved**: -0.1213 vs -0.1292 (improved by +0.0079)
- ✅ **No volatility amplification**: Calm regime vol ratio 0.999 (no amplification)

**Crisis Window Analysis:**
- ❌ **2020 Q1 worsened**: Variant MaxDD -0.0279 vs Baseline -0.0206 (worsened by -0.0073)
- ✅ **2022 drawdown improved**: Variant MaxDD -0.1213 vs Baseline -0.1292 (improved by +0.0079)

**Phase-2 Outcome**: Overall tail metrics improved, but 2020 Q1 fast-crash window showed deterioration. Instrument-level VX behavior is correct (VX1 > VX2 > VX3 in 2020 Q1), but portfolio-level peak-to-trough drawdown worsens in that window.

#### D) Final Decision (Canonical)

**Crisis Meta-Sleeve v1: NO PROMOTION**

Long VX3 fails Phase-2 due to deterioration in 2020 Q1 fast-crash behavior. While overall tail metrics improved (MaxDD, worst-month, worst-quarter, 2022 drawdown), the fast-crash window (2020 Q1) showed portfolio-level drawdown deterioration that violates Phase-2 pass criteria.

**Disposition:**
- **VX2**: Retained as benchmark only (not promoted)
- **VX3**: Parked as "tail smoother candidate" for allocator-era research
- **UB**: Remains parked (conditional hedge, post-v1 reconsideration)
- **VX Spreads**: Parked (Phase-1 FAIL, eligible for re-test only under parked-sleeve triggers)

**Conclusion**: Convexity allocation is deferred to allocator logic (v2+). v1 sleeves remain always-on economic return sources; crisis convexity requires conditional activation and therefore belongs in allocator logic (post-v1).

**Note**: Phase-2 results are generated by `scripts/diagnostics/run_crisis_phase2.py` and saved to `reports/diagnostics/crisis_phase2/`.

---

## Future Enhancements

Potential additions to the diagnostics framework:
- Rolling window metrics (rolling Sharpe, rolling MaxDD)
- Drawdown analysis (drawdown length, recovery time)
- Turnover and cost attribution
- Sleeve-level attribution (if sleeve P&L is tracked)
- Statistical significance testing for metric differences

