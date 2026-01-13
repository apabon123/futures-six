# Procedures: How to Add / Change Sleeves, Assets, and Parameters

**Purpose:**  

This document defines **when** and **how** to run the sleeve lifecycle (Phase-0 → Phase-3) for any change to the Futures-Six framework:

- Adding a new **Meta-Sleeve** (e.g., Cross-Sectional Momentum, Vol Risk Premium)
- Adding a new **Atomic Sleeve** inside a Meta-Sleeve
- Changing the **asset universe**
- Changing **parameters** (lookbacks, weights, overlays)
- Deciding **when to look at bad assets** and when *not* to touch them

It complements:

- `SOTs/STRATEGY.md` – architecture & execution flow (what the system does)
- `SOTs/DIAGNOSTICS.md` – diagnostics capabilities & tools (how we analyze runs)
- `SOTs/ROADMAP.md` – development sequencing and sleeve status

This file is about **process** and **checklists**.

## Related Documents

- **STRATEGY.md**: Current implementation details and architecture
- **DIAGNOSTICS.md**: Validation results and metrics (single source of truth for Phase-0/1/2 results)
- **ROADMAP.md**: Sleeve status and development priorities (single source of truth for status)

---

## 1. Core Principles

1. **Architecture first, optimization later**

   - Goal of current phase: **get all Meta-Sleeves and core overlays implemented and wired in**.
   - We **do not** aggressively tune asset lists, ranks, or parameters until the architecture is complete.

2. **Sleeve lifecycle is about the *idea*, not every tiny config tweak**

   From `SOTs/STRATEGY.md` (sleeve lifecycle):

   - **Phase-0 – Simple Sanity Check**
     - Sign-only, no overlays, equal-weight / DV01-neutral.
     - Objective: "Does the economic idea have any edge at all?"
     - **Exception: Crisis Sleeves** – Phase-0 evaluates tail-risk mitigation (MaxDD, worst-month), not Sharpe. Bleed is expected and acceptable.
   - **Phase-1 – Clean Implementation**
     - Proper feature engineering, z-scoring, cross-sectional ranking, etc.
     - **Exception: Crisis Sleeves** – Phase-1 focuses on reducing carry bleed without destroying convexity.
   - **Phase-2 – Overlay Integration**
     - Macro regime filters, vol targeting, allocator integration.
     - **Exception: Crisis Sleeves** – Phase-2 validates portfolio interaction with Core.
   - **Phase-3 – Production + Monitoring**
     - Full runs, performance monitoring, alerts, regression tests.

**Crisis Sleeves — Lifecycle Exception:**

Crisis sleeves follow a modified lifecycle:

| Phase | Objective |
|-------|-----------|
| Phase-0 | Validate tail-risk mitigation (MaxDD, worst-month) |
| Phase-1 | Reduce carry bleed without destroying convexity |
| Phase-2 | Validate portfolio interaction with Core |

**Critical Rule**: At no stage in v1 are Crisis sleeves gated, timed, or conditionally activated. They maintain always-on exposure throughout all phases.

Sharpe ratio is explicitly not a Phase-0 gating metric for Crisis sleeves.

**Phase-1 Decision (2025-12-17)**: Phase-1 winner: Long VX3 (cost-efficient convexity that preserves tails); VX2 retained as benchmark ceiling reference.

**Final Decision (2025-12-17)**: Crisis Meta-Sleeve v1 — NO PROMOTION. Long VX3 failed Phase-2 due to 2020 Q1 fast-crash deterioration. Allocator logic is the only approved mechanism for conditional crisis protection.

3. **We don't prune assets during build-out**

   - "Bad" assets (e.g., SR3 front, CL, 6J in CSMOM) are **kept during the build phase** so:
     - We preserve information about where sleeves fail.
     - We avoid chasing performance before the whole machine is assembled.
   - Asset pruning and multipliers are **Phase-B: Optimization & Pruning**, *after* core architecture is in place.

---

## 2. Run Consistency Contract ⚠️

**CRITICAL:** All runs, diagnostics, and comparisons must follow these rules to ensure valid, apples-to-apples performance comparisons.

### 2.1 Canonical Start Date Rule

**Rule**: All runs must begin on the **earliest timestamp T** where **every enabled atomic sleeve** produces a valid, non-NA signal for **all assets in the universe**.

This applies to:
- Phase-0 sanity checks
- Phase-1 standalone sleeve tests
- Phase-2 integrated A/B tests
- All production profiles
- All diagnostics comparisons
- All research runs

**Why This Matters:**

Different sleeves have different warmup requirements:
- Long-Term (252d): Requires ~252 trading days of history
- Medium-Term (84d): Requires ~84 trading days of history
- Short-Term (21d): Requires ~21 trading days of history
- Breakout Mid (50-100d): Requires ~100 trading days of history

If you request `start_date = 2018-01-01`, the **effective start date** will be:
```
2018-01-01 + max(all_sleeve_warmups) ≈ 2018-01-01 + 252d ≈ 2018-12-15
```

**What This Guarantees:**
- Same starting index for every run with the same profile
- Same number of rows for all comparisons
- No silent NA propagation
- No mismatched time series
- Valid diagnostics comparisons

**What This Prevents:**
- Phase-1 "passing" because it only ran on 400 days
- Diagnostics comparing 500-row runs vs 1700-row runs
- Bogus Sharpe/Vol/MaxDD from truncated samples
- Silent failures due to missing features

### 2.2 Canonical NA Handling Rule

**Rule**: Drop rows with **ANY** NA values at each stage of the pipeline:

1. **After feature computation** (FeatureService):
   ```python
   features_df = features_df.dropna(how="any")
   ```

2. **After sleeve signal computation** (each atomic sleeve):
   ```python
   signals = signals.dropna(how="any")
   ```

3. **Before meta-sleeve aggregation** (TSMOMMultiHorizon, etc.):
   ```python
   sleeve_signals_df = pd.concat(all_sleeves, axis=1).dropna(how="any")
   ```

4. **Before allocator/position sizing** (run_strategy):
   ```python
   aligned_signals = combined_signals.dropna(how="any")
   ```

**Why This Matters:**

Inconsistent NA handling causes:
- Different runs to have different effective sample sizes
- Silent feature misalignment
- Bogus performance comparisons
- Impossible-to-debug discrepancies

### 2.3 Canonical Logging Rule

**Rule**: Every run MUST log:

1. **Requested start date** (from command line or config)
2. **Effective start date** (after warmup and NA dropping)
3. **Total trading days** (requested date range)
4. **Valid rows** (after all alignments and NA drops)
5. **Rows dropped** (due to NAs, warmup, etc.)

**Example Output:**
```
Requested start: 2018-01-01
Effective start: 2018-12-15
Requested end: 2025-10-31
Total calendar days: 2860
Total trading days: 1950
Valid rows after warmup: 1754
Rows dropped due to NA: 0
Final rows used for metrics: 1754
```

**Red Flags** (if you see these, something is wrong):
- Valid rows: 450 (too short, missing warmup?)
- Valid rows: 723 (misaligned features?)
- Effective start: 2020-03-20 (when requested 2018-01-01?)
- Rows dropped: 800 (major feature alignment issue?)

### 2.4 Implementation Checklist

**For All New Runs:**

- [ ] Verify `start_date` is set consistently across all profiles being compared
- [ ] Check that effective start date is logged in run output
- [ ] Confirm row counts match between variant and baseline
- [ ] Verify no silent NA propagation (rows_dropped should be 0 or minimal)
- [ ] Check that all enabled sleeves have valid features from effective start date

**For Diagnostics Comparisons:**

- [ ] Verify both runs use the same effective date range
- [ ] Check that row counts match exactly
- [ ] Confirm overlapping days are identical (no misalignment)
- [ ] Validate that metrics are computed on the same sample

**Red Flags to Investigate:**

- Runs with different start dates being compared
- Row count mismatch between variant and baseline
- Large numbers of dropped rows (>5% of sample)
- Effective start date differs from expected warmup

### 2.5 Known Issues and Migration Plan

**Current State** (as of Nov 2025):

Some existing runs have inconsistent start dates:
- `core_v3_no_macro_baseline` (old): Started 2021-01-01  
- `core_v3_shortcanon_v1`: Started 2018-01-01
- Phase-0/1/2 runs: Mix of start dates

**Migration Plan:**

1. ✅ Document the Run Consistency Contract (this section)
2. 🚧 Update `run_strategy.py` to enforce canonical start date and log effective dates
3. 🚧 Update all Phase-0/1/2 scripts to use consistent start dates
4. 🚧 Re-run all baselines with consistent dates for clean comparisons
5. 🚧 Add validation checks to `run_perf_diagnostics.py` to warn on mismatched samples

**Action Required:**

When you see diagnostic comparisons with mismatched row counts or start dates, **do not trust the results**. Re-run both variant and baseline with:
- Same requested `start_date`
- Same requested `end_date`
- Same profile structure (same enabled sleeves)
- Verify effective start dates match

---

## 3. Data Expansion Protocol

### Canonical Research Window vs Structural Validation Window

**Hard Rule**: Initial research and sleeve validation may be performed on a subset window (e.g., 2020–2025), but before production freeze, the system must be validated on an expanded historical window (e.g., 2010–present).

**Purpose of Expanded Data:**

Expanded historical data is used for:
- **Invariance checks**: Verify that sleeve behavior is consistent across different market regimes
- **Failure mode discovery**: Identify edge cases and stress scenarios not visible in shorter windows
- **Allocator behavior validation**: Ensure allocator logic performs correctly across diverse market conditions

**Critical Rule: No Re-Optimization**

**Expanding historical data does not permit re-optimization of parameters unless a structural failure is discovered.**

If conclusions materially change with expanded data:
- **Architecture is revised** (not parameters)
- Structural changes to sleeve logic or allocator design are acceptable
- Parameter tuning based on expanded data is prohibited until production freeze

**Rationale:**

This rule prevents overfitting to expanded historical windows. The goal is to validate that the architecture is robust across regimes, not to find parameters that work on a longer window.

**Implementation Checklist:**

- [ ] Initial research performed on canonical window (e.g., 2020–2025)
- [ ] Phase-0/Phase-1 validation completed on canonical window
- [ ] Before production freeze, re-run validation on expanded window (e.g., 2010–present)
- [ ] Document any structural failures discovered
- [ ] If failures found, revise architecture (not parameters)
- [ ] Re-validate revised architecture on both canonical and expanded windows

---

## 4. Change Types (Taxonomy)

Every change falls into one of these buckets:

1. **New Meta-Sleeve**
   - Example: adding **Cross-Sectional Momentum** or **Volatility Risk Premium** as a new economic idea.

2. **New Atomic Sleeve inside an existing Meta-Sleeve**
   - Example: adding a new short-term trend variant inside the Trend Meta-Sleeve.

3. **Universe Change (Assets)**
   - Add, remove, or swap assets (e.g., change SR3 front → SR3 rank 3).
   - Change roll rule or contract mapping that effectively changes the traded instrument.

4. **Parameter Change (Within a Sleeve)**
   - Change lookbacks, feature weights, clipping thresholds, vol lookbacks, etc.

5. **Overlay / Allocator Change**
   - Change macro overlay logic, vol targeting logic, risk limits, allocator method.

6. **Non-functional Change**
   - Refactors, code cleanup, logging changes, doc edits only.

---

## 4. When Do We Run Which Phases?

Think of this as the **decision table**:

| Change Type                               | Phase-0?                      | Phase-1?                           | Phase-2?                                     | Notes |
|-------------------------------------------|-------------------------------|------------------------------------|----------------------------------------------|-------|
| New **Meta-Sleeve**                       | **YES (mandatory)**           | **YES (mandatory)**               | **YES** once Phase-1 passes                  | Treat as new economic idea. |
| New **Atomic Sleeve** in existing Meta    | **YES** (cheap sign-only)     | **YES** (clean implementation)    | Optional (depends if atomic is used in live) | Usually quick. |
| **Universe change** (add/remove/swap asset)| **NO** fresh Phase-0/1        | **Re-run existing diagnostics**    | Optional (if affects overlays materially)    | Regression testing only. |
| **Parameter change** (lookbacks, weights) | **NO** new Phase-0/1          | **Re-run existing diagnostics**    | Optional                                     | Treat as variant of same idea. |
| **Overlay / allocator change**           | No (sleeve idea unchanged)    | No (sleeve idea unchanged)        | **YES**: run overlay/portfolio diagnostics   | Uses existing sleeve outputs. |
| **Non-functional change**                 | No                            | No                                | No                                          | Quick sanity check run optional. |

**Key rule:**  

**Phase-0/Phase-1 are for *ideas*.**  

Adding/changing assets or tuning parameters uses **regression diagnostics**, not a whole new sleeve lifecycle.

---

## 5. Procedure: Adding a New Meta-Sleeve

### 4.1 Design Spec

Before touching code, create a short design section in `SOTs/STRATEGY.md`:

- Economic thesis (why this sleeve should have edge).
- Universe it will trade.
- Key features (e.g., returns, curve, carry, macro inputs).
- Expected behavior (when it should make / lose money).

### 4.1.1 VRP Meta-Sleeve: Data Prerequisites

**Before running VRP Phase-0**, ensure the following data pipeline steps have been completed:

**Step 1: CBOE Data Ingestion (financial-data-system)**

1. Run CBOE VX futures scraper:
   ```bash
   # In financial-data-system repo
   python scripts/ingest_cboe_vx.py
   ```

2. Run CBOE VIX3M index scraper:
   ```bash
   # In financial-data-system repo
   python scripts/ingest_cboe_vix3m.py
   ```

3. Run CBOE VVIX index scraper:
   ```bash
   # In financial-data-system repo
   python scripts/ingest_cboe_vvix.py
   ```

**Step 2: Sync to Canonical DB (databento-es-options)**

3. Sync VIX from FRED (if not already done):
   ```bash
   # In databento-es-options repo
   python scripts/sync_fred_vix.py
   ```

4. Sync VX, VIX3M, and VVIX from financial-data-system:
   ```bash
   # In databento-es-options repo
   python scripts/sync_vix_vx_from_financial_data_system.py
   ```

**Step 3: Verify VRP Data in Futures-Six**

5. Run VRP Phase-0 diagnostics:
   ```bash
   # In futures-six repo
   python scripts/diagnostics/run_vrp_phase0.py
   ```

**Expected Outputs:**
- `data/diagnostics/vrp_phase0/vrp_inputs.parquet`: Full VRP dataset
- `data/diagnostics/vrp_phase0/summary_stats.json`: Coverage and summary stats
- `data/diagnostics/vrp_phase0/*.png`: Diagnostic plots
- `reports/phase_index/vrp/phase0.txt`: Phase-0 registration

**VRP Coverage Targets:**
- VIX: ≥95% coverage (starts 1990)
- VIX3M: ≥95% coverage (starts 2009-09-18)
- VVIX: ≥95% coverage (starts ~2010, required for VRP-Convexity)
- VX1/2/3: ≥90% coverage (starts ~2004)

**Pass Criteria:**
- All required data sources (VIX, VIX3M, VVIX, VX1/2/3) are present
- Coverage meets minimum thresholds
- VRP spreads (VIX-VX1, VIX3M-VIX, VX2-VX1) are computable
- VVIX available for VRP-Convexity sleeve (CBOE via financial-data-system → canonical sync)
- No extended periods of missing data

If any criteria fail, revisit Steps 1-4 above before proceeding to VRP strategy implementation.

### 4.1.2 VRP-Core Canonical Sleeve: Phase-0 / Phase-1 / Phase-2

**VRP-Core** is the canonical VRP atomic sleeve using z-scored VRP spread (VIX - realized ES vol).

**Prerequisites**: Complete Steps 1-5 in § 4.1.1 (VRP data pipeline must pass data diagnostics)

**Units Fix (Critical)**:
- **VIX**: In vol points (e.g., 20 = 20%)
- **Realized ES vol**: Computed as `std(returns) * sqrt(252)` → gives decimals (e.g., 0.18 = 18%)
- **Fix**: Multiply realized vol by 100 to convert to vol points before computing spread:
  ```python
  vrp_spread = VIX - (RV_21 * 100.0)  # Both in vol points
  ```
- **Without fix**: Spread ≈ 21 vol points (nonsense: 20 - 0.18 = 19.82)
- **With fix**: Spread ≈ 2-6 vol points (realistic: 20 - 18 = 2)
- All documented results refer to post-fix implementation.

**Phase-0 (Signal Test)**:

VRP-Core Phase-0 tests the raw economic idea using a toy rule:

- **Economic spec**: VIX (30d implied vol) vs 21-day realized ES volatility
- **Toy rule**: Short VX1 when (VIX - RV_21 × 100) > 1.5 vol points, otherwise flat
  - Threshold of 1.5 is fixed for Phase-0 documentation and clarity only
  - This threshold is NOT used in Phase-1 or Phase-2
- **No z-scores, no clipping, no vol targeting**
- **Pass criteria**: Sharpe ≥ 0.1 (marginal edge), ideally ≥ 0.2

**Phase-0 Guidance: Asymmetric Economic Drivers**

When an economic driver is asymmetric, symmetric Phase-0 may fail. In such cases, a directional Phase-0 rule is acceptable if:

1. **Justified by economic structure**: The asymmetry is inherent to the economic relationship (e.g., VIX term structure contango)
2. **Supported by literature**: Academic or industry research documents the asymmetry
3. **Methodological integrity preserved**: The Phase-0 test still validates the core economic idea

**Documented Exception: VRP-Convergence**

VRP-Convergence Phase-0 uses a **short-only rule** because:
- **Positive spreads (VIX > VX1) do NOT imply mean reversion**
  - They often indicate momentum/expansion regimes
  - These are not mean-reverting and should not be traded
- **Only negative spreads (VX1 > VIX) produce stable convergence**
  - VX1 typically trades above VIX (contango)
  - Decay from VX1 → VIX produces reliable downward convergence
  - This aligns with academic VIX term structure literature

**Phase-0 Process (Extended)**:
- Started with symmetric rule (long/short), which failed (Sharpe -0.15, MaxDD -86%)
- Moved to short-only rule, tested thresholds (1.0, 1.5, 2.0 vol points)
- Selected 1.0 as canonical because it passed with Sharpe ≈ 0.43
- **Phase-1 replaces the Phase-0 threshold with a short-only continuous z-scored signal**

This preserves methodological integrity while respecting the asymmetric economics of the VIX term structure.

```bash
# Run Phase-0 signal test (includes data diagnostics)
python scripts/diagnostics/run_vrp_phase0.py --start 2020-01-01 --end 2025-10-31
```

**Outputs**:
- Data diagnostics: `data/diagnostics/vrp_phase0/data_diagnostics/`
- Phase-0 signal test: `data/diagnostics/vrp_phase0/phase0_signal_test/`
- Phase index: `reports/phase_index/vrp/phase0.txt`

**Results**: See `SOTs/DIAGNOSTICS.md` § "VRP-Core Phase-0 Signal Test" for full metrics and analysis.

**Note**: Any prior runs with spread ~21 vol points were invalid due to units mismatch.

**Phase-1 (Engineered Sleeve)**:

1. **Run VRP-Core Phase-1 diagnostics**:
   ```bash
   python scripts/diagnostics/run_vrp_core_phase1.py --start 2020-01-01 --end 2025-10-31
   ```

2. **Review metrics**:
   - Sharpe ratio (target: ≥0.2 for Phase-1 pass)
   - MaxDD, hit rate, signal distribution
   - Compare vs Phase-0 (should show improvement)

3. **Outputs**:
   - Results: `data/diagnostics/vrp_core_phase1/<timestamp>/`
   - Phase index: `reports/phase_index/vrp/vrp_core_phase1.txt`
   - Plots: equity curve, distributions, signals over time

**Key Differences from Other Sleeves**:
- VRP-Core trades VX1 futures, not the standard futures universe
- Directional strategy (not market-neutral like CSMOM)
- Requires VRP data pipeline (VIX, VX from canonical DB)
- Units fix: realized vol must be multiplied by 100 before computing spread

**Pass Criteria (Phase-1)**:
- Sharpe ≥ 0.2 over full test window
- MaxDD within reasonable bounds (<50%)
- Signal distribution shows meaningful variation (not stuck at extremes)
- Improvement over Phase-0 (z-scoring and vol targeting should help)

**Results**: See `SOTs/DIAGNOSTICS.md` § "VRP-Core Phase-1 Diagnostics" for full metrics and analysis.

**Phase-2 (Portfolio Integration)**:

1. **Run Phase-2 diagnostics**:
   ```bash
   python scripts/diagnostics/run_core_v5_trend_csmom_vrp_core_phase2.py --start 2020-01-01 --end 2025-10-31
   ```

2. **This script**:
   - Runs baseline: Core v4 (Trend 75% + CSMOM 25%) - now superseded
   - Runs VRP-enhanced: Core v5 (Trend 65% + CSMOM 25% + VRP-Core 10%) - now historical reference baseline
   - Compares portfolio metrics (Sharpe, CAGR, Vol, MaxDD, HitRate)
   - Analyzes crisis-period performance (2020 Q1/Q2, 2022)
   - Computes correlation between portfolios

3. **Outputs**:
   - Comparison returns: `data/diagnostics/phase2/core_v5_trend_csmom_vrp_core/<timestamp>/`
   - Comparison summary: `comparison_summary.json`
   - Plots: equity curves, drawdown curves
   - Phase index: `reports/phase_index/vrp/phase2_core_v5_trend_csmom_vrp_core.txt`

4. **Promotion Decision**:
   - ✅ **Promoted**: Core v5 promoted to baseline (Dec 2025) after Phase-2 pass, then superseded by Core v6 (Dec 2025)
   - Criteria:
     - Portfolio Sharpe improves or remains similar with lower drawdown
     - Crisis behavior is acceptable
     - VRP contribution is additive and not redundant

### 4.2 Phase-0 Checklist (Sign-Only)

1. Implement a **minimal sign-only** version:

   - Simple rule (e.g., sign of k-day return, sign of slope, sign of roll yield).
   - Equal-weight or DV01-neutral.
   - Daily rebalance.
   - No overlays, no vol targeting.

2. Add a **Phase-0 script** under `scripts/`, e.g.:

   - `scripts/run_<sleeve>_sanity.py`

3. Output to `reports/sanity_checks/<meta_sleeve>/<atomic>/archive/<timestamp>/` (automatically updates `latest/` and `phase_index`).

4. Run over a **long window** (≥ 5y).

5. Evaluate with `DIAGNOSTICS` tools:

   - Sharpe ≥ 0.2?
   - MaxDD, hit rate, per-asset stats.
   - Subperiod behavior (pre/post 2022 if relevant).

6. Log result in `SOTs/DIAGNOSTICS.md` under **Phase-0 Sanity Checks** and in `SOTs/STRATEGY.md` under that Meta-Sleeve's status.   

**If Phase-0 fails → Meta-Sleeve stays disabled.**

### 4.3 Phase-1 Checklist (Clean Implementation)

Once Phase-0 passes:

1. Implement production-quality sleeve:

   - Feature engineering (multi-horizon, z-scores, etc.).
   - Vol normalization inside sleeve if appropriate.
   - Cross-sectional ranking logic where applicable.

2. Integrate into `FeatureService` and strategy agents (see `SOTs/STRATEGY.md` sections for existing sleeves as templates).

3. Add a **Phase-1 diagnostics script**, e.g.:

   - `src/diagnostics/<sleeve>_phase1.py`
   - `scripts/run_<sleeve>_phase1.py`

4. Run Phase-1 diagnostics:

   - Compare to Phase-0 (Sharpe, MaxDD, Vol).
   - Confirm behavior matches the idea (e.g., market-neutral, cross-sectional structure).

5. Update phase index: `python scripts/update_phase_index.py <meta_sleeve> <sleeve_name> phase1 <run_id>`

6. Update `SOTs/STRATEGY.md` status:

   - "Meta-Sleeve X: Phase-1 PASSED; ready for overlay integration."

### 4.4 Phase-2 Checklist (Overlays & Portfolio Integration)

1. Wire the Meta-Sleeve into:

   - Macro overlay (if applicable).
   - Vol targeting overlay.
   - CombinedStrategy / allocator.

2. Add / update a **strategy profile** in the config (e.g., `core_v4_with_<sleeve>`).

3. Run `run_strategy.py` with a new `run_id`, store under `reports/runs/<run_id>/`.

4. Analyze with `run_perf_diagnostics.py` (core metrics, yearly stats, per-asset, baseline comparison).

5. Update phase index: `python scripts/update_phase_index.py <meta_sleeve> <sleeve_name> phase2 <run_id>`

6. If results acceptable, mark sleeve as **Phase-2 integrated**.

> **Baseline Profiles (as of Dec 2025)**
> - `core_v3_no_macro`: Trend-only baseline. Used when testing new Trend atomic sleeves or when isolating Trend behavior.
> - **Core v7**: ✅ **Current canonical baseline**. Trend + CSMOM + VRP-Core + VRP-Convergence + VRP-Alt. Used when testing new Meta-Sleeves or overlays.
> - **Core v6**: Historical reference baseline (Trend + CSMOM + VRP-Core + VRP-Convergence). Retained for historical comparison.
> - **Core v5**: Historical reference baseline (Trend + CSMOM + VRP-Core). Retained for historical comparison.
> - `core_v4_trend_csmom_no_macro`: Superseded baseline (retained for historical comparison). Pre-VRP baseline with Trend 75% + CSMOM 25%.
>
> When adding a new Meta-Sleeve, default to using **Core v7** as the baseline profile unless the experiment specifically concerns Trend-only behavior.

### VRP-Core Baseline Promotion (core_v5)

**Sleeve**: VRP-Core (canonical VRP sleeve)

**Lifecycle**: Passed Phase-0 (see `SOTs/DIAGNOSTICS.md` § "VRP-Core Phase-0 Signal Test"), Phase-1 (see `SOTs/DIAGNOSTICS.md` § "VRP-Core Phase-1 Diagnostics"), and Phase-2 (integration vs Trend+CSMOM baseline; see `SOTs/DIAGNOSTICS.md` § "VRP-Core Phase-2 Diagnostics").

**Baseline Change**:
- **Previous baseline**: `core_v4_trend_csmom_no_macro` (Trend 75%, CSMOM 25%)
- **New baseline**: Core v5 (Trend 65%, CSMOM 25%, VRP-Core 10%)

**Phase-2 Decision Logic** (2020-2025 window):
- Sharpe improved slightly (+0.0074)
- CAGR improved slightly (+0.11%)
- Max drawdown did not worsen (slight improvement)
- Crisis-period performance (2020 Q1, 2020 Q2, 2022) was neutral or better
- **Full Phase-2 analysis**: See `SOTs/DIAGNOSTICS.md` § "VRP-Core Phase-2 Diagnostics"

**Outcome**: VRP-Core is promoted into the baseline; Core v4 is retained for historical comparison but tagged as superseded.

#### VRP-Convergence Baseline Promotion (Core v6)

**Lifecycle**: Passed Phase-0 (see `SOTs/DIAGNOSTICS.md` § "VRP-Convergence Phase-0 Diagnostics"), Phase-1 (see `SOTs/DIAGNOSTICS.md` § "VRP-Convergence Phase-1 Diagnostics"), and Phase-2 (portfolio integration vs Core v5; see `SOTs/DIAGNOSTICS.md` § "VRP-Convergence Phase-2 Diagnostics").

#### VRP-Alt Baseline Promotion (Core v7) — Non-Standard Lifecycle

**Lifecycle**: Completed a modified 3-stage lifecycle with mandatory scaling verification:

1. **Phase-0**: Borderline pass (Sharpe ≈ 0.10, catastrophic MaxDD ≈ –94% expected for raw short-vol signals)
2. **Phase-1**: Strong pass after engineering (Sharpe ≈ 0.91, MaxDD ≈ –2%, very low volatility ~2%)
3. **Phase-2**: Inconclusive at 5% allocation (Portfolio Sharpe +0.0024, MaxDD +0.05%, limited impact due to low sleeve volatility)
4. **Scaling Analysis (Mandatory)**: Evaluated at 5%, 7.5%, 10%, 15% weights
   - Monotonic Sharpe improvement
   - Stable MaxDD (+0.15% at 15%)
   - No crisis vulnerabilities
   - Stable marginal Sharpe (~+0.0005 per 1% increase)
5. **Promotion**: VRP-Alt promoted at 15% weight, forming Core v7

**Baseline Change**:
- **Previous baseline**: Core v6 (Trend 62.5%, CSMOM 25%, VRP-Core 7.5%, VRP-Convergence 5%)
- **New baseline**: Core v7 (Trend 60%, CSMOM 25%, VRP-Core 7.5%, VRP-Convergence 2.5%, VRP-Alt 15%)

**Outcome**: VRP-Alt is promoted into the baseline at 15% weight; Core v6 is retained for historical comparison but tagged as superseded.

### 4.4.1 Promotion Exception: Borderline Phase-0 + Scaling Verification Lifecycle

**Codified Lifecycle Path**: Borderline Phase-0 → Phase-1 strong → Phase-2 inconclusive → scaling verification → promotion

Certain VRP sleeves may exhibit catastrophic raw Phase-0 drawdowns but demonstrate clear economic edge (e.g., implied minus short-term realized vol). In such cases, the following lifecycle path is codified:

1. **Phase-0 borderline pass is acceptable** if Phase-1 engineering produces a stable sleeve.
2. **If Phase-2 results are inconclusive** due to low weight or low volatility, the sleeve must undergo mandatory scaling analysis (e.g., 5%, 7.5%, 10%, 15%).
3. **If scaling analysis shows monotonic improvement and controlled MaxDD**, the sleeve may be promoted.

**Canonical Example**: VRP-Alt (Dec 2025)
- Phase-0: Borderline pass (Sharpe ≈ 0.10, catastrophic MaxDD expected)
- Phase-1: Strong pass (Sharpe ≈ 0.91, MaxDD ≈ –2%)
- Phase-2: Inconclusive at 5% (limited impact due to low volatility)
- Scaling verification: Tested 5%, 7.5%, 10%, 15% → monotonic improvement, controlled MaxDD
- Promotion: VRP-Alt promoted at 15% weight, forming Core v7

This lifecycle path is now a documented procedure for future borderline VRP sleeves, not a one-off exception.

**Baseline Change**:
- **Previous baseline**: Core v5 (Trend 65%, CSMOM 25%, VRP-Core 10%)
- **New baseline**: Core v6 (Trend 62.5%, CSMOM 25%, VRP-Core 7.5%, VRP-Convergence 5%)

**Phase-2 Metrics (2020-2025, canonical window)**:
- Sharpe: 0.5774 → 0.5796 (+0.0022)
- CAGR: 6.74% → 6.77% (+0.03%)
- MaxDD: -17.22% → -17.18% (+0.04%, less negative)
- **Full Phase-2 analysis**: See `SOTs/DIAGNOSTICS.md` § "VRP-Convergence Phase-2 Diagnostics"

**Decision**: Core v6 was the canonical baseline for Phase-2 comparisons. Core v5 is retained as a historical reference baseline.

**Outcome**: VRP-Convergence is promoted as the 2nd canonical VRP atomic sleeve alongside VRP-Core. Core v6 was the canonical baseline until superseded by Core v7 (Dec 2025).

### 4.5 Phase-3 Checklist (Production + Monitoring)

1. Promote sleeve + strategy profile to "production" set.

2. For each production run, ensure:

   - `run_id` is logged and frozen.
   - Diagnostics run vs baseline.
   - Results summarized in `SOTs/DIAGNOSTICS.md` under **Production Monitoring**.

3. Only in **Phase-3 / Optimization** do we start:

   - Asset-level pruning.
   - Aggressive parameter tuning.
   - Per-sleeve asset multipliers.

---

## 6. Procedure: Adding an Atomic Sleeve to an Existing Meta-Sleeve

**Example: Adding Trend Breakout 50-100d inside the Trend Meta-Sleeve (Phase-0 lite → Phase-1 → Trend-level diagnostics).**

1. **Spec**:  

   - Add a short section in `SOTs/STRATEGY.md` under that Meta-Sleeve:

     - What horizon? (50-100d medium-term)
     - What features? (50-day and 100-day breakout scores)
     - What role in the Meta-Sleeve? (Complementary to return/slope-based signals, captures volatility expansions)
   
   - Add detailed design section in `docs/META_SLEEVES/TREND_RESEARCH.md`:
     - Economic idea (Donchian-style breakouts, regime shifts)
     - Data requirements (continuous prices only)
     - Expected behavior and hypotheses

2. **Phase-0 lite**:

   - If it's a simple TSMOM or CSMOM variant, you *can* reuse existing Phase-0 scripts.
   - For new signal types (e.g., breakout), add a sign-only variant check:
     - Create `scripts/run_trend_breakout_mid_sanity.py`
     - Implement simple sign-only logic (no z-scoring, equal weight, no overlays)
     - Run over long window (≥5y), check Sharpe ≥ 0.2
     - Store results in `reports/sanity_checks/trend/breakout_mid_50_100/archive/<timestamp>/` (canonical results in `latest/`)

3. **Phase-1**:

   - Implement atomic sleeve features and signal logic:
     - Add breakout features to `src/agents/feature_long_momentum.py`
     - Compute 50-day and 100-day breakout scores
     - Apply time-series z-scoring (252-day rolling, clipped at ±3.0)
   - Ensure it plugs into the Meta-Sleeve combiner cleanly (e.g., as one more horizon in Trend):
     - Extend `TSMOMMultiHorizon` to load breakout features
     - Add `breakout_mid_50_100` atomic sleeve signal
     - Combine with existing atomic sleeves using configurable horizon weights

4. **Diagnostics**:

   - Run Meta-Sleeve-level diagnostics to see impact of the new atomic sleeve:

     - Compare Meta-Sleeve with and without it using `run_perf_diagnostics.py`.

**Example: Adding VRP-Convergence inside the VRP Meta-Sleeve (Phase-0 → Phase-1 → Phase-2).**

1. **Spec**:  
   - Add a section in `SOTs/STRATEGY.md` under VRP Meta-Sleeve:
     - Economic idea: VIX (spot) vs VX1 (front-month futures) convergence
     - What features? (spread_conv = VIX - VX1, optional curve_slope_vx = VX2 - VX1)
     - What role in the Meta-Sleeve? (Second atomic VRP sleeve alongside VRP-Core)
   
   - Document in `docs/SOTs/STRATEGY.md`:
     - Economic idea (convergence/dislocation signal)
     - Data requirements (VIX from FRED, VX1 from canonical DB)
     - Expected behavior and hypotheses

2. **Phase-0**:
   - Create `scripts/diagnostics/run_vrp_convergence_phase0.py`
   - Implement simple tri-state convergence rule:
     - Long VX1 when (VIX - VX1) > +T
     - Short VX1 when (VIX - VX1) < -T
     - Flat otherwise
   - Use T = 1.0 vol points (fixed for Phase-0 documentation only)
   - Run over canonical window (2020-01-01 to 2025-10-31)
   - Check Sharpe ≥ 0.1, non-degenerate signal distribution
   - Store results in `data/diagnostics/vrp_convergence_phase0/phase0_signal_test/`
   - Register in `reports/phase_index/vrp/vrp_convergence_phase0.txt`

3. **Phase-1**:
   - Implement atomic sleeve features and signal logic:
     - Create `src/agents/feature_vrp_convergence.py` with VRPConvergenceFeatures class
     - Compute convergence spread: `spread_conv = VIX - VX1`
     - Apply z-scoring (252-day rolling window, clipped ±3σ): `conv_z = (spread_conv - rolling_mean_252) / rolling_std_252`
     - Create `src/agents/strat_vrp_convergence.py` with VRPConvergencePhase1 and VRPConvergenceMeta classes
     - Generate signals: `conv_z_neg = min(conv_z, 0.0)` (short-only), then `signal = np.tanh(conv_z_neg / 2.0)` (bounded in [-1, 0])
     - **Phase-0 discovered that only the negative side of VIX – VX1 is economically valid**
     - **Phase-1 therefore encodes this as a short-only engineered sleeve** (replaces Phase-0 threshold with continuous z-score signal)
     - Apply volatility targeting (target vol = 10%, vol lookback = 63 days, vol floor = 5%)
     - Use 1-day lag between signal and position to avoid lookahead
     - Apply volatility targeting (target vol = 10%, vol lookback = 63 days)
   - Run Phase-1 diagnostics:
     ```bash
     python scripts/diagnostics/run_vrp_convergence_phase1.py --start 2020-01-01 --end 2025-10-31
     ```
   - Check pass criteria: Sharpe ≥ 0.20, reasonable MaxDD, balanced signal distribution
   - Store results in `data/diagnostics/vrp_convergence_phase1/<timestamp>/`
   - Register in `reports/phase_index/vrp/vrp_convergence_phase1.txt`

4. **Phase-2 (Portfolio Integration)**:
   - Add new strategy profile Core v6 to `configs/strategies.yaml`:
     - Baseline: Core v5 (Trend 65% + CSMOM 25% + VRP-Core 10%)
     - Variant: Trend 62.5% + CSMOM 25% + VRP-Core 7.5% + VRP-Convergence 5%
   - Update `run_strategy.py` to support `vrp_convergence_meta` strategy
   - Run Phase-2 diagnostics:
     ```bash
     python scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py --start 2020-01-01 --end 2025-10-31
     ```
   - Compare baseline vs variant:
     - Portfolio metrics (Sharpe, CAGR, Vol, MaxDD, HitRate)
     - Crisis-period performance (2020 Q1/Q2, 2022)
     - Sleeve-level correlations (VRP-Convergence vs Trend, CSMOM, VRP-Core)
   - Check pass criteria:
     - Portfolio Sharpe improves or stays similar with no worse MaxDD
     - Crisis behavior is neutral or improved
     - Sleeve-level correlations show VRP-Convergence is not redundant (e.g., corr(VRP-Convergence, VRP-Core) < 0.95)
   - Store results in `data/diagnostics/phase2/core_v6_trend_csmom_vrp_core_convergence/<timestamp>/`
   - Register in `reports/phase_index/vrp/phase2_core_v6_trend_csmom_vrp_core_convergence.txt`

5. **Documentation Updates**:
   - Update `docs/SOTs/STRATEGY.md` with VRP-Convergence section
   - Update `docs/SOTs/PROCEDURES.md` with VRP-Convergence example (this section)
   - Update `docs/SOTs/DIAGNOSTICS.md` with VRP-Convergence Phase-0/1/2 sections

**Key Differences from Other Atomic Sleeves**:
- VRP-Convergence trades VX1 futures, not the standard futures universe
- Directional strategy (not market-neutral like CSMOM)
- Uses same VRP data pipeline as VRP-Core (VIX, VX1 from canonical DB)
- Both VIX and VX1 are already in vol points (no units conversion needed)

**Case Study: VRP-TermStructure — Phase-0 Economic Failure**

A Phase-0 version of the VRP-TermStructure sleeve tested whether the slope of the VIX futures curve (VX2 – VX1) could support a directional short-volatility rule. The specification followed the Phase-0 standard:

- No vol targeting
- No z-scoring or normalization
- No overlays or filters
- Pure sign-only trading rule

Phase-0 results showed economic failure (not technical failure). See `SOTs/DIAGNOSTICS.md` § "VRP-TermStructure Phase-0 Diagnostics" for full metrics and analysis.

Confirmed correct data alignment, signal generation, and PnL mechanics.

This constitutes an economic failure, not a technical failure. Per the Sleeve Lifecycle:

- Sleeves failing Phase-0 are not advanced to Phase-1
- Sleeves with structurally invalid economic behavior are PARKED
- Documentation of failure must be retained for reproducibility and future research context

VRP-TermStructure is therefore marked PARKED with no active development until revisited under a revised economic hypothesis (e.g., term-structure as a regime filter or crisis indicator).

**Case Study: VRP-RollYield — Borderline Phase-0 Result**

The VRP-RollYield sleeve tested a sign-only roll-down carry idea: short VX1 when the front future is above spot VIX on a per-day-to-expiry basis. The Phase-0 specification followed the standard rules:

- No vol targeting
- No z-scoring or overlays
- Simple sign-only rule:
  - Compute roll = VX1 – VIX and roll_yield = roll / days_to_expiry
  - signal = -1 if roll_yield > 0, else 0

Phase-0 performance (2020-01-01 to 2025-10-31):

- Sharpe: +0.02 (fails the ≥ 0.10 Phase-0 threshold)
- MaxDD: –85.65% (slightly beyond the catastrophic drawdown line)
- Signal distribution: non-degenerate (~75% short, ~25% flat)

Diagnostics confirmed:

- Correct data loading and alignment
- Correct signal generation and position lagging
- PnL mechanics consistent with other VRP sleeves

Interpretation:

Unlike VRP-Core (which showed a small but clearly positive Sharpe in Phase-0) and VRP-Convergence, the simple roll-yield sign rule delivered only a borderline, near-zero edge with a catastrophic drawdown profile. Under the Sleeve Lifecycle rules, this is treated as a Phase-0 failure:

- The simple economic mapping is not strong enough to justify Phase-1 engineering.
- The sleeve is PARKED in its current form.
- Roll-down carry may still be revisited with richer modeling (e.g., multi-tenor baskets, additional filters, or integration with other VRP components), but that would constitute a new Phase-0 specification, not a continuation of this one.

4. **Diagnostics**:

   - Run Meta-Sleeve-level diagnostics to see impact of the new atomic sleeve:

     - Compare Meta-Sleeve with and without it using `run_perf_diagnostics.py`.
     - Create experimental strategy profile (e.g., `core_v3_trend_breakout`)
     - Run full backtest and compare against baseline (`core_v3_no_macro`)
     - Evaluate Sharpe, MaxDD, year-by-year performance, per-asset stats

5. **Phase-1B (Refinement Cycle)** - *If Phase-1 fails*:

   - **Example**: Breakout Mid Phase-1B tuning cycle
   - Initial Phase-1 test (10% weight, 50/50 feature weights) failed
   - Refinement approach:
     - Reduce horizon weight (10% → 3%)
     - Test multiple feature weight schemes (70/30, 30/70, 100/0, 0/100)
     - Run systematic comparison across all variants
   - **Outcome**: 70/30 configuration (3% weight) passed all criteria
   - **Key Learning**: Integration failures often require weight reduction and feature weight tuning

6. **Status**:

   - Record in `SOTs/STRATEGY.md` which atomic sleeves within the Meta-Sleeve are currently **enabled** vs **experimental**.
   - Update `TREND_RESEARCH.md` with Phase-0, Phase-1, and Phase-1B results
   - If Phase-1/1B passes, promote to Phase-2 validation
   - If Phase-2 passes, promote to "Active Atomic Sleeve" in production configuration

### 5.1 Completed Sleeve Lifecycle Example: Breakout Mid (50-100d)

**Full Lifecycle Summary**:

1. **Phase-0 (Sanity Check)**: ✅ PASSED
   - Sign-only implementation validated core economic idea
   - Sharpe ≥ 0.2 criteria met
   - Results stored in `reports/sanity_checks/trend/breakout_mid_50_100/latest/` (canonical Phase-0)
   - Historical runs in `reports/sanity_checks/trend/breakout_mid_50_100/archive/`
   - Phase index: `reports/phase_index/trend/breakout_mid_50_100/phase0.txt`

2. **Phase-1 (Initial Integration)**: ❌ FAILED
   - Initial test with 10% horizon weight, 50/50 feature weights
   - Performance degradation (Sharpe: -0.038 vs baseline 0.086)
   - Identified as integration problem, not alpha problem

3. **Phase-1B (Refinement Cycle)**: ✅ PASSED
   - Reduced horizon weight to 3%
   - Tested 4 feature weight schemes: 70/30, 30/70, 100/0, 0/100
   - **Winner**: 70/30 configuration (breakout_50: 70%, breakout_100: 30%)
   - Results: Sharpe 0.0953 vs baseline 0.0857, MaxDD -31.52% vs -32.02%

4. **Phase-2 (Validation)**: ✅ PASSED
   - Full backtest with Phase-1B winning configuration
   - Target window: 2021-01-01 to 2025-11-19
   - All promotion criteria met:
     - Sharpe ≥ baseline: ✅
     - MaxDD ≤ baseline: ✅
     - Robust across years: ✅
     - Stable correlations: ✅
   - **Verdict**: Approved for Phase-3 (production monitoring)

### 5.2 Failed Promotion Example: Canonical Short-Term (21d)

**Canonicalization Attempt for Short-Term Atomic Sleeve** (Nov 2025):

1. **Specification**:
   - **Type**: Parameter canonicalization (not new atomic sleeve; refines existing short-term sleeve)
   - **Existing**: Short-term (21d) with legacy weights (0.5, 0.3, 0.2)
   - **Canonical**: Short-term (21d) with equal-weight (1/3, 1/3, 1/3) composite
   - **Rationale**: Match pattern from Long-Term and Medium-Term canonicalizations (academically grounded equal-weight)
   - **Features**: Same as legacy (ret_21, breakout_21, slope_fast); reversal filter weight 0 in both variants

2. **Phase-0 (Sign-Only Sanity Check)**: ✅ PASSED
   - Script: `scripts/run_trend_short_canonical_phase0.py`
   - Parameters: 21d lookback, skip 5d, equal-weight across assets
   - **Results**: Sharpe 0.31, hit rate 52.5% (passes Sharpe ≥ 0.2 threshold)
   - Results path: `reports/sanity_checks/trend/short_canonical/archive/20251119_153628/`
   - Phase index: `reports/phase_index/trend/short_canonical/phase0.txt`
   - **Verdict**: Phase-0 passed, proceed to Phase-1

3. **Phase-1 (Standalone Canonical vs Legacy)**: ❌ FAILED
   - Script: `scripts/run_trend_short_canonical_phase1.py`
   - Strategy profiles: `short_canonical_phase1` (1/3, 1/3, 1/3) vs `short_legacy_phase1` (0.5, 0.3, 0.2)
   - **Results**:
     - Canonical: CAGR -4.13%, Sharpe -0.324
     - Legacy: CAGR -3.87%, Sharpe -0.308
     - **Delta**: Legacy outperforms by +0.26% CAGR, +0.016 Sharpe
   - Phase index: `reports/phase_index/trend/short_canonical/phase1.txt`
   - **Verdict**: Legacy wins standalone test

4. **Phase-2 (Integrated A/B Test)**: ❌ FAILED
   - Strategy profile: `core_v3_trend_shortcanon_no_macro` vs `core_v3_no_macro` baseline
   - **Results** (2018-01-01 to 2025-10-31):
     - Canonical: CAGR 6.63%, Sharpe 0.5735, MaxDD -17.24%
     - Legacy: CAGR 6.83%, Sharpe 0.5876, MaxDD -16.72%
     - **Delta**: Legacy outperforms by +0.20% CAGR, +0.014 Sharpe, +0.52% MaxDD
   - Phase index: `reports/phase_index/trend/short_canonical/phase2.txt`
   - **Verdict**: Legacy wins integrated test, canonical not promoted

5. **Final Outcome**: ❌ NOT PROMOTED
   - **Conclusion**: Legacy weights (0.5, 0.3, 0.2) remain production standard
   - **Key Insight**: Unlike Long-Term and Medium-Term horizons where equal-weighting improved diversification, Short-Term (21d) benefits from empirically-tuned weights that preserve the economically stronger return signal
   - **Rationale for rejection**: 21d return is more informative than breakout/slope at this short horizon; equal-weighting dilutes the strongest signal
   - **Preservation**: Canonical variant preserved as research option (`variant="canonical"` in code) for future re-testing if universe/regime changes
   - **Documentation**: Logged in `TREND_RESEARCH.md` as tested but not promoted
   - Phase index: `reports/phase_index/trend/short_canonical/phase2.txt`

5. **Status**: Phase-0/1/2 in progress (Nov 2025)
   - Implementation: Complete (variant parameter added to `ShortTermMomentumStrategy`)
   - Integration: Complete (`TSMOMMultiHorizon` supports `short_variant` parameter)
   - Configuration: Complete (strategy profiles added to `configs/strategies.yaml`)
   - Scripts: Complete (Phase-0 and Phase-1 diagnostic scripts created)
   - Documentation: Complete (all docs updated)
   - Next step: Run Phase-0, Phase-1, Phase-2 diagnostics

5. **Phase-3 (Production Integration)**: ✅ COMPLETED
   - Updated `core_v3_no_macro` production profile
   - Horizon weight: 3% for `breakout_mid_50_100`
   - Feature weights: 70% `breakout_50`, 30% `breakout_100`
   - Phase indices updated: `reports/phase_index/trend/breakout_mid_50_100/phase0.txt`, `phase1.txt`, `phase2.txt`
   - Documentation updated across all relevant files

**Key Learnings**:
- Integration failures often require weight reduction and feature weight tuning
- Phase-1B refinement cycle is critical for optimizing integration
- Lower weights can reduce signal conflicts with existing sleeves
- 70/30 blend outperformed pure configurations, indicating synergy

---

## 7. Procedure: Universe / Asset Changes

**Goal:** Don't rerun sleeve lifecycle; do **regression diagnostics** instead.

### 6.1 When This Applies

- Add/remove one or more futures.
- Swap SR3 front ↔ SR3 rank 3.
- Change roll rules that effectively change traded contract.

### 6.2 Checklist

1. Update universe definition (`configs/data.yaml`, etc.) and ensure `MarketData` maps correctly (see Universe section in `SOTs/STRATEGY.md`).

2. Rebuild any **precomputed features** if necessary (FeatureService).

3. Run **sleeve-level diagnostics** for all *active* Meta-Sleeves:

   - For each active Meta-Sleeve:

     - Run its Phase-1 diagnostic script over the standard test window.
   - Confirm:

     - Sharpe didn't collapse unexpectedly.
     - Per-asset stats are sane.
     - No obviously broken series.

4. Run **full strategy** with a known baseline profile (e.g., `core_v3_no_macro`) and new universe.

5. Run `run_perf_diagnostics.py` with:

   - `--run_id` = new universe run
   - `--baseline_id` = previous universe run  

   To quantify the impact of the change.

6. If things look acceptable, document in `SOTs/STRATEGY.md` "Universe change: <date>, <summary>".

> **Important:**  

> We still **keep bad assets** at this stage. Universe pruning is a **Phase-B optimization** task.

---

## 8. Procedure: Parameter Changes (Lookbacks, Weights, etc.)

### 7.1 When This Applies

- Change TSMOM lookbacks (e.g., 252 → 200).
- Change CSMOM horizon weights (0.4, 0.35, 0.25 → something else).
- Change clipping, EWMA half-life, etc.

**⚠️ IMPORTANT: Weight Freeze Policy**

**Trend Meta-Sleeve v2 internal weights (45/28/20/15) are FROZEN during the architecture build-out phase.**

- **Do NOT** sweep or optimize these weights now.
- **Do NOT** run parameter searches on horizon weights.
- These weights are fixed until **Phase-B: Optimization & Pruning**.
- In Phase-B, weights must be optimized **jointly across all sleeves** with proper out-of-sample controls.

This prevents overfitting and keeps us focused on architecture completion rather than premature optimization.

### 7.2 Checklist

1. Update the config (YAML or class config).

2. Run **sleeve-level diagnostics**:

   - Use Phase-1 diagnostic script for that sleeve.
   - Compare to previous config using `run_perf_diagnostics.py` if you saved as a full run.

3. If the change is minor and performance is within an expected band:

   - No need to reclassify sleeve phases.
   - Note in `SOTs/STRATEGY.md` that parameters for that sleeve were updated on <date>.

4. If change is **major** (effectively new behavior):

   - Treat it as a **new atomic sleeve variant**:

     - Document under same Meta-Sleeve in `SOTs/STRATEGY.md`.
     - Optionally give it its own Phase-0 mini-check.

---

## 9. Procedure: Overlay / Allocator Changes

### 8.1 When This Applies

- Macro overlay logic changes (inputs, mapping, thresholds).
- Vol targeting changes (target vol, caps, lookbacks).
- Allocator changes (constraints, method, turnover penalty).

### 8.2 Checklist

1. Update overlay / allocator code and configs (see relevant sections in `SOTs/STRATEGY.md`).

2. Choose one or more **reference strategy profiles**:

   - E.g., `core_v3_no_macro`, and later `core_v4_with_csmom`.

3. Run **full backtests** with:

   - Old overlay/allocator (baseline run_id).
   - New overlay/allocator (variant run_id).

4. Use `run_perf_diagnostics.py` to compare:

   - Core metrics.
   - Year-by-year stats.
   - Per-asset stats.
   - Equity ratio over time.

5. If behavior is consistent and improved, update `SOTs/STRATEGY.md` to point to the new overlay/allocator as the current standard.

---

## 9.1 Allocator Development Lifecycle (v2 Track)

Allocator logic is subject to the same promotion discipline as sleeves. The allocator development follows a structured lifecycle to ensure robust, validated conditional exposure control.

### Allocator Phases

| Phase | Objective | Key Activities |
|-------|-----------|---------------|
| **Allocator Phase-A** | Architecture & invariants (no tuning) | Define allocator architecture, establish invariants, design conditional logic structure |
| **Allocator Phase-B** | Deterministic rule-based allocator | Implement rule-based allocator (no optimization), validate deterministic behavior |
| **Allocator Phase-C** | End-to-end validation | Validate allocator behavior across historical windows, stress test failure modes |
| **Allocator Phase-D** | Production deployment | Freeze allocator logic, version control, deploy to production track |
| **Allocator Phase-E** | Post-production enhancements | Iterative improvements following formal promotion process |

### Phase-A: Architecture & Invariants

**Objective**: Establish allocator design principles without parameter tuning.

**Key Activities:**
- Define allocator inputs (Meta-Sleeve signals, regime indicators, portfolio state)
- Establish invariants (e.g., no leverage beyond X, always maintain Y% cash buffer)
- Design conditional logic structure (regime detection, crisis response, leverage control)
- Document architectural decisions

**Deliverables:**
- Allocator architecture specification
- Invariant definitions
- Design document

### Phase-B: Deterministic Rule-Based Allocator

**Objective**: Implement rule-based allocator with deterministic behavior.

**Key Activities:**
- Implement allocator logic based on Phase-A architecture
- Use fixed thresholds and rules (no optimization)
- Ensure deterministic output for given inputs
- Validate rule-based behavior

**Deliverables:**
- Functional allocator implementation
- Deterministic behavior validation
- Rule documentation

### Phase-C: End-to-End Validation

**Objective**: Validate allocator behavior across historical windows and stress scenarios.

**Key Activities:**
- Run allocator on expanded historical window (see Data Expansion Protocol)
- Validate behavior during crisis periods (2020 Q1, 2022, etc.)
- Test failure modes and edge cases
- Verify portfolio-level protection (not just instrument-level)

**Deliverables:**
- Historical validation results
- Stress test results
- Failure mode documentation

### Phase-D: Production Deployment

**Objective**: Freeze allocator logic and deploy to production.

**Key Activities:**
- Freeze allocator code and parameters
- Version control (e.g., Allocator v1)
- Deploy to production track
- Establish monitoring and alerting

**Deliverables:**
- Versioned allocator release
- Production deployment documentation
- Monitoring setup

### Phase-E: Post-Production Enhancements

**Objective**: Iterative improvements following formal promotion process.

**Key Activities:**
- Develop enhancements in research track
- Paper integration with production copy
- Portfolio-level diagnostics
- Formal promotion decision
- Versioned production release (e.g., Allocator v2)

**Deliverables:**
- Enhanced allocator version
- Promotion decision documentation
- Versioned release

---

## 9.1.1 Allocator v1 Production Procedures (December 2024)

**Status**: Phase-D Complete (Production-Ready)

Allocator v1 has completed Phases A-D and is production-ready. The following procedures govern how to use and validate Allocator v1.

### Running Allocator v1 in Research Mode

**Default: Artifacts Only (No Weight Scaling)**

By default, Allocator v1 computes and saves all artifacts but does NOT scale portfolio weights. This allows for research and validation without affecting backtest results.

**Command:**
```bash
python run_strategy.py --strategy_profile core_v9 --start 2024-01-01 --end 2024-12-15
```

**Configuration (default):**
```yaml
allocator_v1:
  enabled: false  # Artifacts computed but not applied
```

**Artifacts Generated:**
- `allocator_state_v1.csv` - 10 daily state features
- `allocator_regime_v1.csv` - Daily regime labels
- `allocator_risk_v1.csv` - Daily risk scalars (computed)
- `allocator_risk_v1_applied.csv` - Lagged risk scalars (ready for application)
- Metadata JSON files for each layer

### Two-Pass Audit Workflow (Recommended Validation)

The two-pass audit is the canonical way to validate Allocator v1 behavior before enabling it for weight scaling.

**Purpose:**
- Compare baseline (allocator off) vs scaled (allocator applied)
- Validate that allocator reduces MaxDD without destroying returns
- Audit regime transitions and de-risking events
- Ensure deterministic, reproducible behavior

**Command:**
```bash
python scripts/diagnostics/run_allocator_two_pass.py \
  --strategy_profile core_v9 \
  --start 2024-01-01 \
  --end 2024-12-15
```

**What It Does:**
1. **Pass 1 (Baseline)**: Runs backtest with `allocator_v1.enabled=false`
   - Generates portfolio returns without allocator
   - Computes and saves all allocator artifacts
   - Produces `allocator_risk_v1_applied.csv`

2. **Pass 2 (Scaled)**: Re-runs backtest with `mode="precomputed"`
   - Loads `allocator_risk_v1_applied.csv` from Pass 1
   - Applies scalars with 1-rebalance lag
   - Generates scaled portfolio returns

3. **Comparison Report**: Generates `two_pass_comparison.md` and `.json`
   - Performance metrics (CAGR, vol, Sharpe, MaxDD, worst month/quarter)
   - Allocator usage statistics (% rebalances scaled, mean/min/max scalar)
   - Top 10 de-risking events
   - Regime distribution and transition counts

**Outputs:**
- `reports/runs/{baseline_run_id}/` - Baseline artifacts
- `reports/runs/{scaled_run_id}/` - Scaled artifacts
- `reports/runs/{scaled_run_id}/two_pass_comparison.md` - Human-readable comparison
- `reports/runs/{scaled_run_id}/two_pass_comparison.json` - Machine-readable comparison

### Running Individual Allocator Diagnostics

If you have an existing run and want to re-compute or inspect specific allocator layers:

**Recompute State Features:**
```bash
python scripts/diagnostics/run_allocator_state_v1.py --run_id <run_id>
```

**Recompute Regime Classification:**
```bash
python scripts/diagnostics/run_allocator_regime_v1.py --run_id <run_id>
```

**Recompute Risk Scalars:**
```bash
python scripts/diagnostics/run_allocator_risk_v1.py --run_id <run_id>
```

These scripts load existing run artifacts and recompute the respective allocator layer, saving updated artifacts back to the run directory.

### Enabling Allocator v1 for Production

**⚠️ WARNING**: Do not enable allocator in production until two-pass audit validates expected behavior.

**Configuration (production):**
```yaml
allocator_v1:
  enabled: true
  mode: "precomputed"
  precomputed_run_id: "<baseline_run_id>"
  precomputed_scalar_filename: "allocator_risk_v1_applied.csv"
  apply_missing_scalar_as: 1.0
```

**Behavior:**
- Loads precomputed scalars from baseline run
- Applies scalars with 1-rebalance lag at each rebalance date
- Scales raw weights: `weights_scaled = weights_raw * risk_scalar_applied[t-1]`
- Saves both `weights_raw.csv` and `weights_scaled.csv`

### Allocator v1 Validation Checklist

Before enabling allocator in production:

- [ ] Two-pass audit completed with expected MaxDD reduction
- [ ] Regime classifications are sticky (transitions <20 over multi-year period)
- [ ] Top de-risk events align with known stress periods (2020 Q1, 2022, etc.)
- [ ] Feature coverage is 100% (all 10 features present if optional data available)
- [ ] Row drop rate <5% (no major data quality issues)
- [ ] Comparison report shows allocator acts as risk governor (not return enhancer)
- [ ] Worst month/quarter metrics improved vs baseline
- [ ] No unexpected regime thrashing during normal periods

### Phase 1C: Risk Targeting + Allocator Integration (Completion Checklist)

**Status:** ✅ **COMPLETE** (January 2026)

**Phase 1C Acceptance Criteria:**

**Golden Proof Run:**
- [x] **Run ID:** `rt_alloc_h_apply_precomputed_2024`
- [x] **Config:** `configs/proofs/phase1c_allocator_apply.yaml`
- [x] **Validator:** `scripts/diagnostics/validate_phase1c_completion.py <run_id>` must PASS

**Acceptance Criteria (All Passed):**
1. [x] **RT Artifacts:** Panel data bug fixed (all 13 instruments per date, gross matches logs)
2. [x] **RT Layer:** Leverage calculation correct, weight scaling correct
3. [x] **Allocator Computation:** Regimes + scalars correct (42% active, min 0.68)
4. [x] **Allocator Application:** Multipliers applied to weights (% active > 0%)
5. [x] **Returns Differentiation:** RT + Alloc-H returns differ from RT-only (difference > 1e-6)
6. [x] **Weight Scaling Verification:** `final_weights ≈ post_rt_weights * multiplier` (error < 0.01)
7. [x] **ExecSim Logs:** "Risk scalars applied: X/52 rebalances" where X > 0
8. [x] **Contract Tests:** All tests pass (`test_risk_targeting_contracts.py`, `test_allocator_profile_activation.py`)

**Phase 1C Validation Command:**
```bash
# Validate Phase 1C completion
python scripts/diagnostics/validate_phase1c_completion.py rt_alloc_h_apply_precomputed_2024

# Expected output: "OVERALL: PASS - Allocator was applied!"
```

**Proof Config Location:**
- **Stable config:** `configs/proofs/phase1c_allocator_apply.yaml`
- **Temp config (legacy):** `configs/temp_phase1c_proof_precomputed.yaml` (deprecated, use proofs/ version)

**Important Nuance (Documented):**

Phase 1C validation uses a **two-step process**:
1. **Step 1:** Compute allocator scalars (`rt_alloc_h_apply_proof_2024` in `compute` mode)
2. **Step 2:** Apply scalars via `precomputed` mode (`rt_alloc_h_apply_precomputed_2024`)

**This proves:**
- ✅ Allocator application path works correctly
- ✅ Config plumbing is correct
- ✅ Weight scaling is deterministic and auditable
- ✅ End-to-end integration is sound

**Behavioral Difference (Phase 2/3 Validation):**

There is a difference between:
- **`compute` mode:** Compute-and-apply in-loop (live-like behavior, has warmup issues)
- **`precomputed` mode:** Compute once, apply later (replay behavior, production-ready)

**Phase 1C proves the application path and config plumbing.**  
**Phase 2/3 will validate compute-and-apply stability** (or explicitly choose `precomputed` for paper-live v0 if that's acceptable).

**Artifact Validation:**
```bash
# Test RT artifacts integrity
python scripts/diagnostics/test_rt_artifact_fix.py <run_id>

# Expected: All tests PASS (13 instruments per date, gross consistency)
```

**Phase 1C Completion Declaration:**

Phase 1C is complete when:
- ✅ All acceptance criteria pass
- ✅ Validation script returns PASS
- ✅ All contract tests pass
- ✅ Documentation updated in SOTs

**Date Completed:** 2026-01-10  
**Signed Off:** AI Agent + User Validation

---

### Phase 2: Engine Policy v1 (Completion Checklist)

**Status:** ✅ **COMPLETE** (January 2026)

**Phase 2 Acceptance Criteria:**

**Golden Proof Runs:**
- [x] **Compute Mode Run ID:** `policy_trend_gamma_compute_proof_2024`
- [x] **Compute Mode Config:** `configs/proofs/phase2_policy_trend_gamma_compute.yaml`
- [x] **Precomputed Mode Run ID:** `policy_trend_gamma_apply_precomputed_2024`
- [x] **Precomputed Mode Config:** `configs/proofs/phase2_policy_trend_gamma_apply_precomputed.yaml`
- [x] **Validator:** `scripts/diagnostics/validate_phase2_policy_v1.py <run_id>` must PASS

**Acceptance Criteria (All Must Pass):**
1. [x] **Artifacts Exist:** `engine_policy_state_v1.csv`, `engine_policy_applied_v1.csv`, `engine_policy_v1_meta.json` present
2. [x] **Determinism:** Re-run with same config yields identical `engine_policy_applied_v1.csv` (or identical hash)
3. [x] **Lag Correct:** Multiplier used at rebalance t equals policy decision from t-1 (lag=1)
4. [x] **Policy Has Teeth:** Compare baseline vs policy-enabled run — weights differ on at least one rebalance when stress triggers
5. [x] **Isolation:** Only trend and VRP engines affected (trend gates 15/253 = 5.9%, VRP gates 3/253 = 1.2%); other engines unchanged

**Architectural Constraints (Enforced in Code):**
- [x] Engine Policy is a validity filter, NOT an optimizer
- [x] Binary gate only (multiplier ∈ {0, 1})
- [x] Inputs are context features (gamma/vol-of-vol), NOT portfolio metrics
- [x] Does NOT use: portfolio drawdown, correlation, sizing (allocator territory)
- [x] **Hierarchy Rule:** If policy gates engine OFF, nothing downstream can resurrect it

**Phase 2 Validation Commands:**
```bash
# Step 1: Run compute mode proof (generates multipliers)
python run_strategy.py --config configs/proofs/phase2_policy_trend_gamma_compute.yaml \
    --run_id policy_trend_gamma_compute_proof_2024

# Step 2: Run precomputed mode proof (applies multipliers from step 1)
python run_strategy.py --config configs/proofs/phase2_policy_trend_gamma_apply_precomputed.yaml \
    --run_id policy_trend_gamma_apply_precomputed_2024

# Step 3: Validate
python scripts/diagnostics/validate_phase2_policy_v1.py policy_trend_gamma_apply_precomputed_2024

# Expected output: "OVERALL: PASS - Engine Policy v1 validated!"
```

**A/B Run Workflow (Evaluation):**
```bash
# Baseline: policy disabled
python run_strategy.py --strategy_profile core_v9 --run_id baseline_no_policy \
    --override "engine_policy_v1.enabled=false"

# Variant: policy enabled for trend
python run_strategy.py --strategy_profile core_v9 --run_id variant_policy_trend \
    --override "engine_policy_v1.enabled=true" --override "engine_policy_v1.mode=compute"

# Compare runs
python scripts/diagnostics/compare_two_runs.py baseline_no_policy variant_policy_trend
```

**Important:** A/B comparison is for checking policy mechanics (does it trigger? does it isolate correctly?), NOT for Sharpe approval.

**Date Completed:** 2026-01-12  
**Signed Off:** AI Agent + User Validation

---

### Allocator v1 Known Limitations

**Warmup Period:**
- State features require 60-day rolling windows
- Early dates (first ~60 days) will have insufficient data
- Two-pass audit sidesteps this by using precomputed scalars
- Future Stage 9 will implement incremental state computation

**Optional Features:**
- `trend_breadth_20d` requires `trend_unit_returns.csv` (Trend sleeve must be active)
- `sleeve_concentration_60d` requires `sleeve_returns.csv` (multi-sleeve portfolio)
- These features are excluded (not set to NaN) if inputs unavailable
- Regime classifier still works with 8 core features

**Mode Recommendations:**
- Use `mode: "off"` for initial research and artifact generation
- Use `mode: "precomputed"` for two-pass audit and production
- Avoid `mode: "compute"` until warmup period is resolved

### Stage 6: Production Mode (LOCKED)

**Decision:** `mode="off"` is the default for Allocator v1 (artifacts only, no scaling). `mode="precomputed"` is the production mode when explicitly configured with a valid `precomputed_run_id`.

**Rationale:**
- ✅ No warmup period issues (baseline has full history)
- ✅ No circular dependency (scalars from baseline portfolio)
- ✅ Fully deterministic (same baseline → same results)
- ✅ Complete audit trail (baseline vs scaled comparison)
- ✅ Institutional standard (compute from history, apply to forward)

**Mode Status:**
- **`precomputed`**: Production-ready ✅
- **`compute`**: Research-only (has warmup issues, not production-safe)
- **`off`**: Always safe (baseline generation)

**See:** `docs/ALLOCATOR_V1_PRODUCTION_MODE.md` for complete production mode specification.

### Stage 6.5: Stability & Sanity Review

**Purpose:** Qualitative validation before production deployment (NOT tuning or optimization)

**Validation Questions:**
1. **Does the allocator reduce MaxDD meaningfully?** (Target: 2-5% reduction)
2. **Does it avoid killing returns in NORMAL regimes?** (Target: CAGR reduced <1%)
3. **Are regime transitions sparse and intuitive?** (Target: <20 per year)
4. **Does it correctly flag known stress windows?** (2020 Q1, 2022 must appear in top de-risk events)

**Decision Criteria:**
- **PASS**: All questions "mostly yes" → Lock v1 and deploy
- **FAIL**: Any critical issue → Review thresholds and retry

**Workflow:**
1. Run two-pass audit on canonical window (2020-2025)
2. Review `two_pass_comparison.md` for metrics
3. Complete Stage 6.5 validation checklist
4. Make go/no-go decision

**See:** `docs/ALLOCATOR_V1_STAGE_6_5_VALIDATION.md` for detailed validation checklist and sign-off template.

**Key Principle:** "Mostly yes" is good enough. Do not over-optimize before going live. Stage 7 (threshold tuning) is post-deployment.

### Future Enhancements (Phase-E)

**Stage 7**: Threshold tuning against historical stress events (post-deployment)  
**Stage 8**: Convexity overlays (VIX calls) gated by regime (post-deployment)  
**Stage 9**: True incremental state computation (resolve warmup period, enables `compute` mode)

---

## 9.2 Post-Production Sleeve Additions

**Formal Institutional Rule**: Once a system is live, new sleeves are developed in parallel and never injected directly into production.

### Required Steps for Post-Production Sleeve Addition

1. **Standalone research (Phase-0 / Phase-1)**
   - New sleeve developed and validated in research track
   - Follows standard sleeve lifecycle (Phase-0 → Phase-1)
   - No capital dependency

2. **Paper integration with production copy**
   - Integrate new sleeve with production system copy
   - Run parallel paper trading simulation
   - Compare performance vs production baseline

3. **Portfolio-level diagnostics**
   - Analyze portfolio metrics (Sharpe, CAGR, MaxDD, Vol)
   - Evaluate crisis-period performance
   - Assess correlation and diversification impact
   - Validate sleeve-level loss attribution

4. **Promotion decision**
   - Formal review of research results
   - Decision to promote or reject
   - If promoted, determine allocation weight

5. **Versioned production release (e.g., Core v10)**
   - New sleeve integrated into versioned production release
   - Production system frozen at new version
   - Monitoring and validation established

### Governance Rules

- **No direct injection**: New sleeves never bypass research track
- **Parallel development**: Research operates independently of production
- **Formal promotion**: All additions require explicit promotion decision
- **Version control**: Production releases are versioned and frozen
- **Capital protection**: Production capital is never exposed to untested sleeves

---

## 10. When Do We Examine "Bad Assets"?

You asked specifically: **"At what stage do we look at bad assets?"**

We'll encode that explicitly:

### Stage 1 – Development (Current Phase)

- We **observe** per-asset Sharpe and contributions in diagnostics (as we already saw with SR3, 6J, CL in CSMOM).
- We **do not** prune or reweight just because they look bad.
- The purpose is understanding, not optimization.

### Stage 2 – Architecture Complete

Once:

- Trend Meta-Sleeve + CSMOM + other key Meta-Sleeves are implemented and Phase-1/2 integrated.
- Macro overlay, vol targeting, and allocator are in place.
- At least one **full multi-sleeve profile** is running (e.g., `core_vX_with_trend_csmom_...`).

Then we run a **system-wide optimization / pruning cycle**, where we:

1. Freeze a **reference configuration** and run a long backtest with artifacts stored under a clear `run_id`.

2. Use `run_perf_diagnostics.py` to examine per-asset stats across:

   - Each Meta-Sleeve individually.
   - The combined portfolio.

3. **Weight Optimization (Phase-B Only)**:
   
   - **Trend Meta-Sleeve v2 internal weights** (45/28/20/15) can now be optimized.
   - **Critical**: Weights must be optimized **jointly across all sleeves** (Trend, CSMOM, etc.), not in isolation.
   - **Requirement**: Use proper out-of-sample controls (train/validation/test splits, walk-forward analysis, etc.).
   - **Documentation**: Record optimization methodology, results, and rationale in `SOTs/STRATEGY.md` or `OPTIMIZATION_NOTES.md`.

4. For each asset:

   - If **consistently negative Sharpe** across multiple sleeves and subperiods, and **no diversification benefit**:

     - Candidate to be removed from that sleeve (asset multipliers, sleeve-specific universes).
   - If weak in one sleeve but strong in another:

     - Keep it; let sleeves that understand it carry the load.

5. Implement **asset multipliers** or per-sleeve universes only at this stage, *not before*.

Record these decisions and their rationale in `SOTs/STRATEGY.md` (or a dedicated `OPTIMIZATION_NOTES.md`).

---

## 11. Quick Reference Checklist

When you do anything, ask:

> **What kind of change is this?**

Then:

### A. New Meta-Sleeve

- [ ] Design spec in `SOTs/STRATEGY.md`
- [ ] Phase-0 script + run
- [ ] Phase-1 implementation + diagnostics
- [ ] Phase-2 integration + full-run diagnostics
- [ ] Status updated in `SOTs/STRATEGY.md` and `SOTs/DIAGNOSTICS.md`

### B. New Atomic Sleeve

- [ ] Design snippet in Meta-Sleeve section
- [ ] Optional Phase-0 mini-check
- [ ] Phase-1 implementation
- [ ] Meta-Sleeve diagnostics with/without atomic
- [ ] Status updated

### C. Universe Change (add/remove/swap asset)

- [ ] Update universe config
- [ ] Rebuild features if needed
- [ ] Re-run sleeve Phase-1 diagnostics
- [ ] Re-run one or more full strategy profiles + perf diagnostics
- [ ] Document change; **no pruning yet**

### D. Parameter Change

- [ ] Update config
- [ ] Re-run relevant Phase-1 diagnostics
- [ ] (Optional) run full strategy profile & compare
- [ ] Note change in `SOTs/STRATEGY.md` if significant

### E. Overlay / Allocator Change

- [ ] Update overlay/allocator code/config
- [ ] Run full baseline + variant profiles
- [ ] Compare via `run_perf_diagnostics.py`
- [ ] Promote new overlay if better / more robust

### F. Asset Pruning / Multipliers

- [ ] Only after architecture is complete
- [ ] Use per-asset diagnostics across sleeves
- [ ] Implement multipliers / exclusions per sleeve
- [ ] Document decisions and reasoning

---

## 12. Re-Test Conditions for Parked Sleeves

A **parked sleeve** (Phase-1 FAIL) is an idea that has been tested and rejected, but is **not deleted**. Parked sleeves are automatically re-tested when the data regime or architecture materially expands.

### 8.1 What is a Parked Sleeve?

A parked sleeve is an atomic sleeve or meta-sleeve that:
- Passed Phase-0 (or partially passed)
- Failed Phase-1 (did not improve baseline performance)
- Has academic justification or economic rationale
- Is documented in `SOTs/STRATEGY.md` (high-level) and research docs (detailed)

**Examples**:
- **Persistence (Trend Meta-Sleeve)**: Phase-0 partial pass, Phase-1 fail
- **Carry Meta-Sleeve**: Phase-0 fail
- **Rates Curve RV Meta-Sleeve**: Phase-0 fail

### 8.2 Re-Test Triggers

A parked sleeve **must be re-tested** when any of the following occur:

#### Trigger 1: Universe Expansion (≥25% growth)
- **When**: Traded universe grows by ≥25% (e.g., from 13 to 20+ assets)
- **Rationale**: Many ideas (especially persistence, cross-asset effects) require broad, diverse universes
- **Action**: Re-run Phase-0 and Phase-1 for the parked sleeve

#### Trigger 2: Historical Expansion (≥5 years added)
- **When**: Total backtest length increases by ≥5 years (e.g., extending to 2010 or earlier)
- **Rationale**: Access to different market regimes may support previously weak ideas
- **Action**: Re-run Phase-0 and Phase-1 for the parked sleeve

#### Trigger 3: Architecture Change (Related Component Added)
- **When**: A closely related component is added or changed in the same Meta-Sleeve
- **Examples**:
  - Adding vol-adjusted trend → re-test Persistence
  - Adding cross-asset trend confirmation → re-test Persistence
  - Adding short-term reversal filters → re-test Persistence
- **Rationale**: Parked ideas may add value only when combined with new enhancements
- **Action**: Re-run Phase-0 and Phase-1 for the parked sleeve

#### Trigger 4: Feature Platform Upgrades
- **When**: FeatureService adds materially new features relevant to the parked idea
- **Examples**:
  - Cross-asset trend validation features
  - Multi-horizon acceleration features
  - Volatility-adjusted acceleration
- **Rationale**: New features may enable better implementation of parked ideas
- **Action**: Re-run Phase-0 and Phase-1 for the parked sleeve

#### Trigger 5: Market Regime Shifts (Optional)
- **When**: More than 3 years of new live data accumulate
- **Rationale**: New regime may support previously weak ideas
- **Action**: Re-run Phase-0 and Phase-1 for the parked sleeve (optional, not mandatory)

### 8.3 Re-Test Procedure

When a trigger occurs:

1. **Document the trigger**: Note in research docs why the sleeve is being re-tested
2. **Repeat Phase-0**: Run sign-only sanity check with current data
3. **Repeat Phase-1**: If Phase-0 passes, implement engineered version and run diagnostics
4. **Evaluate**: Compare Phase-1 results against baseline
5. **Decision**:
   - If Phase-1 **passes**: Sleeve becomes eligible for Phase-2 integration
   - If Phase-1 **fails**: Sleeve remains parked, update documentation with new results

### 8.4 Overfitting Prevention

Re-testing parked sleeves does **not** constitute overfitting because:

- **Definition is fixed**: The sleeve definition (per academic paper or economic rationale) does not change
- **Thresholds are fixed**: Phase-0 (Sharpe ≥ 0.2) and Phase-1 (must improve baseline) remain constant
- **No parameter tuning**: We do not tweak parameters until something "passes"
- **Architecture freeze**: We freeze architecture between phases (see Weight Freeze Policy)

Re-testing when the data regime expands is **not overfitting** — it's seeing whether a previously parked idea becomes usable when conditions improve.

### 8.5 Documentation Requirements

When a parked sleeve is re-tested:

- **SOTs/STRATEGY.md**: Update status (if re-test passes, move from PARKED to active)
- **Research docs** (e.g., `TREND_RESEARCH.md`): Add new Phase-0/Phase-1 results section
- **SOTs/PROCEDURES.md**: Note the trigger that caused re-test (for audit trail)

---

End of procedures.

