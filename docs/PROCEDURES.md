# Procedures: How to Add / Change Sleeves, Assets, and Parameters

**Purpose:**  

This document defines **when** and **how** to run the sleeve lifecycle (Phase-0 → Phase-3) for any change to the Futures-Six framework:

- Adding a new **Meta-Sleeve** (e.g., Cross-Sectional Momentum, Vol Risk Premium)
- Adding a new **Atomic Sleeve** inside a Meta-Sleeve
- Changing the **asset universe**
- Changing **parameters** (lookbacks, weights, overlays)
- Deciding **when to look at bad assets** and when *not* to touch them

It complements:

- `STRATEGY.md` – architecture & execution flow (what the system does)
- `DIAGNOSTICS.md` – diagnostics capabilities & tools (how we analyze runs)

This file is about **process** and **checklists**.

---

## 1. Core Principles

1. **Architecture first, optimization later**

   - Goal of current phase: **get all Meta-Sleeves and core overlays implemented and wired in**.
   - We **do not** aggressively tune asset lists, ranks, or parameters until the architecture is complete.

2. **Sleeve lifecycle is about the *idea*, not every tiny config tweak**

   From `STRATEGY.md` (sleeve lifecycle):

   - **Phase-0 – Simple Sanity Check**
     - Sign-only, no overlays, equal-weight / DV01-neutral.
     - Objective: "Does the economic idea have any edge at all?"
   - **Phase-1 – Clean Implementation**
     - Proper feature engineering, z-scoring, cross-sectional ranking, etc.
   - **Phase-2 – Overlay Integration**
     - Macro regime filters, vol targeting, allocator integration.
   - **Phase-3 – Production + Monitoring**
     - Full runs, performance monitoring, alerts, regression tests.

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

## 3. Change Types (Taxonomy)

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

## 3. When Do We Run Which Phases?

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

## 4. Procedure: Adding a New Meta-Sleeve

### 4.1 Design Spec

Before touching code, create a short design section in `STRATEGY.md`:

- Economic thesis (why this sleeve should have edge).
- Universe it will trade.
- Key features (e.g., returns, curve, carry, macro inputs).
- Expected behavior (when it should make / lose money).

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

6. Log result in `DIAGNOSTICS.md` under **Phase-0 Sanity Checks** and in `STRATEGY.md` under that Meta-Sleeve's status.   

**If Phase-0 fails → Meta-Sleeve stays disabled.**

### 4.3 Phase-1 Checklist (Clean Implementation)

Once Phase-0 passes:

1. Implement production-quality sleeve:

   - Feature engineering (multi-horizon, z-scores, etc.).
   - Vol normalization inside sleeve if appropriate.
   - Cross-sectional ranking logic where applicable.

2. Integrate into `FeatureService` and strategy agents (see `STRATEGY.md` sections for existing sleeves as templates).

3. Add a **Phase-1 diagnostics script**, e.g.:

   - `src/diagnostics/<sleeve>_phase1.py`
   - `scripts/run_<sleeve>_phase1.py`

4. Run Phase-1 diagnostics:

   - Compare to Phase-0 (Sharpe, MaxDD, Vol).
   - Confirm behavior matches the idea (e.g., market-neutral, cross-sectional structure).

5. Update phase index: `python scripts/update_phase_index.py <meta_sleeve> <sleeve_name> phase1 <run_id>`

6. Update `STRATEGY.md` status:

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

> **Baseline Profiles (as of Nov 2025)**
> - `core_v3_no_macro`: Trend-only baseline. Used when testing new Trend atomic sleeves or when isolating Trend behavior.
> - `core_v4_trend_csmom_no_macro`: Multi-sleeve baseline. Used when testing new Meta-Sleeves (Carry, Curve RV, VRP, Value, etc.) or overlays on top of Trend + CSMOM.
>
> When adding a new Meta-Sleeve, default to using `core_v4_trend_csmom_no_macro` as the baseline profile unless the experiment specifically concerns Trend-only behavior.

### 4.5 Phase-3 Checklist (Production + Monitoring)

1. Promote sleeve + strategy profile to "production" set.

2. For each production run, ensure:

   - `run_id` is logged and frozen.
   - Diagnostics run vs baseline.
   - Results summarized in `DIAGNOSTICS.md` under **Production Monitoring**.

3. Only in **Phase-3 / Optimization** do we start:

   - Asset-level pruning.
   - Aggressive parameter tuning.
   - Per-sleeve asset multipliers.

---

## 5. Procedure: Adding an Atomic Sleeve to an Existing Meta-Sleeve

**Example: Adding Trend Breakout 50-100d inside the Trend Meta-Sleeve (Phase-0 lite → Phase-1 → Trend-level diagnostics).**

1. **Spec**:  

   - Add a short section in `STRATEGY.md` under that Meta-Sleeve:

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

   - Record in `STRATEGY.md` which atomic sleeves within the Meta-Sleeve are currently **enabled** vs **experimental**.
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

## 6. Procedure: Universe / Asset Changes

**Goal:** Don't rerun sleeve lifecycle; do **regression diagnostics** instead.

### 6.1 When This Applies

- Add/remove one or more futures.
- Swap SR3 front ↔ SR3 rank 3.
- Change roll rules that effectively change traded contract.

### 6.2 Checklist

1. Update universe definition (`configs/data.yaml`, etc.) and ensure `MarketData` maps correctly (see Universe section in `STRATEGY.md`).

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

6. If things look acceptable, document in `STRATEGY.md` "Universe change: <date>, <summary>".

> **Important:**  

> We still **keep bad assets** at this stage. Universe pruning is a **Phase-B optimization** task.

---

## 7. Procedure: Parameter Changes (Lookbacks, Weights, etc.)

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
   - Note in `STRATEGY.md` that parameters for that sleeve were updated on <date>.

4. If change is **major** (effectively new behavior):

   - Treat it as a **new atomic sleeve variant**:

     - Document under same Meta-Sleeve in `STRATEGY.md`.
     - Optionally give it its own Phase-0 mini-check.

---

## 8. Procedure: Overlay / Allocator Changes

### 8.1 When This Applies

- Macro overlay logic changes (inputs, mapping, thresholds).
- Vol targeting changes (target vol, caps, lookbacks).
- Allocator changes (constraints, method, turnover penalty).

### 8.2 Checklist

1. Update overlay / allocator code and configs (see relevant sections in `STRATEGY.md`).

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

5. If behavior is consistent and improved, update `STRATEGY.md` to point to the new overlay/allocator as the current standard.

---

## 9. When Do We Examine "Bad Assets"?

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
   - **Documentation**: Record optimization methodology, results, and rationale in `STRATEGY.md` or `OPTIMIZATION_NOTES.md`.

4. For each asset:

   - If **consistently negative Sharpe** across multiple sleeves and subperiods, and **no diversification benefit**:

     - Candidate to be removed from that sleeve (asset multipliers, sleeve-specific universes).
   - If weak in one sleeve but strong in another:

     - Keep it; let sleeves that understand it carry the load.

5. Implement **asset multipliers** or per-sleeve universes only at this stage, *not before*.

Record these decisions and their rationale in `STRATEGY.md` (or a dedicated `OPTIMIZATION_NOTES.md`).

---

## 10. Quick Reference Checklist

When you do anything, ask:

> **What kind of change is this?**

Then:

### A. New Meta-Sleeve

- [ ] Design spec in `STRATEGY.md`
- [ ] Phase-0 script + run
- [ ] Phase-1 implementation + diagnostics
- [ ] Phase-2 integration + full-run diagnostics
- [ ] Status updated in `STRATEGY.md` and `DIAGNOSTICS.md`

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
- [ ] Note change in `STRATEGY.md` if significant

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

## 8. Re-Test Conditions for Parked Sleeves

A **parked sleeve** (Phase-1 FAIL) is an idea that has been tested and rejected, but is **not deleted**. Parked sleeves are automatically re-tested when the data regime or architecture materially expands.

### 8.1 What is a Parked Sleeve?

A parked sleeve is an atomic sleeve or meta-sleeve that:
- Passed Phase-0 (or partially passed)
- Failed Phase-1 (did not improve baseline performance)
- Has academic justification or economic rationale
- Is documented in `STRATEGY.md` (high-level) and research docs (detailed)

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

- **STRATEGY.md**: Update status (if re-test passes, move from PARKED to active)
- **Research docs** (e.g., `TREND_RESEARCH.md`): Add new Phase-0/Phase-1 results section
- **PROCEDURES.md**: Note the trigger that caused re-test (for audit trail)

---

End of procedures.

