# Allocator v1 — Stage 6.5 Stability & Sanity Review

**Version:** v1.0  
**Purpose:** Qualitative validation before production deployment  
**Date:** December 2024

---

## Purpose

**Stage 6.5 is NOT tuning or optimization.**

This is a qualitative stability review to answer the question:

**"Does Allocator v1 behave as a sensible risk governor?"**

If the answer is "mostly yes," **lock v1 and deploy**. Do not tune thresholds yet.

---

## Validation Questions

### 1. Does the allocator reduce MaxDD meaningfully?

**What to check:**
- Compare baseline vs scaled MaxDD in two-pass report
- Target: 2-5% MaxDD reduction (e.g., -15% → -12%)

**Review:**
- [ ] MaxDD reduced by at least 2%
- [ ] Worst month improved
- [ ] Worst quarter improved

**Interpretation:**
- ✅ **PASS**: MaxDD reduced by 2-5%, worst periods improved
- ⚠️ **CAUTION**: MaxDD reduced by <2% (allocator not protective enough)
- ❌ **FAIL**: MaxDD increased (allocator counterproductive)

**If FAIL:** Do not deploy. Review regime thresholds for stress detection sensitivity.

---

### 2. Does it avoid killing returns in NORMAL regimes?

**What to check:**
- Compare baseline vs scaled CAGR and Sharpe
- Check % of time spent in NORMAL regime
- Check mean risk scalar (should be close to 1.0)

**Review:**
- [ ] CAGR reduced by <1% (slight reduction acceptable)
- [ ] Sharpe preserved or improved
- [ ] Time in NORMAL regime: 70-80%
- [ ] Mean risk scalar: 0.90-0.98

**Interpretation:**
- ✅ **PASS**: CAGR slightly reduced, Sharpe preserved, mostly in NORMAL regime
- ⚠️ **CAUTION**: CAGR reduced by 1-2%, or mean scalar <0.90
- ❌ **FAIL**: CAGR reduced by >2%, or mean scalar <0.85 (allocator too aggressive)

**If FAIL:** Do not deploy. Review regime thresholds for over-sensitivity.

---

### 3. Are regime transitions sparse and intuitive?

**What to check:**
- Review regime transition count in two-pass report
- Target: <20 transitions per year (for multi-year backtest)
- Check for rapid thrashing (same regime entered/exited multiple times in short period)

**Review:**
- [ ] Total transitions <20 per year
- [ ] No thrashing (e.g., NORMAL → ELEVATED → NORMAL within 5 days)
- [ ] Transitions cluster around known stress periods

**Interpretation:**
- ✅ **PASS**: <20 transitions/year, no thrashing, transitions make sense
- ⚠️ **CAUTION**: 20-30 transitions/year, or occasional thrashing
- ❌ **FAIL**: >30 transitions/year, frequent thrashing

**If FAIL:** Do not deploy. Review anti-thrash rules and hysteresis parameters.

---

### 4. Does it correctly flag known stress windows?

**What to check:**
- Review "Top 10 de-risk events" in two-pass report
- Check if allocator triggered during known stress periods:
  - **2020 Q1** (COVID crash: Feb-Mar 2020)
  - **2022** (Fed tightening / inflation: Q1-Q3 2022)
  - Any other obvious stress windows in your sample

**Review:**
- [ ] 2020 Q1 appears in top de-risk events (scalar <0.70)
- [ ] 2022 stress periods appear in top de-risk events
- [ ] No major de-risk events during calm periods (2021, early 2023)

**Interpretation:**
- ✅ **PASS**: Allocator flags 2020 Q1, 2022, no false alarms
- ⚠️ **CAUTION**: Allocator flags stress periods but also some calm periods
- ❌ **FAIL**: Allocator misses 2020 Q1 or 2022, or triggers constantly during calm periods

**If FAIL:** Do not deploy. Review regime thresholds for stress detection sensitivity.

---

## Validation Workflow

### Step 1: Run Two-Pass Audit

```bash
python scripts/diagnostics/run_allocator_two_pass.py \
  --strategy_profile core_v9 \
  --start 2020-01-06 \
  --end 2025-10-31
```

This generates:
- Baseline run (allocator off)
- Scaled run (allocator on with precomputed scalars)
- `two_pass_comparison.md` - Human-readable report

### Step 2: Review Comparison Report

Open `reports/runs/{scaled_run_id}/two_pass_comparison.md`

**Section 1: Performance Metrics**
- Review MaxDD, worst month, worst quarter (Question 1)
- Review CAGR, Sharpe, annualized vol (Question 2)

**Section 2: Allocator Usage Statistics**
- Review % rebalances scaled, mean/min/max scalar (Question 2)

**Section 3: Regime Distribution**
- Review days in each regime (Question 2)
- Review regime transitions (Question 3)

**Section 4: Top De-Risking Events**
- Review top 10 dates with lowest scalars (Question 4)
- Check if 2020 Q1, 2022 appear

### Step 3: Review Regime Timeline

Open `reports/runs/{baseline_run_id}/allocator_regime_v1.csv`

**Spot-check key dates:**

**2020 Q1 (COVID Crash):**
```csv
2020-02-24, ELEVATED  # Market starting to react
2020-02-28, STRESS    # Initial crash
2020-03-09, CRISIS    # Circuit breakers triggered
2020-03-16, CRISIS    # Peak volatility
2020-03-23, CRISIS    # Still in crisis
2020-04-06, STRESS    # Recovery beginning
2020-04-20, ELEVATED  # Downgrade from stress
2020-05-04, NORMAL    # Return to normal
```

**Expected behavior:**
- Enter CRISIS during peak volatility (mid-March 2020)
- Stay in CRISIS for at least 5 days (anti-thrash)
- Gradual downgrade through STRESS → ELEVATED → NORMAL
- Total time in CRISIS: ~2-4 weeks

**2022 (Fed Tightening / Inflation):**
```csv
2022-01-10, ELEVATED  # Inflation concerns rising
2022-02-14, STRESS    # Fed signaling tightening
2022-06-13, STRESS    # Rate hikes ongoing
2022-09-12, STRESS    # Peak inflation concerns
2022-10-03, STRESS    # Continued volatility
2022-11-07, ELEVATED  # Downgrade from stress
2022-12-05, NORMAL    # Return to normal
```

**Expected behavior:**
- Enter STRESS during high inflation / rate hike period
- Stay in STRESS for extended period (months, not days)
- May briefly touch CRISIS during worst drawdowns
- Gradual recovery to NORMAL by late 2022

### Step 4: Review Risk Scalar Timeline

Open `reports/runs/{baseline_run_id}/allocator_risk_v1.csv`

**Spot-check key dates:**

**2020-03-16 (COVID Peak):**
```csv
2020-03-16, 0.30  # CRISIS → 70% de-risk
```

**2022-09-12 (Inflation Peak):**
```csv
2022-09-12, 0.55  # STRESS → 45% de-risk
```

**2021 (Calm Period):**
```csv
2021-06-01, 1.00  # NORMAL → no de-risk
2021-09-01, 1.00  # NORMAL → no de-risk
2021-12-01, 1.00  # NORMAL → no de-risk
```

**Expected behavior:**
- Scalars near 0.30 during CRISIS (2020 Q1)
- Scalars near 0.55-0.70 during STRESS (2022)
- Scalars at 1.00 during NORMAL (2021, early 2023)
- Smooth transitions due to EWMA smoothing (no jumps)

### Step 5: Complete Checklist

Use this checklist to make the go/no-go decision:

```
ALLOCATOR V1 STAGE 6.5 VALIDATION CHECKLIST

Run ID: ___________________________________
Baseline Run: _____________________________
Scaled Run: _______________________________
Date Range: _______________________________
Reviewer: _________________________________
Date: _____________________________________

QUESTION 1: MaxDD Reduction
[ ] MaxDD reduced by at least 2%
[ ] Worst month improved
[ ] Worst quarter improved
Result: PASS / CAUTION / FAIL

QUESTION 2: Returns Preserved in NORMAL
[ ] CAGR reduced by <1%
[ ] Sharpe preserved or improved
[ ] Time in NORMAL: 70-80%
[ ] Mean scalar: 0.90-0.98
Result: PASS / CAUTION / FAIL

QUESTION 3: Regime Transitions Sparse
[ ] <20 transitions per year
[ ] No thrashing
[ ] Transitions cluster around stress periods
Result: PASS / CAUTION / FAIL

QUESTION 4: Known Stress Windows Flagged
[ ] 2020 Q1 in top de-risk events
[ ] 2022 in top de-risk events
[ ] No false alarms during calm periods
Result: PASS / CAUTION / FAIL

OVERALL DECISION
[ ] PASS (deploy to production)
[ ] CAUTION (needs minor review but can deploy)
[ ] FAIL (do not deploy, review thresholds)

Notes:
_____________________________________________
_____________________________________________
_____________________________________________
```

---

## Decision Criteria

### PASS: Deploy to Production

**All 4 questions answered "PASS" or "CAUTION":**
- Lock v1 and deploy
- Do not tune thresholds yet
- Move to production deployment (see `ALLOCATOR_V1_PRODUCTION_MODE.md`)

### FAIL: Do Not Deploy

**Any question answered "FAIL":**
- Do not deploy
- Review specific failure mode
- Adjust thresholds if needed (but avoid over-tuning)
- Re-run validation after changes

---

## What This Review Is NOT

**This is NOT:**
- ❌ Sharpe optimization (we don't care if Sharpe goes from 0.66 → 0.65)
- ❌ Threshold tuning (that's Stage 7, post-deployment)
- ❌ Parameter sweeps or grid search
- ❌ Machine learning or fitting
- ❌ Trying to maximize returns

**This IS:**
- ✅ Qualitative sanity check (does it do what we expect?)
- ✅ Stress window validation (does it protect when it should?)
- ✅ Stability check (does it thrash or behave smoothly?)
- ✅ Go/no-go decision (ship it or fix it)

---

## Common Issues and Fixes

### Issue 1: MaxDD Not Reduced

**Symptoms:**
- MaxDD same or worse than baseline
- Worst month/quarter not improved

**Possible causes:**
- Regime thresholds too conservative (not entering STRESS/CRISIS early enough)
- Risk scalars not aggressive enough (0.85 still too high during stress)

**Fix:**
- Review `REGIME_THRESHOLDS` in `src/allocator/regime_rules_v1.py`
- Consider lowering `DD_ENTER`, `VOL_ACCEL_ENTER`, `CORR_SHOCK_ENTER`
- This is Stage 7 work (threshold tuning)

### Issue 2: Returns Killed in NORMAL

**Symptoms:**
- CAGR reduced by >2%
- Sharpe degraded significantly
- Mean scalar <0.85
- Time in NORMAL <60%

**Possible causes:**
- Regime thresholds too sensitive (entering STRESS too often)
- Anti-thrash rules not working (frequent transitions)

**Fix:**
- Review `REGIME_THRESHOLDS` in `src/allocator/regime_rules_v1.py`
- Consider raising ENTER thresholds or widening hysteresis gap
- Check anti-thrash rule (`MIN_DAYS_IN_REGIME`)

### Issue 3: Regime Thrashing

**Symptoms:**
- >30 transitions per year
- Rapid NORMAL → ELEVATED → NORMAL cycles
- Transitions not clustering around stress periods

**Possible causes:**
- Hysteresis gap too small (ENTER and EXIT thresholds too close)
- Anti-thrash rule not enforced (`MIN_DAYS_IN_REGIME` too low)
- Features too noisy (correlation shock, vol accel)

**Fix:**
- Widen hysteresis gap (increase difference between ENTER and EXIT thresholds)
- Increase `MIN_DAYS_IN_REGIME` from 5 to 7 or 10 days
- Review state feature calculations for noise

### Issue 4: Stress Windows Not Flagged

**Symptoms:**
- 2020 Q1 or 2022 not in top de-risk events
- Scalars at 1.0 during known crisis periods
- Never entering STRESS or CRISIS regime

**Possible causes:**
- Regime thresholds too conservative (not sensitive enough)
- State features not capturing stress signals correctly

**Fix:**
- Review state features during 2020 Q1 (check `allocator_state_v1.csv`)
- Verify features spiking as expected (vol_accel, corr_shock, dd_level)
- Consider lowering ENTER thresholds to be more sensitive

---

## Expected Results (Target Ranges)

**For a well-calibrated Allocator v1 on 2020-2025 canonical window:**

**Performance Impact:**
- MaxDD: -3% to -5% reduction (e.g., -15.32% → -12.00%)
- CAGR: -0.5% to -1.0% reduction (acceptable trade-off)
- Sharpe: +0.05 to +0.15 improvement
- Worst Month: +2% to +4% improvement
- Worst Quarter: +2% to +5% improvement

**Regime Distribution:**
- NORMAL: 70-80% of days
- ELEVATED: 10-20% of days
- STRESS: 5-10% of days
- CRISIS: 1-5% of days

**Scalar Statistics:**
- Mean: 0.92-0.96
- Min: 0.30-0.40 (during 2020 Q1)
- Max: 1.00
- % Rebalances scaled: 20-30%

**Transitions:**
- Total: 10-20 per year
- Most transitions: NORMAL ↔ ELEVATED (small regime changes)
- Few transitions: ELEVATED → STRESS → CRISIS (major regime changes)

**Top De-Risk Events:**
- #1: 2020-03-16 (COVID peak, scalar ~0.30)
- #2-5: 2020 Q1 dates (scalar 0.30-0.60)
- #6-10: 2022 dates (scalar 0.55-0.70)

**If your results are in these ranges, PASS and deploy.**

---

## Sign-Off Template

```
ALLOCATOR V1 — STAGE 6.5 VALIDATION SIGN-OFF

Project: Futures-Six
Allocator Version: v1.0
Validation Date: ___________
Reviewer: __________________

VALIDATION RESULTS:
Question 1 (MaxDD Reduction): PASS / CAUTION / FAIL
Question 2 (NORMAL Returns): PASS / CAUTION / FAIL
Question 3 (Sparse Transitions): PASS / CAUTION / FAIL
Question 4 (Stress Flagging): PASS / CAUTION / FAIL

OVERALL DECISION: PASS / FAIL

DEPLOYMENT DECISION:
[ ] APPROVED for production deployment
[ ] NOT APPROVED, requires threshold review

Signature: _______________________
Date: ____________________________

Notes:
__________________________________________
__________________________________________
__________________________________________
```

---

## Next Steps After Validation

### If PASS → Deploy to Production

1. Lock baseline run ID in production config
2. Update `configs/strategies_production.yaml`:
   ```yaml
   allocator_v1:
     enabled: true
     mode: "precomputed"
     precomputed_run_id: "<validated_baseline_run_id>"
   ```
3. Deploy to production
4. Monitor allocator behavior in live trading
5. **Do NOT tune thresholds yet** (Stage 7 is post-deployment)

### If FAIL → Review and Retry

1. Identify specific failure mode (MaxDD, returns, transitions, or stress detection)
2. Review corresponding thresholds in `src/allocator/regime_rules_v1.py`
3. Make minimal adjustments (avoid over-tuning)
4. Re-run two-pass audit
5. Re-run Stage 6.5 validation
6. Iterate until PASS

---

## Key Principle

**"Mostly yes" is good enough.**

Allocator v1 does not need to be perfect. It needs to:
- Reduce tail risk (MaxDD)
- Avoid killing returns (CAGR)
- Behave sensibly (transitions, stress flagging)

If it does these things "mostly well," **lock it and deploy.**

Stage 7 (threshold tuning) and Stage 8 (convexity overlays) are post-deployment enhancements.

**Do not over-optimize before going live.**

---

**End of Document**

