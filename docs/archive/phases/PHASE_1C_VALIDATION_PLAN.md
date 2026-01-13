# Phase 1C Validation Plan

## Current Status

**Running:** A/B backtests for 2024 period (2024-01-01 to 2024-12-31)

**Scenarios:**
1. Baseline (no RT, no allocator)
2. RT only (Risk Targeting enabled)
3. RT + Alloc-H (Risk Targeting + Allocator-H)

## Validation Steps

### Step 1: Run A/B Backtests ✅ (In Progress)

**Command:**
```bash
python scripts/diagnostics/run_phase1c_ab_backtests.py \
    --strategy_profile core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro \
    --start 2024-01-01 \
    --end 2024-12-31
```

**Expected Output:**
- 3 backtest runs (Baseline, RT only, RT + Alloc-H)
- Comparison report in `reports/diagnostics/phase1c_ab/phase1c_ab_comparison.md`
- JSON summary in `reports/diagnostics/phase1c_ab/phase1c_ab_comparison.json`

---

### Step 2: Validate Artifacts

**Command:**
```bash
# For each run_id:
python scripts/diagnostics/validate_phase1c_artifacts.py --run_id <run_id>
```

**Checklist:**

#### Risk Targeting Artifacts:
- [ ] `risk_targeting/params.json` exists and is correct
- [ ] `risk_targeting/leverage_series.csv` has one row per trading day
- [ ] `risk_targeting/realized_vol.csv` has one row per trading day
- [ ] `risk_targeting/weights_pre_risk_targeting.csv` exists
- [ ] `risk_targeting/weights_post_risk_targeting.csv` exists
- [ ] **Sanity check**: `post_weights ≈ pre_weights * leverage_scalar(date)` for sample dates

#### Allocator Artifacts (RT + Alloc-H only):
- [ ] `allocator/regime_series.csv` exists
- [ ] `allocator/multiplier_series.csv` exists
- [ ] Multiplier series shows infrequent changes (piecewise constant)
- [ ] % days allocator active is low (< 10% for Alloc-H)

---

### Step 3: Validate Allocator Behavior

**Metrics to Check:**

1. **Activity %:**
   - Alloc-H should have < 10% active days
   - If Alloc-H is active > 50% of days → red flag (thresholds too tight)

2. **Monotonicity (if running M/L):**
   - % active days: H < M < L
   - Avg multiplier: H > M > L (closer to 1.0)

3. **Event Table:**
   - Top 10 drawdown days should show allocator active during stress
   - Multiplier < 1.0 during stress periods
   - Not constantly suppressing (multiplier = 1.0 most of the time)

---

### Step 4: Review A/B Results

**Expected Behavior:**

#### Baseline → RT only:
- ✅ Vol should move toward target (20%)
- ✅ Leverage should be meaningful but bounded (< leverage_cap)
- ✅ Avg leverage should be < cap (unless target_vol is aggressive)

#### RT only → RT + Alloc-H:
- ✅ Sharpe might dip slightly (acceptable)
- ✅ Drawdown tails should improve
- ✅ % days allocator active should be low (< 10%)
- ✅ Event table shows allocator biting mainly in stress

#### RT + Alloc-M/L (if included):
- ✅ Clear monotonic progression:
  - More active days (M > H, L > M)
  - Lower worst month / MaxDD
  - Lower returns (expected trade-off)

**Red Flags:**
- ❌ Alloc-H active > 50% of days
- ❌ No improvement in drawdown tails
- ❌ Leverage constantly at cap (RT not working)
- ❌ Artifacts missing or incorrect

---

## Validation Commands

```bash
# 1. Check if backtests completed
ls reports/runs/*2024*

# 2. Validate artifacts for each run
python scripts/diagnostics/validate_phase1c_artifacts.py \
    --run_id core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro_baseline_2024-01-01_2024-12-31

python scripts/diagnostics/validate_phase1c_artifacts.py \
    --run_id core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro_rt_only_2024-01-01_2024-12-31

python scripts/diagnostics/validate_phase1c_artifacts.py \
    --run_id core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro_rt_alloc_h_2024-01-01_2024-12-31

# 3. Check A/B report
cat reports/diagnostics/phase1c_ab/phase1c_ab_comparison.md
```

---

## Next Steps After Validation

1. If all validations pass → Run full period (2020-2025)
2. If issues found → Fix and re-run
3. Once full period passes → Phase 1C is DONE ✅

---

**Last Updated**: 2026-01-09

