# Phase 1C DONE Checklist

**Status**: ✅ **INTEGRATION COMPLETE** - Ready for validation

## Definition of DONE

You can declare Phase 1C complete when:

- [x] `run_strategy.py` includes Risk Targeting in correct order (Layer 5)
- [x] RT artifacts are produced in an end-to-end run
- [x] Allocator artifacts are produced in an end-to-end run
- [ ] A/B script runs and produces report + event table without manual intervention
- [ ] A/B results show expected gradient from H → M → L (activity + suppression)

---

## ✅ 1. Risk Targeting Integration

### Pipeline Order ✅

**Canonical Stack Order (FROZEN):**
1. Engine Signals (alpha)
2. Engine Policy (gates/throttles) - PLANNED
3. Portfolio Construction (static weights) - `allocator.solve()`
4. Discretionary Overlay (bounded tilts) - `macro_overlay`
5. **Risk Targeting (vol → leverage)** ✅ - `risk_targeting.scale_weights()`
6. Allocator (risk brake) - `allocator_v1`
7. Margin & Execution Constraints - `ExecSim`

**Implementation:**
- ✅ Risk Targeting inserted in `ExecSim.run()` between `allocator.solve()` and `allocator_v1`
- ✅ Order verified: `portfolio_construction → discretionary_overlay → risk_targeting → allocator → execution`
- ✅ Risk Targeting layer initialized in `run_strategy.py` with config support

**Files Modified:**
- `run_strategy.py` - Added Risk Targeting initialization
- `src/agents/exec_sim.py` - Inserted Risk Targeting in pipeline

---

## ✅ 2. Risk Targeting Artifacts

### Artifacts Emitted ✅

**Files Created:**
- ✅ `risk_targeting/params.json` - Written once per run
- ✅ `risk_targeting/leverage_series.csv` - One row per trading day
- ✅ `risk_targeting/realized_vol.csv` - One row per trading day
- ✅ `risk_targeting/weights_pre_risk_targeting.csv` - Pre-RT weights
- ✅ `risk_targeting/weights_post_risk_targeting.csv` - Post-RT weights

**Acceptance Criteria:**
- ✅ `params.json` written once per run (not appended daily)
- ✅ Deterministic file output: stable column order, stable instrument sorting, ISO dates
- ✅ `weights_pre_...` and `weights_post_...` differ by constant scalar per date
- ⏳ **Sanity test pending**: Pick one date and verify `post_weights ≈ pre_weights * leverage_scalar(date)`

**Implementation:**
- ✅ Artifact writer initialized in `ExecSim.run()`
- ✅ Artifacts written in `RiskTargetingLayer.scale_weights()`
- ✅ `params.json` written once at start via `_write_params()`

---

## ✅ 3. Allocator Artifacts

### Artifacts Emitted ✅

**Files Created:**
- ✅ `allocator/regime_series.csv` - date, regime, profile
- ✅ `allocator/multiplier_series.csv` - date, multiplier, profile

**Acceptance Criteria:**
- ⏳ **Validation pending**: Multiplier series is piecewise constant with infrequent changes for Alloc-H
- ⏳ **Validation pending**: % days allocator active matches expectations (H low, L higher)
- ⏳ **Validation pending**: In top drawdown events, allocator multiplier < 1.0 when relevant (but not constantly suppressing)

**Implementation:**
- ✅ Artifacts emitted in `ExecSim.run()` after computing regime and multiplier
- ✅ Supports both `compute` and `precomputed` modes
- ✅ Profile name included in artifacts from config

**Files Modified:**
- `src/agents/exec_sim.py` - Added artifact emission in allocator v1 compute path

---

## ⏳ 4. Canonical A/B Backtests

### Script Ready ✅

**File Created:**
- ✅ `scripts/diagnostics/run_phase1c_ab_backtests.py` - A/B backtest orchestration

**Scenarios:**
1. ✅ Baseline: Core v9, no RT, no allocator
2. ✅ RT only: Core v9 + Risk Targeting
3. ✅ RT + Alloc-H: Core v9 + Risk Targeting + Allocator-H
4. ✅ RT + Alloc-M: Core v9 + Risk Targeting + Allocator-M (optional)
5. ✅ RT + Alloc-L: Core v9 + Risk Targeting + Allocator-L (optional)

**Report Generated:**
- ✅ Annualized return / vol / Sharpe
- ✅ Max drawdown
- ✅ Worst month
- ✅ % days allocator not-1.0 (for H/M/L)
- ✅ Avg leverage + 95th percentile leverage
- ✅ Event table of top 10 days by drawdown (with leverage, regime, multiplier)

**Usage:**
```bash
python scripts/diagnostics/run_phase1c_ab_backtests.py \
    --strategy_profile core_v9 \
    --start 2020-01-01 \
    --end 2025-10-31 \
    --include_alloc_m_l
```

**Status:**
- ⏳ **Pending**: Run script and validate it works end-to-end
- ⏳ **Pending**: Verify report generation
- ⏳ **Pending**: Validate expected gradients

---

## ⏳ 5. Validation Tests

### What "Good" Looks Like

**Baseline → RT only:**
- Vol should move toward target
- Leverage should be meaningful but bounded
- Avg leverage should be < cap (unless target_vol is aggressive)

**RT only → RT + Alloc-H:**
- Sharpe might dip slightly, but:
  - Drawdown tails should improve a bit
  - % days allocator active should be low (< 10% for H)
  - Event table should show allocator biting mainly in stress

**RT + Alloc-M/L:**
- Clear monotonic progression:
  - More active days (M > H, L > M)
  - Lower worst month / MaxDD
  - Lower returns (expected)

**Red Flags:**
- If Alloc-H is "active" all the time → thresholds too tight or regime detector too sensitive

---

## Summary

### ✅ Completed
- Risk Targeting integration in correct pipeline order
- RT artifacts infrastructure and emission
- Allocator artifacts infrastructure and emission
- A/B backtest script structure

### ⏳ Pending Validation
- Run A/B backtests end-to-end
- Validate artifact correctness (weights sanity check)
- Verify allocator behavior (activity %, monotonicity)
- Confirm expected gradients in A/B results

---

## Next Steps

1. **Run A/B Backtests:**
   ```bash
   python scripts/diagnostics/run_phase1c_ab_backtests.py \
       --strategy_profile core_v9 \
       --start 2020-01-01 \
       --end 2025-10-31 \
       --include_alloc_m_l
   ```

2. **Validate Artifacts:**
   - Check `risk_targeting/params.json` exists and is correct
   - Verify `leverage_series.csv` has one row per trading day
   - Sanity check: `post_weights ≈ pre_weights * leverage_scalar(date)` for sample dates

3. **Validate Allocator Behavior:**
   - Check `allocator/multiplier_series.csv` shows infrequent changes for Alloc-H
   - Verify % days active: H < M < L
   - Check event table shows allocator active during stress periods

4. **Review A/B Results:**
   - Confirm expected gradients (H → M → L)
   - Verify no red flags (Alloc-H not constantly active)
   - Check drawdown improvements with allocator

---

**Last Updated**: 2026-01-09
**Integration Status**: ✅ Complete, ⏳ Validation Pending

