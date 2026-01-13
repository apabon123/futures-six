# Risk Targeting Bug - CRITICAL FINDINGS

**Date:** 2026-01-09  
**Status:** 🚨 BUG IDENTIFIED - AWAITING FIX

---

## Summary

The Risk Targeting layer IS working correctly for the **backtest**, but the **artifacts are broken**.

---

## Evidence

### What RT Logs Show (CORRECT):
```
2024-01-05: vol=5.00%, leverage=4.00x, gross_before=6.14, gross_after=4.00
```

### What Artifacts Show (WRONG):
```
leverage_series.csv:     leverage=4.0 ✅ CORRECT
weights_pre:             only 1 asset (ZT) with gross=1.5 ❌ WRONG (should be 6.14)
weights_post:            only 1 asset with gross=0.98 ❌ WRONG (should be 4.0)
```

### What Backtest Shows (CORRECT):
```
ExecSim turnover=4.000 ✅ Confirms backtest used gross=4.0 correctly
```

---

## The Bug

**The artifacts are missing most assets!**

- RT receives weights with **13 assets, gross=6.14**
- RT scales correctly to **gross=4.0**  
- RT logs correctly: `gross_before=6.14, gross_after=4.00`
- ❌ **BUT** artifacts only show **1 asset** for that date!

---

## Root Cause

The `_write_artifacts` method in `risk_targeting.py` is being passed a **filtered** version of the weights Series, not the full Series.

Most likely cause: The `weights` parameter passed to `_write_artifacts` has already been filtered to only include "active" weights somewhere in the code flow.

---

## Impact Assessment

### ✅ GOOD NEWS:
1. **Backtest results are CORRECT** - RT is working properly in the main loop
2. **Leverage calculation is CORRECT** - all logic is sound
3. **Returns are CORRECT** - portfolio performance metrics are valid

### ❌ BAD NEWS:
1. **Artifacts are UNUSABLE** - weights_pre/post files are incomplete and misleading
2. **Vol gap analysis was WRONG** - we concluded RT wasn't working based on broken artifacts
3. **Phase 1C validation INCOMPLETE** - artifact integrity check failed

---

## Why This Happened

Looking at `scale_weights`:

```python
def scale_weights(self, weights, returns, date, cov_matrix=None):
    # ... compute vol, leverage ...
    
    # Write artifacts
    if self.artifact_writer is not None:
        self._write_artifacts(date, current_vol, leverage, weights, scaled_weights)
    
    return scaled_weights
```

The `weights` parameter passed to `_write_artifacts` is the ORIGINAL `weights` parameter passed to `scale_weights`. 

**HYPOTHESIS:** Somewhere in ExecSim, the weights Series is being filtered BEFORE being passed to RT, but the filtering is not reflected in the artifacts.

**OR:** There's a bug in `_write_artifacts` where it's only writing non-zero weights, and most weights are effectively zero due to rounding or thresholding.

---

## Next Steps

1. **Immediate:** Add debug logging in `_write_artifacts` to see what it receives:
   ```python
   def _write_artifacts(self, date, current_vol, leverage, weights_pre, weights_post):
       logger.info(f"[RT Artifacts] {date}: weights_pre has {len(weights_pre)} assets, gross={weights_pre.abs().sum():.2f}")
       logger.info(f"[RT Artifacts] {date}: weights_post has {len(weights_post)} assets, gross={weights_post.abs().sum():.2f}")
       # ... rest of method ...
   ```

2. **Fix:** Ensure `_write_artifacts` receives and writes the FULL weights Series, not a filtered version.

3. **Re-run:** Re-run A/B backtests with fixed artifact writing.

4. **Validate:** Verify that artifacts now match logs (gross_pre=6.14, gross_post=4.0 for 2024-01-05).

---

## Revised Phase 1C Status

❌ **Phase 1C is NOT complete** due to artifact bug

**What we learned:**
- RT layer works correctly ✅
- Allocator profiles work correctly ✅  
- Contract tests pass ✅
- ❌ **BUT:** Artifact integrity is broken - artifacts don't reflect actual weights used in backtest

**Timeline:**
- Fix artifact bug: 30 mins
- Re-run A/B backtests: 1 hour  
- Validate artifact integrity: 15 mins
- **THEN** Phase 1C done

---

## Action: Fix the Bug

The fix is likely one of these:

**Option 1:** The weights Series has zero-weight assets that are being skipped. Fix by writing ALL assets:
```python
# In _write_artifacts, write all assets including zeros:
all_assets = weights_pre.index  # Don't filter
```

**Option 2:** ExecSim is passing a filtered Series. Fix by passing the full Series:
```python
# In ExecSim, before calling RT:
weights_full = weights_raw.copy()  # Keep all assets
weights_scaled = risk_targeting.scale_weights(weights_full, ...)
```

**Option 3:** The artifact writer is silently dropping rows. Debug by logging before and after write.

---

**Assigned to:** AI Agent  
**Priority:** 🚨 CRITICAL - blocks Phase 1C completion

