# Allocator v1 — Production Mode Specification

**Version:** v1.0  
**Status:** Production-Ready  
**Date:** December 2024

---

## Production Mode: `precomputed` (Locked)

**For institutional deployment, Allocator v1 uses `mode="precomputed"` as the production-safe default.**

---

## Mode Comparison

### ✅ `mode="precomputed"` (PRODUCTION DEFAULT)

**Status:** Production-safe, institutionally validated

**How it works:**
1. Baseline run (Pass 1) computes allocator artifacts with `enabled=false`
2. Production run (Pass 2) loads `allocator_risk_v1_applied.csv` from baseline
3. Scalars applied with 1-rebalance lag at each rebalance date

**Advantages:**
- ✅ No warmup period issues (baseline run has full history)
- ✅ No circular dependency (scalars computed from baseline portfolio)
- ✅ Fully deterministic (same baseline → same scalars → same results)
- ✅ Complete audit trail (baseline vs scaled comparison)
- ✅ Safe for live deployment

**Use cases:**
- Production trading
- Two-pass validation
- Post-deployment monitoring
- Audited backtesting

**Configuration:**
```yaml
allocator_v1:
  enabled: true
  mode: "precomputed"
  precomputed_run_id: "<validated_baseline_run_id>"
  precomputed_scalar_filename: "allocator_risk_v1_applied.csv"
  apply_missing_scalar_as: 1.0
```

---

### 🔬 `mode="compute"` (RESEARCH ONLY)

**Status:** Research-only, has known issues

**How it works:**
1. Computes state/regime/risk on-the-fly during backtest
2. Applies scalars at each rebalance date

**Issues:**
- ⚠️ Warmup period: First ~60 days produce empty state (no allocator output)
- ⚠️ Not recommended until Stage 9 (incremental state computation)

**Use cases:**
- Research and development only
- Testing new allocator logic
- NOT for production

**Configuration:**
```yaml
allocator_v1:
  enabled: true
  mode: "compute"
```

**Note:** Do not use `mode="compute"` in production until Stage 9 is complete.

---

### 📊 `mode="off"` (BASELINE / RESEARCH)

**Status:** Safe for all use cases

**How it works:**
1. Computes all allocator artifacts (state, regime, risk)
2. Does NOT apply scalars to weights
3. Portfolio runs unscaled (baseline)

**Use cases:**
- Baseline runs for two-pass audit
- Artifact generation for analysis
- Research and development
- Generating `allocator_risk_v1_applied.csv` for Pass 2

**Configuration:**
```yaml
allocator_v1:
  enabled: false  # Artifacts computed but not applied
```

---

## Production Deployment Workflow

### Step 1: Baseline Run (Pass 1)

Generate validated baseline artifacts:

```bash
python run_strategy.py \
  --strategy_profile core_v9 \
  --start 2020-01-06 \
  --end 2025-10-31 \
  --run_id baseline_v1_production
```

**Configuration (default):**
```yaml
allocator_v1:
  enabled: false  # mode="off" implicitly
```

**Critical artifacts produced:**
- `allocator_risk_v1_applied.csv` - Lagged scalars ready for application
- `allocator_state_v1.csv` - State features (for review)
- `allocator_regime_v1.csv` - Regime classifications (for review)

### Step 2: Validation Review (Stage 6.5)

Before deploying to production, validate baseline artifacts:

**Qualitative Review:**
- [ ] Review regime transitions (sparse and intuitive?)
- [ ] Check top de-risk events (align with known stress periods?)
- [ ] Inspect state features (no data quality issues?)
- [ ] Verify effective start date (reasonable given rolling windows?)

**See:** `docs/ALLOCATOR_V1_STAGE_6_5_VALIDATION.md` for detailed checklist

### Step 3: Two-Pass Audit

Run two-pass comparison to validate allocator impact:

```bash
python scripts/diagnostics/run_allocator_two_pass.py \
  --strategy_profile core_v9 \
  --start 2020-01-06 \
  --end 2025-10-31 \
  --baseline_run_id baseline_v1_production
```

**Review comparison report:**
- MaxDD reduction (target: 2-5% improvement)
- Worst month/quarter improvement
- Sharpe preserved or improved
- De-risk events align with stress periods

### Step 4: Production Deployment

If validation passes, deploy to production with precomputed mode:

```yaml
# configs/strategies_production.yaml
allocator_v1:
  enabled: true
  mode: "precomputed"
  precomputed_run_id: "baseline_v1_production"
  precomputed_scalar_filename: "allocator_risk_v1_applied.csv"
  apply_missing_scalar_as: 1.0
```

```bash
python run_strategy.py --config configs/strategies_production.yaml
```

---

## Why `precomputed` is Production-Safe

### 1. No Warmup Period Issues

**Problem with `compute` mode:**
- State features require 60-day rolling windows
- First ~60 days have insufficient history
- Allocator produces empty state → no risk scalars → defaults to 1.0

**Solution with `precomputed` mode:**
- Baseline run has full history (e.g., 2020-2025)
- `allocator_risk_v1_applied.csv` starts after warmup period naturally
- No empty state, no defaults needed

### 2. No Circular Dependency

**Potential issue:**
- Risk scalars depend on portfolio returns
- Portfolio returns depend on weights
- Weights depend on risk scalars
- Circular!

**How `precomputed` avoids this:**
- Baseline run computes scalars from **unscaled portfolio** returns
- Production run applies these **fixed** scalars to weights
- No circularity: scalars computed from one portfolio, applied to another

### 3. Fully Deterministic

**`precomputed` guarantees:**
- Same baseline run → same `allocator_risk_v1_applied.csv`
- Same scalar file → same scaled portfolio
- Rerun produces identical results (bit-for-bit)

**Audit trail:**
- Baseline artifacts frozen and versioned
- Production artifacts reference specific baseline run ID
- Complete audit trail from baseline → scalars → scaled portfolio

### 4. Institutional Standard

**This is how institutional allocators work:**
1. Compute risk metrics from historical data
2. Apply computed metrics to forward-looking decisions
3. Avoid on-the-fly computation in live systems
4. Maintain complete audit trail

**`precomputed` mode follows this pattern exactly.**

---

## Future: When to Use `compute` Mode

**Stage 9 will resolve warmup issues by implementing:**
- True incremental state computation (no rolling windows from scratch)
- Efficient online updates (state[t] = f(state[t-1], new_data))
- Proper handling of early dates (explicit warmup period handling)

**Once Stage 9 is complete:**
- `compute` mode will be safe for production
- Still recommend `precomputed` for audited backtesting
- `compute` mode useful for live trading (no baseline run needed)

**Until Stage 9:**
- Use `precomputed` for all production deployments
- Use `mode="off"` for baseline generation
- Use `compute` only for allocator logic research

---

## Configuration Examples

### Production (Recommended)

```yaml
allocator_v1:
  enabled: true
  mode: "precomputed"
  precomputed_run_id: "baseline_core_v9_2020_2025"
  precomputed_scalar_filename: "allocator_risk_v1_applied.csv"
  apply_missing_scalar_as: 1.0
  state_version: "v1.0"
  regime_version: "v1.0"
  risk_version: "v1.0"
```

### Baseline Generation

```yaml
allocator_v1:
  enabled: false  # Computes artifacts but doesn't apply
  state_version: "v1.0"
  regime_version: "v1.0"
  risk_version: "v1.0"
```

### Research (Off)

```yaml
allocator_v1:
  enabled: false
```

### Research (Compute Mode - Use with Caution)

```yaml
allocator_v1:
  enabled: true
  mode: "compute"
  # Note: First ~60 days will have no allocator output
```

---

## Decision: Precomputed as Default

**Philosophical Decision (Institutional Standard):**

For Allocator v1, `mode="precomputed"` is the production-safe default because:

1. **Safety first**: Avoids warmup edge cases, circularity, non-determinism
2. **Audit-friendly**: Complete baseline → scaled comparison trail
3. **Battle-tested**: Institutional allocators use this pattern
4. **Stage 9 not required**: Can deploy v1 now without fixing `compute` mode

**We do not need to "fix" `compute` mode to call v1 production-ready.**

**Stage 9 (incremental state) is a future enhancement, not a v1 blocker.**

---

**End of Document**

